"""
MetaGoHealth QA Engine — GitHub Actions version.

Same pipeline as the old Streamlit app, but fully unattended:
  1. Scan the Drive inbox folder for CSVs
  2. For each CSV, process every call (download -> ffprobe -> GCS upload ->
     Gemini transcript -> Gemini scoring -> append to Sheet)
  3. Move the CSV to the Drive "processed" subfolder when done

Auth: same service account as drive-sync-job, passed in via the
GCP_SERVICE_ACCOUNT_JSON env var (GitHub secret GCP_SA_KEY).

Re-running is safe: the Sheet's "File Name" column is used to dedupe, so if
a run gets interrupted partway, dragging the same CSV back into the inbox
just picks up where it left off.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import gspread
import pandas as pd
import requests
import vertexai
from google.cloud import storage
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from vertexai.generative_models import GenerativeModel, Part

# =============================================================================
# CONFIG — confirmed infra, same values as the Streamlit app
# =============================================================================
PROJECT_ID  = "gtm-5wnb34pj"
REGION      = "asia-south1"
BUCKET_NAME = "metago-qa-oudits"
SHEET_ID    = "1r8dMXKllejmvaVcy-4whH_fX-7lZ2hD6Mf5PtYVRJNQ"
MODEL_NAME  = "gemini-2.5-flash"
MIN_DURATION_SEC = 90

DRIVE_INBOX_FOLDER_ID     = "1ZsHA5EUVqhDY5xeVNWJ2DkKLkAfAcCuo"   # 1799-Audit-Inbox
DRIVE_PROCESSED_FOLDER_ID = "1346tO8l_RzmvwVyDfNd67jTB7bHLnCC-"   # 1799-Audit-Inbox/processed

BRAIN_DIR     = os.path.join(os.path.dirname(__file__), "brain_files")
LEGEND_FILE   = os.path.join(BRAIN_DIR, "legend.csv")
OBJ_FILE      = os.path.join(BRAIN_DIR, "objections.txt")
SCRIPT_FILE   = os.path.join(BRAIN_DIR, "agent_script.csv")
FOLLOWUP_FILE = os.path.join(BRAIN_DIR, "follow_up.csv")
BEHAVIOR_FILE = os.path.join(BRAIN_DIR, "behavior.txt")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# BRAIN FILE LOADING (unchanged from the Streamlit app)
# =============================================================================

def read_text_robust(path):
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} with any known encoding.")


def read_csv_robust(source):
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {source} with any known encoding.")


def load_brain():
    legend_df = read_csv_robust(LEGEND_FILE)
    obj_text = read_text_robust(OBJ_FILE)
    behavior_text = read_text_robust(BEHAVIOR_FILE)

    script_df = read_csv_robust(SCRIPT_FILE)
    script_text = "\n".join(script_df.iloc[:, 0].dropna().astype(str).tolist())

    followup_df = read_csv_robust(FOLLOWUP_FILE)
    followup_lines = followup_df.iloc[:, 0].dropna().astype(str).tolist()
    followup_text = "\n".join(f"- {line}" for line in followup_lines)

    label_cols = ["A", "B", "C", "D", "E"]
    blocks = []
    output_cols = []

    for _, row in legend_df.iterrows():
        pid = str(row["Parameter_ID"]).strip()
        name = str(row["Parameter_Name"]).strip()
        scoring_type = str(row["Scoring_Type"]).strip()
        desc = str(row["Description"]).strip()
        na_when = row.get("Score_NA_When")
        out_col = str(row["Output_Column_Name"]).strip()
        out_reason_col = str(row["Output_Reason_Column_Name"]).strip()

        label_lines = []
        for L in label_cols:
            lab = row.get(f"Label_{L}")
            cond = row.get(f"Label_{L}_When")
            if pd.notna(lab) and str(lab).strip() and str(lab).strip().upper() != "NA":
                label_lines.append(f"    * {str(lab).strip()} -> {str(cond).strip()}")

        block = f"[{pid}] {name} | Type: {scoring_type}\n  {desc}\n" + "\n".join(label_lines)
        if pd.notna(na_when) and str(na_when).strip() and str(na_when).strip().upper() != "NA":
            block += f"\n    Mark NA when: {str(na_when).strip()}"

        blocks.append(block)
        output_cols.append((out_col, out_reason_col))

    rubric_text = "\n\n".join(blocks)

    json_lines = []
    for out_col, out_reason_col in output_cols:
        json_lines.append(f'  "{out_col}": "<label or numeric score or Pass/Fail or NA>"')
        json_lines.append(f'  "{out_reason_col}": "<1-2 sentence justification, cite a moment/timestamp if relevant>"')
    for col, instr in [
        ("Customer Name", "Extract from transcript or write Unknown."),
        ("Overall area of opportunity", "2-3 sentence synthesis of the biggest conversion gap on this call."),
        ("Primary Objection", "Main reason the customer hesitated, or None if converted."),
        ("Reason for Non-Conversion", "Why the call did not convert, or Converted if it did."),
        ("Coaching Tip", "One specific tip referencing the exact low-scoring parameter ID and moment it failed."),
    ]:
        json_lines.append(f'  "{col}": "{instr}"')

    json_template = "{\n" + ",\n".join(json_lines) + "\n}"

    meta_cols = ["Processed Date", "Call Date", "Agent Name", "Customer Name", "File Name"]
    computed_cols = ["Call Duration (sec)", "Call Duration Flag", "Conversion Outcome"]
    coaching_cols = ["Overall area of opportunity", "Primary Objection",
                      "Reason for Non-Conversion", "Coaching Tip", "Full Transcript"]
    dynamic_cols = []
    for out_col, out_reason_col in output_cols:
        dynamic_cols.append(out_col)
        dynamic_cols.append(out_reason_col)

    final_cols = meta_cols + computed_cols + dynamic_cols + coaching_cols

    return {
        "rubric_text": rubric_text,
        "obj_text": obj_text,
        "script_text": script_text,
        "followup_text": followup_text,
        "behavior_text": behavior_text,
        "json_template": json_template,
        "final_cols": final_cols,
    }


# =============================================================================
# GCP / DRIVE CLIENTS
# =============================================================================

def get_clients():
    key_dict = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = service_account.Credentials.from_service_account_info(key_dict, scopes=scopes)

    gcs_client = storage.Client(credentials=creds, project=PROJECT_ID)
    vertexai.init(project=PROJECT_ID, location=REGION, credentials=creds)
    ai_model = GenerativeModel(MODEL_NAME)
    bucket = gcs_client.bucket(BUCKET_NAME)

    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(SHEET_ID).sheet1

    drive_service = build("drive", "v3", credentials=creds)

    return ai_model, bucket, worksheet, drive_service


def get_existing_filenames(worksheet, final_cols):
    try:
        records = worksheet.get_all_records(expected_headers=final_cols)
        return {r.get("File Name", "") for r in records if r.get("File Name")}
    except Exception:
        return set()


def ensure_headers(worksheet, final_cols):
    first_row = worksheet.row_values(1)
    if first_row != final_cols:
        worksheet.clear()
        worksheet.append_row(final_cols, value_input_option="USER_ENTERED")


# =============================================================================
# DRIVE HELPERS
# =============================================================================

def list_inbox_csvs(drive_service):
    """CSVs sitting directly in the inbox folder (not the processed subfolder).

    supportsAllDrives / includeItemsFromAllDrives are required because
    1799-Audit-Inbox lives inside a Shared Drive, not My Drive. Without
    these, the API silently returns zero results with no error.
    """
    query = (
        f"'{DRIVE_INBOX_FOLDER_ID}' in parents "
        "and trashed = false "
        "and (mimeType = 'text/csv' or name contains '.csv')"
    )
    results = drive_service.files().list(
        q=query,
        fields="files(id, name)",
        pageSize=100,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        corpora="allDrives",
    ).execute()
    return results.get("files", [])


def download_drive_file(drive_service, file_id):
    request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf


def move_to_processed(drive_service, file_id):
    drive_service.files().update(
        fileId=file_id,
        addParents=DRIVE_PROCESSED_FOLDER_ID,
        removeParents=DRIVE_INBOX_FOLDER_ID,
        fields="id, parents",
        supportsAllDrives=True,
    ).execute()


# =============================================================================
# CALL PROCESSING HELPERS (unchanged from the Streamlit app)
# =============================================================================

def download_mp3(url, dest_path):
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "audio/mpeg,*/*;q=0.8"}
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, headers=headers, timeout=30)
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception:
            time.sleep(3)
    return False


def get_duration_seconds(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=20,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def upload_to_gcs(bucket, local_path, blob_name):
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return blob


def get_transcript(ai_model, audio_bytes, mime_type="audio/mpeg"):
    prompt = (
        "Transcribe this call verbatim, in the original language mix (English/Hindi/Marathi as spoken). "
        "Label each turn as Agent: or Customer:. Include approximate timestamps in [MM:SS] format "
        "at the start of each turn if you can infer them from pacing. Do not summarize or translate."
    )
    part = Part.from_data(data=audio_bytes, mime_type=mime_type)
    response = call_with_backoff(ai_model.generate_content, [prompt, part])
    return response.text.strip()


def call_with_backoff(func, *args, max_attempts=5, **kwargs):
    """Retries with exponential backoff on rate-limit / transient errors."""
    delay = 5
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            transient = any(k in msg for k in ["429", "rate", "resourceexhausted", "quota", "503", "unavailable"])
            if not transient or attempt == max_attempts - 1:
                raise
            log(f"  transient error, retrying in {delay}s: {e}")
            time.sleep(delay)
            delay = min(delay * 2, 60)


def score_call(ai_model, transcript, brain, agent_name, call_date, file_name, duration_sec):
    system_prompt = f"""You are an expert QA auditor for MetaGoHealth's sales calls (GLP-1 weight management program, 1799/1000 assessment fee).

--- SCORING RUBRIC ---
{brain['rubric_text']}

--- OBJECTION HANDLING REFERENCE (correct answers) ---
{brain['obj_text']}

--- APPROVED AGENT SCRIPT (reference for expected structure) ---
{brain['script_text']}

--- FOLLOW-UP CALL DETECTION ---
If the transcript matches a follow-up scenario, rate ONLY communication-related parameters normally;
mark all other parameters NA. Follow-up indicator phrases:
{brain['followup_text']}

--- BEHAVIOR & PROFESSIONALISM GUIDELINES ---
{brain['behavior_text']}

--- CALL METADATA ---
Agent Name: {agent_name}
Call Date: {call_date}
File Name: {file_name}
Call Duration: {duration_sec:.0f} seconds

--- TRANSCRIPT ---
{transcript}

--- INSTRUCTIONS ---
Score every parameter in the rubric above using the exact Output_Column_Name and Output_Reason_Column_Name keys.
Respect every "Mark NA when" condition and every cross-parameter dependency stated in the rubric
(e.g. some parameters are only scored if a specific customer signal was present in another parameter).
Respond ONLY with a single valid JSON object, no markdown fences, no preamble, matching this exact shape:

{brain['json_template']}
"""
    response = call_with_backoff(ai_model.generate_content, system_prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def detect_column(cols, keywords):
    for c in cols:
        if any(k in c.lower() for k in keywords):
            return c
    return None


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

def process_csv(df, brain, ai_model, bucket, worksheet, final_cols):
    cols = df.columns.tolist()
    url_col = detect_column(cols, ["url", "link"])
    agent_col = detect_column(cols, ["agent"])
    date_col = detect_column(cols, ["date"])

    if not url_col:
        log("  Could not find a recording URL column in this CSV. Skipping file.")
        return None

    existing = get_existing_filenames(worksheet, final_cols)
    counts = {"success": 0, "skipped": 0, "excluded": 0, "failed": 0}
    total = len(df)

    for i, row in df.iterrows():
        url = str(row[url_col]).strip()
        agent_name = str(row[agent_col]).strip() if agent_col else "Unknown"
        call_date = str(row[date_col]).strip() if date_col else ""
        file_name = url.split("/")[-1].split("?")[0] or f"call_{i}.mp3"

        if file_name in existing:
            counts["skipped"] += 1
            log(f"[{i+1}/{total}] {file_name} — already scored, skipped")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, file_name)
            if not download_mp3(url, local_path):
                counts["failed"] += 1
                log(f"[{i+1}/{total}] {file_name} — download failed")
                continue

            duration = get_duration_seconds(local_path)
            if duration is None:
                counts["failed"] += 1
                log(f"[{i+1}/{total}] {file_name} — could not read duration, failed")
                continue

            if duration < MIN_DURATION_SEC:
                counts["excluded"] += 1
                log(f"[{i+1}/{total}] {file_name} — excluded ({duration:.0f}s < 90s)")
                continue

            try:
                upload_to_gcs(bucket, local_path, f"processed/{file_name}")

                with open(local_path, "rb") as f:
                    audio_bytes = f.read()

                transcript = get_transcript(ai_model, audio_bytes)
                scored = score_call(ai_model, transcript, brain, agent_name, call_date,
                                     file_name, duration)

                row_out = {
                    "Processed Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Call Date": call_date,
                    "Agent Name": agent_name,
                    "Customer Name": scored.get("Customer Name", "Unknown"),
                    "File Name": file_name,
                    "Call Duration (sec)": round(duration),
                    "Call Duration Flag": "Short" if duration < 180 else "Normal",
                    "Conversion Outcome": scored.get("Reason for Non-Conversion", ""),
                    "Overall area of opportunity": scored.get("Overall area of opportunity", ""),
                    "Primary Objection": scored.get("Primary Objection", ""),
                    "Reason for Non-Conversion": scored.get("Reason for Non-Conversion", ""),
                    "Coaching Tip": scored.get("Coaching Tip", ""),
                    "Full Transcript": transcript,
                }
                for col in final_cols:
                    if col not in row_out:
                        row_out[col] = scored.get(col, "")

                call_with_backoff(
                    worksheet.append_row,
                    [row_out.get(c, "") for c in final_cols],
                    value_input_option="USER_ENTERED",
                )
                counts["success"] += 1
                log(f"[{i+1}/{total}] {file_name} — scored and written")
                # free memory before the next iteration
                del audio_bytes, transcript, scored

            except Exception as e:
                counts["failed"] += 1
                log(f"[{i+1}/{total}] {file_name} — failed: {e}")

    return counts


def main():
    log("Starting 1799 QA audit run")
    brain = load_brain()
    ai_model, bucket, worksheet, drive_service = get_clients()
    ensure_headers(worksheet, brain["final_cols"])

    csv_files = list_inbox_csvs(drive_service)
    if not csv_files:
        log("No CSVs found in the inbox folder. Nothing to do.")
        return

    log(f"Found {len(csv_files)} CSV(s) in the inbox.")

    for f in csv_files:
        log(f"--- Processing file: {f['name']} ---")
        try:
            buf = download_drive_file(drive_service, f["id"])
            df = read_csv_robust(buf)
        except Exception as e:
            log(f"  Could not read {f['name']}: {e}. Leaving it in the inbox for review.")
            continue

        counts = process_csv(df, brain, ai_model, bucket, worksheet, brain["final_cols"])
        if counts is None:
            continue

        log(
            f"  Done with {f['name']}: {counts['success']} scored, {counts['skipped']} skipped, "
            f"{counts['excluded']} excluded, {counts['failed']} failed."
        )

        move_to_processed(drive_service, f["id"])
        log(f"  Moved {f['name']} to processed/")

    log("Audit run complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        sys.exit(1)
