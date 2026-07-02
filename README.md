# MetaGoHealth 1799 QA Engine

Streamlit app: teammate uploads the calls CSV, clicks Run, every call gets
transcribed and scored against the Legend rubric, results land in the
Google Sheet. No manual step from Mark once deployed.

## One-time setup (Mark, ~10 minutes)

1. `git init && git add . && git commit -m "initial deploy"`
2. Push to a **private** GitHub repo
3. Deploy on [share.streamlit.io](https://share.streamlit.io), main file `streamlit_app.py`
4. In app Settings -> Secrets, paste your service account JSON (see `secrets.toml.template`)
5. Share the Google Sheet with `qa-audio-pipeline@gtm-5wnb34pj.iam.gserviceaccount.com` as Editor
6. Send the app URL to your teammate

## What the teammate does

1. Open the app URL
2. Upload the calls CSV
3. Confirm the column mapping (recording URL / agent name / call date)
4. Click Run Audit
5. Watch the live progress list
6. Click the Google Sheet link at the end

## Brain files

Located in `worker/brain_files/`. To update the rubric (e.g. add a parameter),
replace `legend.csv` and push. No code change needed as long as the column
structure (`Parameter_ID, Parameter_Name, Scoring_Type, Description,
Label_A..E, Label_A..E_When, Score_NA_When, Output_Column_Name,
Output_Reason_Column_Name`) stays the same.

## Notes

- Calls under 90 seconds are auto-excluded (not scored, not written to the sheet)
- Re-running the same CSV skips calls already present in the sheet (matched by File Name)
- MP3s are also archived to `gs://metago-qa-oudits/processed/` for audit trail
