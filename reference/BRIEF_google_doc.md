# BL_MLOps_Assignment_Briefing (Google Doc version)
Source: https://docs.google.com/document/d/125bEdAl0RljQUo4NlKqrBSWY5FjtWOAMeeZBbAdLb7I (fetched 2026-10-16)
Sections 1–5 match reference/README.md. Section 6 (Task) DIFFERS from the repo README and is reproduced in full below.
The email that came with it (Shaked Bar, 2026-10-16): 48 hours; deliverables = a repository with runnable code and a README, plus a short document or presentation covering implementation, decisions and considerations.

## 2. Data
Drive folder: https://drive.google.com/drive/folders/1pX2IxzcJEV4Fl520ZkeyT935bNhlotY3 — sessions 2025-12-11 → 2026-02-11.

## 4/5. Script I/O (as in the doc)
- Training inputs: bl_full_data.csv at input_path + input_file; TABPFN_TOKEN env var; train_test boolean in the constructor.
- train_test=True: time split, the last 7 days are test; logs per-day classification reports + payout MAPE/MAE at thresholds; no artifacts.
- train_test=False: "Trains on all data and writes artifacts. No evaluation."
- Outputs: payout_tfm_context.joblib (x, y, columns), CB_bl_lead.cbm, all_clients.csv, log_file_bl_train_<timestamp>.log
- Predictor: user dict (the current code reads it as a 1-row DataFrame); the 4 artifacts; TABPFN_TOKEN.
  Output {brand: {rank, expected_payout}}. Synchronous, user-facing path.

## 6. Task

### Part 1
Productize the two attached scripts on Databricks with MLflow. The models are given (Git link above) - do not change functions and logic.
Data is given (Drive link above). Productize the two attached scripts into a production-ready ML system using MLflow. The solution should be runnable locally. You may use Docker and any reasonable open-source or cloud infrastructure components.
Training pipeline. Support both existing modes: train/test mode and production (trains on all data, registers artifacts). Schedule the production run weekly, Sunday 05:00. Log technical parameters to MLflow so runs are comparable. Save researcher-defined log as an artifact for accuracy assessment. Version artifacts so a serving version can be identified and rolled back.
Pay attention to tabpfn regression model - you will need to productize it in the most efficient way (https://priorlabs.ai/)
Serving. Expose the predictor as an endpoint. Input: one user's post-funnel data as a dictionary (find example in the code). Output: the ranked brand dictionary. This sits on a synchronous, user-facing path — the user is waiting on the landing page.
Load simulation. Simulate production traffic against your endpoint. Report median and tail latency Minimize latency as possible.
Runnable code will be in main

### Part 2
Present your implementation, the dilemmas you faced, and the decisions you made. Describe how the system runs in production, potential issues, and ways to improve it. Discuss alternative solutions you considered.
The deliverable of Part 2 - word/pdf/presentation: document - not more than 2 pages; presentation - not more than 15 slides.
Deliverables - repo with README and runnable code, document / presentation.
