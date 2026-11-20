TWO-PAGE WEBSITE: YOUR MODEL + SHARED-BUDGET CLINIC

PAGE 1: Single-patient consultation; you answer Yes/No or skip a question.
PAGE 2: Shared-budget clinic; a doctor avatar walks to the patient chosen by the
allocator, asks that patient's model-selected question, and reveals the simulated
patient response. Predictions and the remaining shared budget update each step.
Both pages use your SAME loaded clean XGBoost and GDFS-inspired selector.
No training occurs. Inference runs locally on CPU. No external image/font service.

FILES IN THIS BUNDLE (keep together)
  clinic_web.py       local web server and single-patient consultation
  clinic_model.py     loader for your CURRENT two-layer clean selector + XGBoost
  cohort.py           sequential allocation and dataset replay
  binary_web.html    preferred single-patient UI, with a link to the second page
  cohort.html        animated clinic page

1. DOWNLOAD YOUR CURRENT MODEL FROM HPC
From /data/student_storage/SDP1_group1/sdp1_extension/outputs/YOUR_JOB_ID/ copy:
  clean_xgb.json
  clean_selector.pt
  config.json
Put all three from the SAME completed run in, for example:
C:\Users\Hzaab\SDP-Sequential-Diagnosis\models\sdp1_full
Use a completed selector checkpoint; config.json records completed epochs.
The loader also works with the earlier clean_30q folder, but that is a DIFFERENT
trained model. Use full-training weights when demonstrating full-training results.
This loader does not accept old xgboost.json/selector.pt/schema.json files or the
older selector architecture with two 128-unit hidden layers. The CURRENT selector
has ONE 128-unit hidden layer: 2*features -> 128 -> features.

2. USE YOUR EXISTING WORKING PYTHON ENVIRONMENT
Keep the PyTorch installation that works on your laptop.
If necessary: python -m pip install numpy xgboost
Prefer the same XGBoost/PyTorch versions as training when transferring checkpoints.
No Flask, Streamlit, pandas, GPU setup or new environment is required by this app.

3. RUN FROM THE EXTRACTED WEBSITE FOLDER
Stop any older website server on port 8000 first (Ctrl+C in its terminal).
Run this as ONE command in your terminal:
python clinic_web.py --model "C:\Users\Hzaab\SDP-Sequential-Diagnosis\models\sdp1_full" --metadata "C:\Users\Hzaab\SDP-Sequential-Diagnosis\data\release_evidences.json" --data "C:\Users\Hzaab\SDP-Sequential-Diagnosis\Casande-RLold\Baselines\diaformer\data\ddxplus_dataset"

Adjust the three paths to your actual files. Metadata must correspond to the
checkpoint evidence IDs; use original release_evidences.json, not a flattened CSV.
--data points at the converted folder containing test_data.pkl, a dict with 'test'.
For a different port add --port 8001.
For a smaller maximum add --max-turn 20; cannot exceed the trained model budget.

Open:
  http://127.0.0.1:8000/          single patient
  http://127.0.0.1:8000/cohort    shared-budget clinic
Keep the terminal running while using the pages. Ctrl+C stops the server.
If only the single-patient page is needed, omit --data. Cohort replay needs it.

4. USE THE COHORT VISUALIZATION
Choose 2-12 patients, policy, average follow-up budget, cap, minimum and seed.
Click Load patients. The first load can take time while test_data.pkl is read.
Click Next question for one visit or Play for automatic replay. Pause finishes
an interaction already underway, then prevents the next one. Adjust animation
speed at any time. Reduced-motion browser settings skip walking animation.
Click a patient card to inspect that patient's current top-five predictions.
Optional reference labels are hidden visually by default, but available for
inspection. They NEVER determine patient priorities or question selection.
Export replay downloads the question/answer/prediction history as JSON.
Reload the same seed to compare different policies on the SAME sampled cases.
The current session's displayed policy stays fixed until you load a new cohort.

EXAMPLE
6 patients * average 10 = 60 follow-ups. Initial findings are free.
Cap 30 limits each patient. Minimum=2 first spends 12 units on coverage, then
allocates the remaining 48 by current diagnostic entropy.

POLICIES
Equal: rounds of one follow-up per patient until everyone gets the average.
Gain: next patient has the highest predicted loss reduction from their best
      unanswered question. Negative scores do not cause stopping.
Uncertainty: next patient has the highest entropy of current disease probabilities.
Minimum + uncertainty: first everyone gets the chosen minimum, then entropy.
All policies use the SAME model to choose the selected patient's next question.
All spend the SAME total. Average=cap forces equal final question counts.
No early stopping, clinical severity rule or new learned allocator is added.

DATA AND MODEL SEPARATION
This page replays binary answers from held-out converted dataset records. It
reads the selected question response only after allocation chooses the patient.
An unlisted binary finding is simulated as No, matching the training convention.
The disease predictor and selector receive observed values plus observation
mask only. Reference diseases and unrevealed answers are not model inputs or
allocation priorities. Every patient starts with their recorded initial finding.
Nonbinary initial findings are excluded. Dataset sampling here is random for a
small visual demonstration, not the stratified large-sample report experiment.
The same seed samples the same patients across policies. No measured clinical
severity, treatment or waiting-time interpretation should be inferred from the
walking avatar; walking time does not affect the question budget or model.
The cohort page is automatic dataset replay. The original single-patient page
is where YOU enter answers. Cohort visual events are real backend decisions,
not a prerecorded or random animation. True model results require your weights.

The summary of matches to dataset labels is a small-cohort inspection aid, not
a replacement for your held-out evaluation. Top-five values are raw model
probabilities and are not guaranteed calibrated or to sum to 100% among five.
The minimum is an allocation guarantee, not a diagnostic-accuracy guarantee.
Sessions exist in local server memory. Export before restarting or reloading.
The server binds to 127.0.0.1, not a public network interface.

VALIDATION
Python 3.8 syntax and JavaScript syntax checked. Allocation/budget/cap/floor,
no repeated follow-up, hidden-answer isolation, reference-label isolation,
stale-turn rejection, HTTP page/API serving, four complete cohort runs, and
single-patient Yes/No/skip behavior checked using a test-double model.
Real checkpoint inference was not executed here because torch/xgboost are not
installed in the authoring runtime. Full browser rendering was not verified;
the available Chromium process could not launch in this execution environment.
No test-double model, test server or synthetic demo scores are included in the
shipped app. The app only starts with actual model files.

UPDATE: GUARANTEED FIRST QUESTIONS ARE EXPLICITLY VISIBLE
For Minimum + uncertainty, the clinic shows the guaranteed-question phase with
a cohort progress bar and a numbered round. Each patient card shows guaranteed
questions completed. Clicking a patient reveals their first minimum follow-up
questions, with the actual question wording and Yes/No responses, below the
ranking. Before a question is asked its slot says Waiting to be asked; it is not
prefilled or treated as answered. Visit-history entries identify guaranteed 1/2,
guaranteed 2/2, and later uncertainty-phase questions. The doctor walks for every
one of these interactions. Uncertainty allocation begins only once ALL patients
have completed the minimum. The initial evidence is separate and free; both
minimum follow-ups consume units from the shared budget. Default minimum is two,
but the display also follows the configured value. Existing users replace
cohort.py and cohort.html, restart the server, refresh, and load a new cohort.
Phase/progress/history checks passed using a test-double engine.

UPDATE: FIVE SINGLE-PATIENT REFERENCE CASES
The single-patient page now offers five fixed example patients from the same
converted test_data.pkl (sample seed 42), plus the original Custom consultation.
Run with --data to enable the cases. Choosing a patient supplies their recorded
initial finding only. The reference panel lists their positive binary findings;
other supported binary findings have simulated answer No under the dataset's
encoding. The panel also shows the dataset answer for the CURRENT question,
without selecting a radio button or submitting anything. You still enter Yes,
No or skip manually. A collapsible section optionally reveals reference pathology.
Reference labels and positive findings are separate from model inference inputs.
Manually entering a different answer uses YOUR answer; the reference is not
silently substituted. It remains the original case reference, not truth about a
modified consultation. The selected reference sheet persists for that session.
Five cases are a stable sample, not five guaranteed distinct diseases.
Existing users replace clinic_web.py and binary_web.html, restart, and refresh.
The cohort page and its explicit minimum-question rounds remain included.
Checks passed for all five selections, initial-only inference, manual answers
opposite to reference, skip isolation, and custom consultations using a test double.

LAYOUT UPDATE
Single-patient page: setup and question card in the left column, patient reference
guide immediately beneath the question, disease probabilities in the upper right,
and conversation history beneath the probabilities. For this layout update alone,
replace binary_web.html and refresh the browser (restart the server because HTML
pages are loaded into memory on startup). Model files and inference are unchanged.
