import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score

# Input data only: no previous scripts or trained models are required.
root = r"C:\Users\Hzaab\SDP-Sequential-Diagnosis"
data = os.path.join(root, "Casande-RLold", "Baselines", "diaformer", "data", "ddxplus_dataset")
metadata_file = os.path.join(root, "data", "release_evidences.json")
output = "sdp1_extension_results.csv"

train_patients = 10000  # Use 0 for all eligible training patients.
eval_patients = 5000   # Use 0 for all eligible patients per evaluation split.
splits = ["val", "test"]
budgets = [3, 5, 10, 15, 17, 20, 30]
max_questions = 30
minimum = 2
cohort_size = 50
seeds = [42, 43, 44]   # Cohort partitions, not independent model training runs.
sample_seed = 42
views = 4
num_trees = 200
selector_epochs = 5
train_batch_size = 256
batch_size = 128
candidates = 4
learning_rate = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(sample_seed)
torch.manual_seed(sample_seed)

with open(metadata_file, encoding="utf-8-sig") as f:
    metadata = json.load(f)

features = []
for name, definition in metadata.items():
    if definition["data_type"] == "B":
        features.append(name)
features.sort()
feature_index = {name: i for i, name in enumerate(features)}

patients = pd.read_pickle(os.path.join(data, "goal_set.p"))
diseases = sorted(set(p["disease_tag"] for p in patients["train"]))
label_index = {name: i for i, name in enumerate(diseases)}


def make_dataset(records, limit):

    kept = []
    labels = []
    for p in records:
        first = list(p["goal"]["explicit_inform_slots"])[0]
        if first in feature_index:
            kept.append(p)
            labels.append(label_index[p["disease_tag"]])

    ids = np.arange(len(kept))
    if 0 < limit < len(ids):
        ids, _ = train_test_split(
            ids, train_size=limit,
            stratify=labels, random_state=sample_seed
        )

    answers, y, initial = [], [], []
    for i in ids:
        p = kept[i]
        explicit = p["goal"]["explicit_inform_slots"]
        findings = explicit.copy()
        findings.update(p["goal"]["implicit_inform_slots"])
        row = []
        for name in features:
            value = 0
            if name in findings:
                value = findings[name]["option"]
            row.append(value)
        answers.append(row)
        y.append(labels[i])
        initial.append(feature_index[list(explicit)[0]])

    print("Patients:", len(y), "classes:", len(set(y)), flush=True)
    return np.array(answers, dtype=np.float32), np.array(y), np.array(initial)


def make_input(x, observed):
    # The value and mask together distinguish observed "no" from unobserved.
    return np.concatenate([x * observed, observed.astype(np.float32)], axis=1)


def sample_observed(x, initial, rng):
    # Each training state contains the initial finding and 0-30 extra answers.
    scores = rng.random(x.shape)
    scores[np.arange(len(x)), initial] = -1
    ranks = scores.argsort(axis=1).argsort(axis=1)
    counts = rng.integers(0, max_questions + 1, size=(len(x), 1))
    return ranks <= counts


print("Device:", device, "Binary questions:", len(features), "Diseases:", len(diseases), flush=True)
x_train, y_train, initial_train = make_dataset(patients["train"], train_patients)
val_records = patients["test"]  # In these converted files, "test" is validation.
del patients

# 1. Train the disease predictor on randomly observed subsets.
rng = np.random.default_rng(sample_seed)
inputs = []
for view in range(views):
    observed = sample_observed(x_train, initial_train, rng)
    inputs.append(make_input(x_train, observed))

predictor = XGBClassifier(
    n_estimators=num_trees,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    device=device,
    objective="multi:softprob",
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    random_state=sample_seed
)
print("Training XGBoost...", flush=True)
predictor.fit(np.concatenate(inputs), np.tile(y_train, views))
del inputs
# Prediction receives NumPy CPU arrays; keep inference on the same device.
predictor.set_params(device="cpu")

# 2. Train the within-patient question selector; keep XGBoost fixed.
# This is GDFS-inspired gain regression, not the original GDFS algorithm.
selector = nn.Sequential(
    nn.Linear(2 * len(features), 128),
    nn.ReLU(),
    nn.Linear(128, len(features))
).to(device)
optimizer = torch.optim.Adam(selector.parameters(), lr=learning_rate)
rng = np.random.default_rng(sample_seed + 100)

for epoch in range(selector_epochs):
    selector.train()
    total_loss = 0.0
    order = rng.permutation(len(y_train))

    for start in range(0, len(order), train_batch_size):
        ids = order[start:start + train_batch_size]
        x = x_train[ids]
        y = y_train[ids]
        initial = initial_train[ids]
        rows = np.arange(len(x))
        observed = sample_observed(x, initial, rng)
        inputs = make_input(x, observed)

        scores = rng.random(x.shape)
        scores[observed] = np.inf
        selected = scores.argsort(axis=1)[:, :candidates]

        before = predictor.predict_proba(inputs)
        before_loss = -np.log(np.clip(before[rows, y], 1e-8, 1))

        # Reveal each candidate separately to calculate its training target.
        expanded_x = np.repeat(x, candidates, axis=0)
        expanded_observed = np.repeat(observed, candidates, axis=0)
        expanded_observed[np.arange(len(expanded_x)), selected.ravel()] = True
        expanded_y = np.repeat(y, candidates)
        after = predictor.predict_proba(make_input(expanded_x, expanded_observed))
        after_loss = -np.log(np.clip(after[np.arange(len(after)), expanded_y], 1e-8, 1))
        gains = before_loss[:, None] - after_loss.reshape(len(x), candidates)

        # Labels and hidden answers are used for targets, never selector inputs.
        inputs = torch.from_numpy(inputs).to(device)
        predicted = selector(inputs)
        selected = torch.as_tensor(selected, device=device)
        target = torch.as_tensor(gains, dtype=torch.float32, device=device)
        loss = nn.functional.mse_loss(predicted.gather(1, selected), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)

    print("Selector epoch:", epoch + 1, "Gain MSE:", round(total_loss / len(y_train), 5), flush=True)

selector.eval()
del x_train, y_train, initial_train


@torch.no_grad()
def get_states(x, initial):
    # Generate each patient's question path once in memory, then reuse it.
    # Allocation only reads the patient's CURRENT state, never future answers.
    n = len(x)
    probabilities = np.zeros((n, max_questions + 1, len(diseases)), dtype=np.float32)
    gains = np.zeros((n, max_questions), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        answers = x[start:end]
        rows = np.arange(len(answers))
        observed = np.zeros(answers.shape, dtype=bool)
        observed[rows, initial[start:end]] = True

        for step in range(max_questions + 1):
            inputs = make_input(answers, observed)
            probabilities[start:end, step] = predictor.predict_proba(inputs)

            if step < max_questions:
                scores = selector(torch.from_numpy(inputs).to(device)).cpu().numpy()
                scores[observed] = -np.inf
                selected = scores.argmax(axis=1)
                gains[start:end, step] = scores[rows, selected]
                observed[rows, selected] = True

    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-8, 1))).sum(axis=2)
    return probabilities, gains, entropy


def allocate(scores, budget, policy, seed):
    n = len(scores)
    cap = scores.shape[1]
    if policy == "equal":
        return np.full(n, budget, dtype=int)

    counts = np.zeros(n, dtype=int)
    if policy == "minimum_uncertainty":
        counts[:] = minimum

    # Fixed random tie order matches the previous heap implementation.
    tie_order = np.random.default_rng(seed).random(n).argsort()
    rows = np.arange(n)

    for step in range(n * budget - counts.sum()):
        priority = np.full(n, -np.inf)
        available = counts < cap
        priority[available] = scores[rows[available], counts[available]]
        patient = tie_order[priority[tie_order].argmax()]
        counts[patient] += 1

    return counts


results = []
policies = ["equal", "gain", "uncertainty", "minimum_uncertainty"]

for split in splits:
    records = val_records if split == "val" else pd.read_pickle(os.path.join(data, "test_data.pkl"))["test"]
    x, y, initial = make_dataset(records, eval_patients)
    n = len(y)
    print("\n", split, "patients:", n, "classes:", len(np.unique(y)), flush=True)
    probabilities, gains, entropy = get_states(x, initial)
    hard = entropy[:, 0] >= np.quantile(entropy[:, 0], 0.75)
    print("Patient question paths ready.", flush=True)

    for budget in budgets:
        equal_correct = probabilities[:, budget].argmax(axis=1) == y

        for seed in seeds:
            order = np.random.default_rng(seed).permutation(n)

            for policy in policies:
                counts = np.zeros(n, dtype=int)

                for cohort, start in enumerate(range(0, n, cohort_size)):
                    ids = order[start:start + cohort_size]
                    scores = gains[ids] if policy == "gain" else entropy[ids, :max_questions]
                    counts[ids] = allocate(scores, budget, policy, seed + cohort)

                p = probabilities[np.arange(n), counts]
                prediction = p.argmax(axis=1)
                correct = prediction == y
                loss = -np.log(np.clip(p[np.arange(n), y], 1e-8, 1)).mean()

                results.append({
                    "split": split,
                    "budget": budget,
                    "cohort_seed": seed,
                    "policy": policy,
                    "patients": n,
                    "accuracy": correct.mean(),
                    "macro_recall": recall_score(y, prediction, labels=np.unique(y), average="macro", zero_division=0),
                    "macro_f1": f1_score(y, prediction, labels=np.unique(y), average="macro", zero_division=0),
                    "log_loss": loss,
                    "high_entropy_accuracy": correct[hard].mean(),
                    "mean_questions": counts.mean(),
                    "total_questions": counts.sum(),
                    "zero_followup_fraction": (counts == 0).mean(),
                    "newly_correct_vs_equal": (correct & (equal_correct == False)).sum(),
                    "newly_wrong_vs_equal": ((correct == False) & equal_correct).sum()
                })

        # Only one output file; save progress after every completed budget.
        pd.DataFrame(results).to_csv(output, index=False)
        print(split, "budget", budget, "finished", flush=True)

results = pd.DataFrame(results)
print("\nAccuracy averaged across cohort partitions:")
print(results.pivot_table(index=["split", "budget"], columns="policy", values="accuracy").round(4).to_string())
print("\nSaved:", output)
