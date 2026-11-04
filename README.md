# A Budget-Aware Agentic AI Assistant for Sequential Clinical Diagnosis

A Senior Design Project at **Khalifa University**, supervised by **Emadeldeen Eldele, Department of Computer Science**.

This project investigates how an AI diagnostic assistant can decide what information to acquire next while working within a limited acquisition budget.

Instead of receiving a complete patient record, the assistant begins with a minimal patient snapshot. It then selects questions or other available evidence sources, observes the results, updates its differential diagnosis, and decides whether to acquire more information or stop.

## Project Objectives

The proposed diagnostic loop is:

1. Receive an initial patient snapshot and a fixed acquisition budget.
2. Assess the current diagnostic uncertainty.
3. Select the next evidence source from the available catalogue.
4. Acquire the evidence and deduct its assigned cost.
5. Update the differential diagnosis.
6. Repeat until the budget is exhausted or a stopping criterion is met.
7. Return a diagnosis with an auditable trace of the acquisition decisions.

Acquisition costs may represent time, financial expense, or patient burden. The cost model, acquisition policy, and stopping criteria are research decisions to be developed and evaluated.

## Current Development

The repository currently contains notebooks for:

- Exploratory data analysis and correlation analysis.
- DDXPlus feature preparation.
- Classification baseline training.
- AFA-Bench experiment setup.
- Experiment tracking with MLflow on DagsHub.


## Repository Contents

| Path | Description |
|---|---|
| `afabench/dataprep.ipynb` | Converts DDXPlus patient records and evidence definitions into tabular features and numeric diagnosis labels. |
| `utilities/data_analysis/01_EDA.ipynb` | Explores demographics, diagnosis distributions, and recorded evidence. |
| `utilities/data_analysis/Correlation_analysis.ipynb` | Examines correlations within the dataset. |
| `utilities/Running Afabench.ipynb` | Notes and example commands for AFA-Bench experiments. |
| `utilities/Model Development/Baselines.ipynb` | Trains classification baselines and logs validation results. |
| `utilities/Model Development/MLFlow Guide.ipynb` | Provides experiment-tracking instructions and examples. |
| `utilities/Model Development/requirements.txt` | Additional model development dependencies. |
| `pyproject.toml` | Core dependencies and Python requirements. |

## Classification Baselines

The baseline notebook includes:

- Decision Tree
- Random Forest
- Extra Trees
- Logistic Regression
- Multilayer Perceptron
- XGBoost
- Dummy Classifier

It reports accuracy, macro precision, macro recall, macro F1, log loss, and ROC AUC when applicable.

## Experiment Tracking

**Experiments are saved using MLflow on DagsHub:**

[View Sequential Diagnosis Experiments](https://dagshub.com/HindAlz/Sequential-Diagnosis.mlflow/#/experiments)

The baseline notebook records model parameters, validation metrics, training time, dataset settings, and the associated Git commit.

## AFA-Bench Experiments

The project explores existing active feature acquisition methods through [AFA-Bench](https://github.com/Linusaronsson/AFA-Benchmark).

This repository contains data preparation code and experiment notes. The upstream AFA-Bench source and its environment must be configured separately.

See:

- [AFA-Bench reproduction instructions](https://github.com/Linusaronsson/AFA-Benchmark/blob/main/docs/tutorials/reproduce_full_results.md)
- `utilities/Running Afabench.ipynb`


## Project Status

This project is under active development. Current work establishes the data representations, classification baselines, and experimental infrastructure needed for the proposed sequential diagnostic assistant.
