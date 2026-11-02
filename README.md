# ANRF AISEHack 2.0 Polymer Property Prediction

This repository documents our work on both rounds of the ANRF AISEHack 2.0 polymer property prediction challenge. The models take polymer SMILES strings as input and predict experimentally measured material properties.

The code was developed and investigated locally, then adapted and executed in Kaggle Notebooks for competition submissions. This repository is the project record; it is not intended to suggest that the experiments were authored exclusively inside Kaggle.

## Round 1: Egc and Tg

Round 1 required predictions for two targets:

- Chain bandgap (`Egc`)
- Glass transition temperature (`Tg`)

The work progressed from the supplied Ridge baseline to more robust cheminformatics models:

1. RDKit molecular descriptors and cleaning for invalid, missing, and infinite values.
2. Morgan fingerprints, sparse fragment features, and dense molecular descriptors.
3. ExtraTrees, Random Forest, HistGradientBoosting, XGBoost, LightGBM, and regularized linear models.
4. Tanimoto-kernel regressors for structure-similarity modeling.
5. Out-of-fold weighted blending and positive stacking, with separate treatment for `Egc` and `Tg`.
6. Exploratory analysis of target distributions, SMILES length, ring content, and target-specific structure-property relationships.

The main lesson from Round 1 was that diverse molecular representations and target-specific blending were more reliable than a single descriptor model. The final experiments reached a public score of approximately 0.904, while also showing that leaderboard performance can differ from scaffold-based validation.

## Round 2: Seven Properties

Round 2 expanded the task to seven targets:

- Chain bandgap (`Egc`)
- Bulk bandgap (`Egb`)
- Ionisation energy (`Ei`)
- Electron affinity (`Eea`)
- Dielectric constant (`EPS`)
- Refractive index (`Nc`)
- Glass transition temperature (`Tg`)

The final Round 2 implementation uses:

- Rich RDKit features: Morgan radii 1, 2, and 3, RDKit topological fingerprints, hashed atom pairs, MACCS keys, and finite RDKit descriptors.
- Multitask neural learning so related properties can share structural information.
- Self-supervised fingerprint encoder pretraining using the permitted `PI1M.csv` auxiliary data.
- Scaffold-grouped cross-validation to reduce optimistic validation from related polymer structures.
- Target-specific tree ensembles and molecular-kernel models for properties with different data volumes.
- Cross-fitted property teachers that predict all seven properties without seeing the validation scaffold.
- Physics-informed meta-features, including `Ei - Eea`, `Egb - Egc`, `Nc^2`, and `sqrt(EPS)`.
- Guarded model selection: a complex blend or meta-model is accepted only when scaffold out-of-fold validation supports it.
- Exact and collision-safe canonical structure matching using organizer-provided labeled data, only when that auxiliary data is officially permitted.

## Repository Contents

- `anrf-aisehack-2-0-polymer-property-prediction.ipynb`: Final Round 1 submission notebook for `Egc` and `Tg`.
- `scaffold-cross-fitted-multi-property-ensemble-v1.ipynb`: Final Round 2 submission notebook for all seven polymer properties.
- `ROUND2_PROJECT_REPORT.md`: Detailed Round 2 methodology, experiments, findings, limitations, and lessons learned.
- `.gitignore`: Excludes competition datasets, generated submissions, caches, model artifacts, and secrets.

The repository contains the final notebook used for each competition round. Additional exploratory notebooks, including separate GNN, Tanimoto-kernel, and exact fragment-count-kernel experiments, are not included because they were research variants rather than final submissions.

## Data and Competition Compliance

The pipeline is designed to use only official competition files:

- Round 2 `train.csv`
- Round 2 `test.csv`
- Organizer-provided `PI1M.csv` auxiliary SMILES data
- Organizer-provided archive labels, when explicitly permitted for Round 2

The archive logic uses labeled training records supplied by the organizers. It does not use an unlabeled archive test file as a source of answers. External datasets, pretrained weights, cached embeddings, and generated artifacts are not committed to this repository.

Competition rules should always be checked before running the archive or auxiliary-data sections in a submitted notebook.

## Reproducing a Submission

1. Open the final notebook for the relevant round in a Kaggle Notebook:
   - Round 1: `anrf-aisehack-2-0-polymer-property-prediction.ipynb`
   - Round 2: `scaffold-cross-fitted-multi-property-ensemble-v1.ipynb`
2. Attach the official competition data and any organizer-provided auxiliary files.
3. Enable a GPU accelerator for the PyTorch and XGBoost stages when available.
4. Run the notebook from start to finish.
5. Confirm that the output file is named `submission.csv`.
6. Review the printed scaffold OOF scores and target-specific blend decisions before committing the notebook.

RDKit, LightGBM, XGBoost, Optuna, and other missing public packages may be installed during notebook execution where the competition environment permits it.

## Validation Philosophy

Random splits can overestimate performance when closely related polymer structures occur in both training and validation. The Round 2 pipeline therefore uses Murcko scaffold groups and cross-fitted predictions for model comparison. The public leaderboard is treated as a useful signal, not as a training objective, because its test subset may differ from the hidden private subset.

## Limitations

- Scaffold validation is a stronger generalization check, but it is still only an estimate of hidden-test performance.
- `Ei`, `Eea`, `EPS`, and `Nc` have fewer labeled examples than `Egc` and `Tg`.
- RDKit feature generation and kernel models can be computationally expensive.
- GPU availability changes runtime but does not guarantee better predictions.
- No model can guarantee a private leaderboard score from public validation alone.


