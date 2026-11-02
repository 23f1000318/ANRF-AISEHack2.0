# ANRF AISEHack 2.0 Polymer Property Prediction

Machine-learning pipelines for predicting polymer properties from SMILES strings in the ANRF AISEHack 2.0 competition.

## Project Overview

The Round 2 task predicts seven properties:

- Chain bandgap (`Egc`)
- Bulk bandgap (`Egb`)
- Ionisation energy (`Ei`)
- Electron affinity (`Eea`)
- Dielectric constant (`EPS`)
- Refractive index (`Nc`)
- Glass transition temperature (`Tg`)

The primary implementation is a guarded ensemble combining molecular fingerprints, RDKit descriptors, multitask neural learning, scaffold-validated tree models, molecular kernels, and cross-property stacking.

## Repository Contents

- `round2_multitask_pipeline.ipynb`: Kaggle-ready end-to-end notebook.
- `round2_multitask_pipeline.py`: Equivalent script for inspection and local adaptation.
- `ROUND2_PROJECT_REPORT.md`: Methodology, experiments, findings, limitations, and lessons learned.
- `.gitignore`: Prevents competition data, generated submissions, caches, and secrets from being committed.

## Method

1. Load the official competition data and permitted auxiliary `PI1M.csv` data.
2. Merge only organizer-provided archive training labels when available.
3. Extract Morgan, RDKit topological, atom-pair, MACCS, and RDKit descriptor features.
4. Pretrain a compact fingerprint encoder using allowed PI1M self-supervision.
5. Train a seven-task neural model with scaffold-aware out-of-fold predictions.
6. Train target-specific tree and molecular-kernel models.
7. Generate cross-fitted property teachers that do not see the validation scaffold.
8. Add physics-informed cross-property features such as `Ei - Eea`, `Egb - Egc`, `Nc²`, and `sqrt(EPS)`.
9. Select blends and meta-models only when scaffold OOF validation supports them.
10. Apply exact and collision-safe canonical matches from organizer-provided labels, then write `submission.csv`.

## Reproducibility

Run the notebook in a Kaggle Notebook with the official Round 2 files attached. A GPU accelerator is recommended for PyTorch and XGBoost stages. The notebook installs missing public Python packages at runtime when necessary and prints validation results for each target.

The final output must be named:

```text
submission.csv
```

No external datasets, pretrained weights, cached embeddings, or generated artifacts are included in this repository. The `archive` data is used only when it is supplied by the competition organizers as official auxiliary data.

## Validation Philosophy

Random splits can overestimate performance when related polymers appear in both training and validation. Therefore, the pipeline uses Murcko scaffold groups and cross-fitted predictions wherever model selection is performed. Complex models are accepted only when they improve scaffold-based out-of-fold performance by a predefined margin.

## Limitations

- Scaffold validation is more conservative than a random split but remains an estimate of hidden-test performance.
- Some properties have relatively few labeled examples, limiting attainable accuracy.
- RDKit and GPU availability affect runtime and feature extraction.
- Organizer rules should be checked before using any auxiliary archive files in a submitted notebook.

## Short Description

Scaffold-aware multitask learning for polymer property prediction using RDKit descriptors, molecular fingerprints, cross-property physics features, and guarded ensemble blending.

