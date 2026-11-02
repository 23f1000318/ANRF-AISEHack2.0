# AISEHack 2.0 Polymer Property Prediction - Round 2 Report

## 1. Objective

Round 2 required predicting seven polymer properties directly from polymer SMILES:

| Target | Property |
|---|---|
| Eea | Electron affinity |
| Egb | Bulk bandgap |
| Egc | Chain bandgap |
| Ei | Ionisation energy |
| EPS | Dielectric constant |
| Nc | Refractive index |
| Tg | Glass transition temperature |

The competition metric is the unweighted mean R2 across all seven targets. This makes performance on the small targets as important as performance on Egc and Tg.

## 2. Data Findings

The official training data contains 7,409 measurements, but the labels are highly imbalanced:

| Target | Training rows | Test rows |
|---|---:|---:|
| Eea | 221 | 147 |
| Egb | 337 | 224 |
| Egc | 2,028 | 1,352 |
| Ei | 222 | 148 |
| EPS | 229 | 153 |
| Nc | 229 | 153 |
| Tg | 4,143 | 2,763 |

Important observations:

- Four properties have only about 220 labeled examples.
- Most polymers have a label for only one property.
- Only 415 training polymers have labels for multiple properties.
- The public leaderboard represents approximately 37% of the test data; the remaining 63% is private.
- PI1M provides almost one million organizer-approved unlabeled polymer SMILES.

Strong relationships were observed among properties on polymers with shared labels:

| Property pair | Correlation |
|---|---:|
| Egb and Egc | +0.926 |
| EPS and Nc | +0.918 |
| Egb and Nc | -0.826 |
| Eea and Egc | -0.774 |
| Eea and Egb | -0.743 |

Physics-based checks also showed that Egc is closely related to `Ei - Eea`, while EPS and Nc follow a strong optical relationship.

## 3. Main Challenges

### Sparse labels

Ei, Eea, EPS and Nc contain too few rows for large independent models to generalize reliably.

### Unequal target sizes

Without balancing, Tg and Egc dominate neural-network training and the smaller properties receive insufficient updates.

### Chemical generalization

Random validation can be optimistic when closely related polymers appear in different folds. Public leaderboard tuning also risks overfitting the visible 37% split.

### Different physical mechanisms

Electronic, optical and thermal properties do not respond identically to one molecular representation. A single model family is therefore unlikely to be optimal for every target.

## 4. Selected Approach

The final pipeline combines three complementary components.

### 4.1 PI1M self-supervised pretraining

A fingerprint autoencoder is trained from scratch on a deterministic sample of 50,000 PI1M polymers. It learns a compact 128-dimensional molecular representation by reconstructing Morgan fingerprints.

Why it was selected:

- PI1M is explicitly provided by the organizers.
- No external labels or pretrained weights are used.
- Unlabeled pretraining helps the small targets learn from a broader chemical space.
- All weights are generated during the submitted notebook run.

The reconstruction loss decreased consistently from `0.53020` to `0.17236` over six epochs, confirming that the encoder learned meaningful fingerprint structure.

### 4.2 Shared seven-task neural network

The pretrained encoder is fine-tuned jointly across all seven properties. A shared molecular trunk learns common chemistry, while seven separate heads model property-specific behavior.

Inverse-frequency sampling balances the tasks so that scarce targets are sampled more often. Targets are standardized independently to prevent Tg's numerical scale from dominating the loss.

Why it was selected:

- Small targets can borrow representation strength from Egc and Tg.
- Separate heads preserve property-specific mappings.
- It performed substantially better than trees for Eea, Ei, EPS and Nc.

### 4.3 Target-specific tree ensemble

Each target also receives an independent ExtraTrees and LightGBM ensemble using:

- Morgan fingerprints at radii 1, 2 and 3
- RDKit path fingerprints
- Atom-pair fingerprints
- MACCS keys
- Basic molecular descriptors
- Polymer-backbone descriptors

ExtraTrees and LightGBM are blended using out-of-fold performance. Their predictions are then blended with the multitask neural predictions using a coarse, target-specific weight grid.

Why it was selected:

- Trees were substantially stronger for Egc and Tg.
- Neural models were stronger for most small targets.
- OOF target-specific blending avoids forcing one global model choice on all properties.

## 5. Validation and Generalization

The pipeline uses five-fold out-of-fold validation. Identical polymer SMILES are assigned to the same neural fold, preventing exact-polymer leakage.

Generalization safeguards include:

- No public leaderboard score is used as a training objective.
- PI1M is used only for unsupervised learning.
- No external datasets, labels, embeddings or pretrained models are used.
- Every neural validation prediction comes from a model that did not train on that polymer.
- Target blends use OOF predictions rather than training predictions.
- Blend weights are restricted to a coarse grid to reduce small-sample overfitting.
- Cross-property corrections require a minimum OOF gain before acceptance.

## 6. Model Performance

| Target | Neural R2 | Tree R2 | Final blend R2 | Neural weight |
|---|---:|---:|---:|---:|
| Eea | 0.8779 | 0.8390 | **0.8940** | 0.6 |
| Egb | 0.9017 | 0.8904 | **0.9186** | 0.6 |
| Egc | 0.8691 | 0.9002 | **0.9053** | 0.2 |
| Ei | 0.8086 | 0.7579 | **0.8255** | 0.6 |
| EPS | 0.7696 | 0.7370 | **0.7827** | 0.6 |
| Nc | 0.8471 | 0.7747 | **0.8516** | 0.8 |
| Tg | 0.8582 | 0.9032 | **0.9039** | 0.2 |

Mean OOF R2 across the seven final blends was approximately **0.8688**. The public leaderboard score was approximately **0.871**, which is closely aligned with validation and suggests limited public-split overfitting.

## 7. Cross-Property Stacking Experiment

A second-level Ridge model was tested using leakage-controlled predictions for all seven properties plus physics-inspired combinations.

| Target | Existing blend R2 | Cross-property R2 | Decision |
|---|---:|---:|---|
| Eea | 0.8940 | 0.8918 | Rejected |
| Egb | 0.9186 | 0.9172 | Rejected |
| Egc | 0.9053 | 0.9071 | Rejected: gain below threshold |
| Ei | 0.8255 | 0.8033 | Rejected |
| EPS | 0.7827 | 0.7711 | Rejected |
| Nc | 0.8516 | 0.8421 | Rejected |
| Tg | 0.9039 | 0.9050 | Rejected: gain below threshold |

The correction was accepted only if it improved OOF R2 by more than `0.003`. No target passed this gate, so the submitted predictions remained unchanged. This was intentional: physical correlations were real, but the limited overlap among labeled polymers made the learned corrections unstable.

## 8. Why Other Approaches Were Not Selected

### Independent models only

Independent models ignore shared chemistry and performed poorly on properties with about 220 labels.

### One universal model without property heads

A single output mapping cannot represent the different scales and physical mechanisms of all seven properties.

### Tree models only

Trees performed well for Egc and Tg but were weaker for Eea, Ei, EPS and Nc.

### Neural model only

The multitask network helped scarce targets but underperformed rich fingerprint trees for Egc and Tg.

### Direct physics correction

Relations such as `Egc ≈ Ei - Eea` and the EPS/Nc relationship were supported by the data, but only a small number of polymers had all required labels. Hard enforcement could amplify errors from weak component predictions.

### Randomized-SMILES TTA

The selected models consume molecular fingerprints and descriptors, which are invariant to alternative SMILES ordering. Randomized-SMILES TTA would therefore produce effectively identical inputs and no useful diversity.

### External pretrained models

They are prohibited by the competition rules and were not used.

## 9. Limitations

- Ei, EPS and Nc remain the main performance bottlenecks.
- Exact-SMILES grouping does not fully prevent related chemical scaffolds from appearing across folds.
- Small-target validation scores have substantial variance.
- PI1M pretraining learns molecular structure but not direct property information.
- The public leaderboard covers only 37% of test rows and may not perfectly represent the private set.
- OOF blend selection still introduces some model-selection uncertainty on small targets.

## 10. Lessons Learned

1. Multitask learning is valuable when labels are sparse, but property-specific heads are essential.
2. Rich cheminformatics fingerprints remain highly competitive for Egc and Tg.
3. Neural and tree models make complementary errors; target-specific blending is better than selecting one global model.
4. Strong physical correlations do not automatically justify hard constraints when component predictions are noisy.
5. Validation improvements should be required before adding complexity.
6. Public leaderboard feedback should confirm validation, not replace it.
7. Conservative rejection of unstable stacking is important when the private test set determines the final ranking.

## 11. Final Pipeline Summary

The submitted workflow is fully reproducible inside one Kaggle notebook:

1. Load official train, test and PI1M files.
2. Generate molecular fingerprints and polymer descriptors.
3. Pretrain a fingerprint autoencoder on organizer-provided PI1M SMILES.
4. Train five polymer-grouped multitask neural folds.
5. Train target-specific ExtraTrees and LightGBM folds.
6. Select target-specific OOF blends.
7. Evaluate and conservatively reject unstable cross-property corrections.
8. Train full-data tree models and average neural fold predictions.
9. Generate `submission.csv` in the required format.

This approach was selected because it balances predictive performance, compliance, interpretability and private-test generalization.
