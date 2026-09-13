# Methodology

## Research hypothesis (stated precisely)

> Selective quantum-classical inference may improve the fraud-detection
> accuracy-false-positive-resource trade-off for difficult transactions
> under constrained quantum resources.

This project does **not** hypothesize or claim that quantum computing
outperforms classical machine learning in general. Every experiment is
designed to test the hypothesis above objectively, and if classical ML wins
on a given dataset/metric, that result is reported as-is (see
`scripts/generate_report.py`, which renders `RESULT PENDING` rather than
inventing favorable numbers, and reports whatever the actual run produced).

## Why PR-AUC, not accuracy

Real fraud rates run from ~0.17% (ULB) to a few percent (PaySim/IEEE-CIS). A
classifier that predicts "legitimate" for every transaction scores >99.8%
accuracy on ULB while catching zero fraud. Accuracy is therefore never used
as this project's primary or reported headline metric. The primary metric is
**PR-AUC** (average precision), reported alongside ROC-AUC, precision,
recall, F1, FPR, FNR, MCC, Brier score, and ExpectedLoss (see
`src/evaluation/metrics.py`).

## Preprocessing and leakage prevention

- Duplicate rows are dropped **before** splitting, so no duplicate of a
  training transaction can leak into the test split
  (`src/data/preprocessing.clean_dataframe`).
- Scaling/encoding statistics (median imputation, standard scaling, one-hot
  encoding) are fit on the **train split only** and applied unchanged to
  validation/test (`src/data/preprocessing.fit_transform_split`).
- Dataset-specific leakage columns are dropped by configuration
  (`configs/datasets.yaml: drop_columns`), e.g. PaySim's `isFlaggedFraud`
  (a rule-based flag derived from the same signal used to construct the
  label) and the Fraud Detection Handbook's `TX_FRAUD_SCENARIO`.
- `src/data/validators.py` additionally flags any *numeric* feature with
  >=0.98 absolute correlation with the target as a candidate leakage column,
  as a heuristic safety net — always documented, never silently dropped
  without a warning.

## Class imbalance

Default: class-weighted / `scale_pos_weight` training (no synthetic
oversampling), since oversampling in a domain this imbalanced risks
overfitting to synthesized minority patterns. SMOTE/undersampling are
available (`src/data/imbalance.resample_train_only`) for explicit ablation
only, and are enforced to run on the training split alone.

## Splitting

- `random_split`: stratified train/val/test, class balance preserved in
  every split.
- `temporal_split` / rolling-window forward chaining
  (`src/evaluation/temporal_validation.py`): chronological, no shuffling —
  intentionally preserves drift so it can be measured, not accidentally
  averaged away.

## Classical baselines

Logistic Regression, Random Forest, XGBoost, LightGBM — each with
class-weighting, optional isotonic/Platt probability calibration
(`src/classical/calibration.py`), and cost-aware threshold tuning
(`tune_threshold_expected_loss`, `tune_threshold_f1`, `tune_threshold_youden_j`).

## Quantum models

Built against the **installed** Qiskit 2.x API (function-style
`zz_feature_map`/`real_amplitudes`/etc. from `qiskit.circuit.library`, V2
Sampler/Estimator primitives) — never against deprecated 0.x/1.x tutorials.
See `docs/reproducibility.md` for exact pinned versions.

- **Quantum kernel + QSVC**: `FidelityQuantumKernel` + `ComputeUncompute`,
  trained via `qiskit_machine_learning.algorithms.QSVC`. O(n^2) circuit
  evaluations — training sets are explicitly subsampled for tractability.
- **VQC**: feature map -> ansatz -> `VQC`, configurable optimizer
  (COBYLA/SPSA/L-BFGS-B).
- **QNN**: `EstimatorQNN` (Z-expectation output) or `SamplerQNN`
  (parity-interpreted measurement distribution), each feeding a small
  classical logistic "output layer" as specified.

All quantum training subsamples use `src/data/imbalance.stratified_subsample`,
which guarantees minority-class representation — plain random subsampling on
a ~1.7%-fraud dataset can (and, before this was fixed, did) draw zero fraud
rows at small sample sizes, which crashes binary-target quantum classifiers.

## Quantum-Aware Feature Selection (QAFS)

QAFS (`src/quantum/qubo_feature_selection.py`) formulates feature selection
as a QUBO over relevance (mutual information) and redundancy (pairwise
correlation) with a cardinality-constraint penalty, solved via QAOA
(small instances, <=12 variables, on a Qiskit sampler) or classical
simulated annealing (the general-case fallback). Each candidate subset is
scored against the full objective

```
J = alpha*(1 - PR_AUC) + beta*FPR + gamma*ExpectedLoss
    + delta*QubitCost + epsilon*CircuitDepth + zeta*Latency
```

PR-AUC/FPR/ExpectedLoss are computed from a fast classical logistic-regression
surrogate trained on the candidate columns — a documented, explicit
approximation used purely to rank candidate subsets cheaply; QubitCost and
CircuitDepth are exact measurements of the real feature-map circuit for that
subset.

## Cost-sensitive decision engine

`ExpectedLoss = FN * fraud_loss + FP * false_positive_cost + n_review * review_cost`,
with `fraud_loss`/`false_positive_cost`/`review_cost` configured in
`configs/experiments.yaml: cost_model` (never left as undocumented magic
numbers). Both decision thresholds (`review_threshold <= block_threshold`)
are chosen by grid-search minimization of this loss on the validation set
(`src/hybrid/cost_sensitive_decision.optimize_thresholds`), not picked
arbitrarily.

## Statistical rigor

`src/evaluation/statistical_tests.py` provides bootstrap confidence
intervals (`bootstrap_metric_ci`) and McNemar's test (`mcnemar_test`) for
comparing two classifiers on the same test set — used before claiming one
model "beats" another. PSI and KS tests support concept-drift monitoring.
