# Research Gaps and Honesty Notes

This document collects every place in the codebase where a scientific claim
is deliberately hedged, a practical shortcut is taken, or a comparison with
prior art is still needed — gathered in one place so nothing is buried in a
docstring.

## 1. No quantum advantage is assumed or claimed

The project's hypothesis (see `docs/methodology.md`) is about a
resource-aware *trade-off*, not superiority. `scripts/generate_report.py`
renders `RESULT PENDING` rather than fabricating favorable numbers, and
every experiment is designed so a classical-wins result is just as valid an
outcome as a quantum-wins result.

## 2. Quantum PCA is not scalable at dataset scale

`src/quantum/quantum_pca.py` implements a genuine swap-test-based routine
(`swap_test_quantum_pca`) but caps it at `max_samples=8` — true quantum PCA
via density-matrix exponentiation requires quantum RAM / fault-tolerant
depth unavailable on NISQ simulators at ULB/IEEE-CIS scale. `classical_pca`
is the practical default used everywhere else, and is labeled as classical,
never mislabeled as quantum.

## 3. QAFS uses a classical surrogate to score candidate subsets

`src/quantum/qubo_feature_selection.py::evaluate_feature_subset_objective`
trains a fast logistic-regression surrogate to estimate PR-AUC/FPR/
ExpectedLoss for each candidate feature subset, rather than training the
full quantum model for every candidate (which would be combinatorially
expensive). QubitCost/CircuitDepth are real circuit measurements; the
performance terms are a documented proxy. A rigorous follow-up would
validate that surrogate ranking against the actual trained quantum model's
performance on the same subsets.

## 4. QAOA-based QUBO solving does not scale past ~12 variables here

`solve_qubo_qaoa` is explicitly gated to `max_variables<=12` (simulable) and
raises rather than silently truncating the problem for larger instances;
`solve_qubo_simulated_annealing` is the general-purpose fallback. This means
QAFS's "quantum" optimization path is itself resource-constrained — a
finding worth stating plainly rather than glossing over.

## 5. Kernel methods (QSVC) are subsampled for training

Quantum kernel evaluation is O(n^2) circuit evaluations. Every experiment
runner subsamples the quantum training set (`train_sample_size` /
`quantum_train_sample_size` parameters) rather than training on the full
classical training set. Reported quantum-model numbers should always be read
alongside the `n_train` field in the experiment's JSON record.

## 6. Simulator results are not equivalent to real hardware

Unless `backend=ibm_quantum` was explicitly used (and real hardware was
reachable), reported "noise" results come from a simplified configurable
depolarizing + readout noise model (`src/quantum/noise_models.py`), not a
calibration pull from a specific real device. Real-hardware experiments
(spec section 19, Experiment M) require `IBM_QUANTUM_TOKEN` and are the only
way to validate simulator-based noise conclusions against actual hardware
behavior.

## 7. Quantum "explainability" is not SHAP-equivalent

`src/explainability/quantum_explainability.py` is explicit that circuit
structure/score reporting is a transparency mechanism, not a feature
attribution method with SHAP's game-theoretic guarantees (efficiency,
symmetry, additivity). The `feature_sensitivity_scan` finite-difference
proxy is similarly labeled as a local sensitivity measure, not a Shapley
value.

## 8. Novelty claims require literature comparison

Per spec section 32, the following are presented as *potential
contributions requiring comparison with prior art*, not established facts:
Quantum-Aware Feature Selection, resource-aware QML, quantum-gated selective
inference, cost-sensitive quantum-classical risk fusion, temporal
fraud-drift handling, noise-aware QML evaluation, cross-dataset validation,
hardware-aware evaluation, explainable hybrid fraud detection, and
latency-aware quantum inference. None of these have been checked against the
current QML/fraud-detection literature as part of this codebase; that
literature review is a prerequisite before any of these could be claimed as
novel in a publication.

## 9. Synthetic data is for development/testing only

`src.data.loaders.load_synthetic` generates a schema-compatible but
statistically simplified sample (Gaussian-perturbed features, no real fraud
patterns). Every experiment record and the dashboard tag synthetic-sourced
results with an `is_synthetic` flag / `_synthetic` dataset suffix. Research
conclusions must be drawn from real downloaded datasets, never from
synthetic smoke-test numbers.

## 10. Cross-dataset generalization is untested by default

The experiment matrix supports running the same pipeline across ULB,
IEEE-CIS, PaySim, BankSim, and the Fraud Detection Handbook simulator, but
no single default run does this automatically — cross-dataset validation
(Experiment L) requires explicitly invoking each dataset and comparing
results manually or via the registry.
