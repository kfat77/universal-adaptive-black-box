# Changelog

All notable changes are documented in this file. This project follows the principles of [Keep a Changelog](https://keepachangelog.com/) and uses semantic versioning for published releases.

## [0.3.1] - 2026-07-17

### Fixed

- Keep group boundaries during nested final model selection and group-aware MLP early stopping.
- Use the configured selection metric and output weights for non-fast hyperparameter search.

## [0.3.0] - 2026-07-17

### Added

- Nested and group-safe nested validation, tiered search budgets, and optional MLP learning-rate scheduling.
- Split-conformal calibration, finite-sample intervals, distribution-drift reporting, and model-agnostic explanations.
- DataFrame-safe prediction, named output weights, structured inverse targets, and sampled Pareto inverse design.
- Affine inverse constraints, group-safe split-conformal calibration, and multi-output gradient-boosting search support.
- Reproducible local benchmarks with measured CSV results, Windows CI, coverage enforcement, and additional examples.

### Changed

- Strengthened group isolation, finite-sample conformal calibration, Pareto ordering, and active-learning novelty scoring.

## [0.2.0] - 2026-07-17

### Added

- Installable `adaptive_surrogate` package and Python 3.10–3.12 CI matrix.
- Cross-validation validation strategies, baseline regressors, scale-aware metrics, and output weighting.
- Residual-calibrated prediction intervals and lightweight distance-based OOD assessment.
- Bounded inverse-design constraints, fixed variables, tolerances, reference-distance, and OOD penalties.
- Experimental active-learning recommendations and Pareto non-dominated filtering.
- Versioned artifacts, contributor guidance, security policy, citation metadata, and roadmap.

### Changed

- Positioned the project as a numerical surrogate modeling toolkit rather than a universal black-box engine.

## [0.1.0] - 2026-07-17

### Added

- Initial numerical regression training, forward prediction, and bounded inverse solving workflow.
