# Decisions log

Newest at the bottom. Each entry: date, decision, reason.

## 2026-10 (Phases 0 to 3, recorded retroactively in Phase 4)

- **Dataset: Polish `5year.arff`, 1-year horizon.** Public, CC BY 4.0, 64 documented ratios, about 7% positives
  so it is realistic for distress prediction. Other candidates (American NYSE and NASDAQ, Taiwanese) were
  not chosen because of unclear labels or pre-scaled features. See reports/data_report.md.
- **Split: stratified random 70/15/15, seed 42, after dropping exact duplicates.** No company id or year exists,
  so a time-based split is impossible. This overstates real-world performance; flagged.
- **Metric: PR AUC as primary.** About 7% positives make ROC AUC flattering. Also report Brier, recall at fixed
  false positive rates, and bootstrap CIs.
- **No class reweighting.** Keeps probabilities comparable to the true rate, which matters for calibration.
- **Neural network config chosen on train CV only** (quantile scaling), never on validation.
  See reports/models_val.md.

## 2026-10-17 (Phase 4)

- **Calibration data: 5-fold out-of-fold predictions on train.** A carved calibration split would leave about
  85 positives; cross-fitting uses all 286. Validation is only used to report, never to fit calibrators or
  choose thresholds.
- **Calibration method rule (set before results):** per model, lowest cross-validated log loss on the
  out-of-fold train scores, with ties within one standard error resolved toward the simpler method
  (none, then Platt, then isotonic).
- **Band rule (set before results, approved by user):** High is the lowest threshold with out-of-fold precision
  at least 0.25. Watch is the cut between Low and Watch that reaches out-of-fold recall of at least 0.80.
  Wording note: the approved text said "lowest threshold" for Watch, which taken literally flags every firm.
  It was applied as the highest threshold that still reaches 0.80 recall. If the Watch cut lands above the
  High cut, the Watch band is empty and this is flagged.
- **ECE uses 10 equal-frequency bins.** Equal-width bins would be mostly empty at a 7% base rate.
- **Cost ratios 5:1, 10:1, 20:1 are assumptions**, not estimates from data.
- **Calibrators saved as JSON** (Platt coefficients, isotonic knots) in `artifacts/calibration/`, so they load
  without refitting and without pickle version risk.
- **Headline model: UNDECIDED.** Claude recommends, the user decides. See reports/phase4_calibration.md.
- **Graphify NOT installed (paused, needs user confirmation).** The user-given repo `safishamsi/graphify` has a README
  that installs the PyPI package `graphifyy` (double y, described as a temporary name), but that package's PyPI metadata
  points to `Graphify-Labs/graphify`, a different owner, and its latest upload was one day before this phase.
  The README documents `graphify install` (a Claude Code skill, likely writes to user-level config), but not a code-only
  CLI command, an ignore file, or what the post-commit hook runs. Per the instruction to stop if unclear,
  nothing was installed and CLAUDE.md has no Graphify lines. `graphify-out/` is already in `.gitignore`.
- **Band rule result (Phase 4):** the approved rule degenerates on this data (Watch band empty, High broad). The
  result is reported as is. A post-hoc alternative (High at precision 0.50) is shown but NOT adopted. User to decide.
- **Calibrator storage moved to `artifacts/calibration/` (JSON, committed)** instead of the gitignored `models/`,
  so the calibrators and bands survive a fresh clone.
