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
- **Headline model: LightGBM with Platt calibration (user decision, 2026-10-17).** The neural network (TremorNet) stays in the repo and is reported as the deep learning comparison. It is not deployed. Risks are listed in reports/phase4_calibration.md.
- **Graphify deferred to Phase 8 (user decision, 2026-10-17); it did not install.** Reason it was not installed: The user-given repo `safishamsi/graphify` has a README
  that installs the PyPI package `graphifyy` (double y, described as a temporary name), but that package's PyPI metadata
  points to `Graphify-Labs/graphify`, a different owner, and its latest upload was one day before this phase.
  The README documents `graphify install` (a Claude Code skill, likely writes to user-level config), but not a code-only
  CLI command, an ignore file, or what the post-commit hook runs. Per the instruction to stop if unclear,
  nothing was installed and CLAUDE.md has no Graphify lines. To be revisited in Phase 8. `graphify-out/` is already in `.gitignore`.
- **Earlier band rule SUPERSEDED (see the last entry).** Original result: the approved rule degenerates on this data (Watch band empty, High broad). The
  result is reported as is. A post-hoc alternative (High at precision 0.50) is shown but NOT adopted. User to decide.
- **Calibrator storage moved to `artifacts/calibration/` (JSON, committed)** instead of the gitignored `models/`,
  so the calibrators and bands survive a fresh clone.

## 2026-10-17 (Phase 4, band rule replaced, stated BEFORE the rerun)

The first band rule (OOF precision floor 0.25, OOF recall 0.80) degenerated: Watch was empty. The user replaced it.
New rule, fixed before the rerun, applied to the calibrated probability of each model (LightGBM + Platt is the headline):

- **Watch cut:** the highest threshold at which validation recall is still at least 0.80 ("about 80%").
- **High cut:** the lowest threshold at which validation precision is at least 0.60.
- **Collision handling:** if the Watch cut is not below the High cut, there are only two bands, Low and High. In that case
  the single cut is the Watch (80% recall) cut and the flagged group is called High. The report states this plainly and
  no Watch band is forced.
- **Why 0.60:** a round number meaning "most flagged firms are truly distressed, with margin over a coin flip". The
  precision-floor diagnostic and the validation tradeoff table from the previous run had already been seen, so this
  floor is NOT a blind choice. It is fixed here and not tuned further. The report shows the full tradeoff table so
  readers can see sensitivity.
- **Caveat on using validation:** model selection is finished (LightGBM chosen), but the cuts are now fitted on the
  same validation split they are reported on, so validation precision and recall at the cuts are in-sample and
  optimistic, with only 61 positives. The independent check is the same cuts applied to train out-of-fold scores,
  which played no part in setting them. The test split stays unread and is the only fully untouched data.
- The post-hoc 0.50 alternative from the previous run is dropped.
