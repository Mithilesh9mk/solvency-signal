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

## 2026-10-17 (Phase 5 protocol, written BEFORE the test split is read)

Phase 5 is the one-shot evaluation on the locked test split. This section is committed before the test file is
touched. Nothing is tuned, selected or changed based on test results, and disappointing results are reported as they are.

1. **Frozen artifacts.** LightGBM + Platt, with the calibrator and band cuts committed at `fdef552`. The SHA256 of
   every artifact, and the library versions, are in `artifacts/phase5_protocol.json` (written by
   `python -m scripts.final_test_eval --freeze`, once) and in the table below. The model files are gitignored, so
   their hashes pin the exact files on disk; a rebuild is not guaranteed to be bit-identical. The script aborts on any
   hash mismatch, and before reading the test split it also checks that the frozen artifacts reproduce the Phase 4
   validation numbers exactly.

Frozen commit: `fdef552`. Git HEAD when frozen: `14b5dd3`. Headline cuts read from bands.json: Watch 0.1736, High 0.2403, calibration platt.

| File | SHA256 | Note |
|---|---|---|
| `models/lightgbm.joblib` | `4e81d03e88b82208e76675f471c1588e5fdc88a00c9e832b8a4627be5a00f49d` | gitignored, hash pins the exact file |
| `models/logistic_regression.joblib` | `22e76808d8ed31806d5029041721a1b12cce5eea717edd35b62062fd1b1bb43d` | gitignored, hash pins the exact file |
| `models/neural_net.joblib` | `88d98f160fb07f11102ba75daa0736090b158e8abdd52258a455616e37659a6a` | gitignored, hash pins the exact file |
| `artifacts/calibration/calibrators.json` | `b514b722b5a990a966ca49159291e6df37c111194286053c4f78bea48a713b86` | equals the fdef552 blob: True |
| `artifacts/calibration/bands.json` | `b507b74b588c48382f60bae70e2e27719dcd6b2be471363ed569b4e8aded5433` | equals the fdef552 blob: True |
| `artifacts/calibration/config.json` | `f5e746102a2757985b51bfccc5f30b5aa047751f4a684439381f866a19ddaae5` | equals the fdef552 blob: True |
| `reports/phase4_calibration.json` | `9772601cc7974225c39efee9e3c06b156d89e99deb6b2b030d8d28b7f28ce296` | comparison source, committed |
| `reports/models_val.json` | `a907499e7e7bb5fad26bf76bc5fa4b9621ce5b5da6b4aeb18ac8d4a24928cee8` | comparison source, committed |

2. **Headline model.** LightGBM with Platt, bands as committed (Watch 0.1736, High 0.2403). Logistic regression and the
   neural network are scored as reference only and are not eligible to replace the headline model, whatever the test
   shows.
3. **Metrics, stated in advance.** PR AUC and ROC AUC with stratified bootstrap 95% CIs (1000 resamples, seed 42), the
   prevalence baseline PR AUC, Brier score, log loss (clipped at 1e-4 as in Phase 4), ECE (10 equal-frequency bins, as in
   Phase 4), and per-band firm counts, bankrupt counts and rates. Recall and precision for Watch+High and for High
   alone, and the confusion matrix at the High cut.
4. **Verdict rule, fixed now.** The headline model "holds up" only if all three hold: (a) the test PR AUC point
   estimate lies inside the validation 95% CI; (b) the lower bound of the test PR AUC CI is above the prevalence
   baseline; (c) the 95% CI of High-band precision on test contains the 0.60 precision floor. Otherwise it is reported as
   "does not fully hold up", with the numbers.
5. **Comparison.** Test is set next to validation and train OOF. Gaps expected as optimism, stated before the run:
   validation cuts are fitted on validation (in-sample); LightGBM was chosen using validation (winner's curse); OOF
   Brier, log loss and ECE are in-sample for the two-parameter Platt map; OOF models see 80% of train so they understate
   the refit model slightly. The test set is clean of all four. A gap in the direction of those sources is expected; a
   larger or opposite gap is not explained by them.
6. **One-shot enforcement.** `scripts/final_test_eval.py` verifies hashes, refuses to run if
   `reports/phase5_test_results.json` or `artifacts/phase5.lock` exists, and requires a clean git tree. It writes the lock
   BEFORE it reads the test file (a deliberate addition to the user's spec: a crash after the read cannot be followed by a
   second read). On success it completes the lock with the test file SHA256 and the results SHA256. Tests prove the
   second run is refused and that no other source reads the test split.
7. **After the run.** No retraining, recalibration or re-cutting. If something looks wrong it goes into the Flags section
   and work stops. No fix-and-rerun. The markdown report can be re-rendered from the saved JSON (`--render-only`),
   which does not read the test split.
8. **Pre-run dry run.** `--dry-run` executes the same code on the validation split with temporary output, to catch code
   errors without spending the test split.
9. **Expectations before the run (no numbers).** I expect the test PR AUC at or somewhat below validation. I expect the
   Watch band to hold too few firms to read. I do not predict a direction for High precision: validation was built to
   reach the floor in-sample, while train OOF came out higher than the floor.
10. **Out of scope.** SHAP, the app, the Indian case study and Graphify.

## 2026-10-17 (Phase 5, after the one-shot run)

- The run used commit `5d16fa8` (protocol, script and tests committed before the test split was read), a clean tree, and
  frozen artifacts whose hashes matched the protocol. Before reading the test file the script confirmed that the frozen
  artifacts reproduce the Phase 4 validation numbers exactly. The lock file was written before the read and completed
  after. Results: `reports/phase5_test_results.json`, report `reports/phase5_test_results.md`.
- Nothing was retrained, recalibrated or re-cut. The headline model stays LightGBM + Platt with the committed cuts.
- Post-run additions that did not touch the test split: two Flags (the neural network scored slightly above the headline
  model on test, and the validation to test PR AUC change per model) were added to the code that renders the report, and
  the report was re-rendered from the saved JSON with `--render-only`. No metric was recomputed. These flags were added
  after seeing the results; they are observations, not selection or tuning.
- Per the protocol, the neural network and logistic regression are reference only and are not eligible to replace the
  headline model, even though the neural network's test PR AUC point estimate is marginally higher.
- README fix: the Method section said validation is never used for fitting or tuning, which was wrong after the user
  asked for validation-fitted band cuts. It now says so.
- The test split is locked for good. Further changes to the model need new data, not this split.
