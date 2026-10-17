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

## 2026-10-17 (Phase 6 protocol: SHAP explanations, written BEFORE any SHAP is computed)

- **Final model confirmed (user decision, after Phase 5):** LightGBM + Platt, frozen at `fdef552`. The neural network stays
  the deep learning comparison, not deployed. Nothing about the model, calibrator or cuts changes in this phase.
- **Data:** SHAP is computed on the train and validation splits only. The test split stays locked. The frozen model file is
  checked against the hash in `artifacts/phase5_protocol.json` before it is used.
- **Scale and the Platt check (user requirement).** SHAP is computed in the model's raw log-odds. Platt is
  `sigmoid(a * logit(score) + b)`, so if the calibrator was fitted on the raw log-odds, calibrated log-odds are
  `a * raw + b` and each SHAP value converts exactly by multiplying by `a`. Before using that conversion the code verifies
  it: the converted values must reproduce the calibrated log-odds from the saved calibrator within 1e-6 on every
  validation firm, and no score may be clipped by the calibrator's clip at 1e-6. A test fails if this does not hold. If the
  check fails in the real run, SHAP is reported on the raw model scale and the report says so.
- **Feature names (user requirement).** The plain-language label for each feature is its UCI formula text from
  `src/polish_features.py` (UCI "X1 to X64"), with its attribute code. Any definition that the dictionary itself marks as
  ambiguous, inconsistent, unclear or a probable mistranslation is labelled **UNVERIFIED** with its attribute code. The test
  for that marking is a keyword rule on the dictionary flag (translation, unclear, odd, inconsistent, not stated). Nothing
  else is guessed. Quality notes (missing data, duplicates, undefined cases) are shown as notes, not as UNVERIFIED.
- **Categories (user requirement, with a correction to the premise).** The feature dictionary has NO category field and the
  UCI description gives none. The five categories (Liquidity, Leverage, Profitability, Cash Flow, Other) are therefore
  assigned by this project, once, by what the ratio's formula measures: short-term assets against short-term obligations
  is Liquidity, debt and capital structure is Leverage, profit-type numerators (including profit over liabilities and
  interest cover) are Profitability, profit-plus-depreciation measures are Cash Flow, and activity (turnover, days),
  growth and size are Other. The map is stored with the dictionary, is reviewable in the report, and is project
  judgement, not dataset documentation. Category importance is the sum of the member features' mean absolute SHAP. Correlated
  ratios split credit between them, so single-feature importance understates the ones with close twins, and categories
  are only a partial remedy.
- **Attr21 missingness (user requirement).** The bankruptcy rate with Attr21 missing vs present, on train and on
  validation, with Wilson 95% intervals. The pattern is stated only, with no causal explanation. The mean SHAP of Attr21 for
  missing vs present rows is reported alongside.
- **Importance reporting.** Mean absolute SHAP per feature on validation with stratified bootstrap 95% CIs (1000
  resamples, seed 42), the same measure on train for a stability check (Spearman correlation and top-10 overlap), and
  agreement with LightGBM gain importance (reported, not required).
- **Example firms, picked by a rule fixed now, not by eye.** From validation: the highest-scoring true positive, the
  highest-scoring false alarm (healthy firm), and the highest-scoring bankrupt firm below the Watch cut (the nearest miss).
  There is no company id, so firms are identified by validation row number.
- **Cross-checks.** Additivity (base plus SHAP sum equals the model output within 1e-6) and agreement of the `shap`
  library with LightGBM's native `pred_contrib`. Disagreement is reported as a flag, not hidden.
- **Dependency.** `shap` is pinned in `requirements.txt` to the installed version.
- **Out of scope:** the app, the Indian case study, Graphify, and any change to the model, calibrator or cuts.

## 2026-10-17 (Phase 6, after the run)

- SHAP was computed on train and validation only, for the frozen LightGBM + Platt model (hash checked). The test split
  was not read. Report: `reports/phase6_explanations.md`, artifacts in `artifacts/explain/`.
- **Platt slope check passed** on all train and validation firms (max error far below the 1e-6 tolerance, no clipped
  scores), so SHAP is reported in calibrated log-odds. Had it failed, the code falls back to the raw scale and says so.
- **Correction to the premise recorded in the protocol:** the feature dictionary had no categories. They were added to
  `src/polish_features.py` as a project-assigned map, and the report says so. Six borderline features are listed, and the
  report gives the share of importance they carry.
- The `shap` library returned contributions identical to LightGBM's native ones, so that comparison is not an independent
  check. Additivity against the model output is the independent one. This is stated in the report flags.
- Attr21 missingness is reported as a pattern only, as required. Its flags were generated from the numbers.
- `src/explain.py` replaced the Phase 0 stub. `shap` is pinned to the installed version in `requirements.txt`.

## 2026-10-17 (Phase 7 protocol: Attr21 missingness, written BEFORE the analysis runs)

Question: Attr21 (sales growth) missing is almost always a bankrupt firm in train and validation, and the model gives
those firms a very large score on that signal alone. What is in the data, and what would that mean for a user of the app?

- **Data and rules.** Train and validation only. The test split stays locked, so nothing here can be re-measured on it.
  The frozen model, calibrator and cuts are not changed, and no new model is trained or selected. Descriptive analysis,
  no tuning. Findings may feed the app design, which is decided separately.
- **Analyses fixed in advance.**
  1. Counts: how many firms, and what share of all bankrupt firms, have Attr21 missing (train, validation).
  2. Co-missingness: how many other attributes are missing in the missing group vs the present group, and the ten
     attributes whose missing rate differs most between the two groups.
  3. Sales-based check: among Attr21-missing firms, how many have Attr9 or Attr36 (sales / assets) equal to zero or
     missing. Attr21 is sales(n) / sales(n-1) by its definition, so this shows whether current sales exist.
  4. Profile: median of the six most important ratios for bankrupt firms with Attr21 missing, bankrupt firms with it
     present, and healthy firms (train).
  5. Performance split: PR AUC and ROC AUC with bootstrap CIs, and High-band precision and Watch+High recall, for firms
     with Attr21 present vs missing, on validation and on train out-of-fold scores (the Phase 4 five-fold procedure with
     the saved Platt calibrator). Also the share of the High band that has Attr21 missing.
  6. What-if on the frozen model, validation: set Attr21 to missing for firms that have it, and set it to the train
     median for firms that lack it. Report how many firms change band and the mean change in calibrated log-odds.
     This is a description of the frozen model's behaviour, not a recommendation to impute.
  7. The existing Phase 2 ablation (model retrained without Attr21) is quoted from reports/models_val.md, not rerun.
- **Wording rule.** Patterns and counts only. Candidate explanations are not asserted, because the dataset has no company id,
  no year and no field saying why a value is missing. Where the formula itself determines something (Attr21 cannot be
  computed without last year's sales), that is stated as a property of the definition, not as a story about bankruptcy.
- **Phase 5 numbers are unchanged.** Test results already include whatever reliance on Attr21 the model has. The test
  split cannot be broken down by Attr21 status without reading it again, so that breakdown will not be produced.

## 2026-10-17 (Phase 7, after the analysis)

- Analysis ran on train and validation only, per the protocol above. The frozen model, calibrator and cuts were not
  changed, and no model was trained or selected. The test split was not read. Report: `reports/phase7_attr21.md`.
- Result in one line: a missing Attr21 goes with a bankrupt label in nearly every train and validation case, the frozen
  model scores such firms High, and for firms with Attr21 present the model's precision and PR AUC are lower than the
  headline numbers. The numbers are in the report and are not repeated here.
- The headline model decision (LightGBM + Platt, user decision) is not reopened by this phase. What the finding changes is
  how the model may be presented and used, which is a design question for the app and is for the user to decide.
- Flags added after seeing results, as observations only: the headline numbers are flattered by a small group, and a user
  who leaves Attr21 blank gets the model's missing-value behaviour, not a neutral score.
- Not done on purpose: no imputation or retraining, because the test split is spent and any model change could not be
  evaluated on untouched data.

## 2026-10-17 (Phase 8 protocol: the app, written BEFORE demo firms or app numbers are generated)

Phase numbering: Phase 7 was the Attr21 probe, Phase 8 is the Streamlit app. Graphify stays deferred to a later phase.

- **Scope (user decisions).** A Streamlit app that scores firms with the frozen LightGBM + Platt model and explains the
  score. Inputs: demo firms and CSV upload only. The model, calibrator and cuts are not changed, and nothing is retrained,
  recalibrated, re-cut or imputed. The test split is never read by the app or by this phase.
- **Attr21 policy A (user decision).** If Attr21 is blank the app does not score the firm. It says why (sales growth needs
  last year's sales) and shows no band and no probability.
- **Blank-count rule, fixed before any blank count is looked at.** For firms whose Attr21 is present, other blank
  attributes are scored with LightGBM's native missing-value handling, with a warning that lists the blank attributes and
  says their effect is untested. Scoring is blocked if more than 6 of the 63 attributes other than Attr21 are blank (more than 10%).
  The limit is a round product guardrail, not fitted to the data. How many train and validation firms it would block is
  reported afterwards, as counts only.
- **Demo firms.** The Phase 6 rule picked firms driven by a missing Attr21, which policy A would refuse. The rule is re-run
  unchanged except that it is restricted to Attr21-present validation firms: the highest-scoring true positive, the
  highest-scoring false alarm (healthy firm), and the highest-scoring bankrupt firm below the Watch cut (the nearest miss).
  Firms are identified by validation row number, there is no company id. If a pick would itself be blocked by the blank
  rule, that is reported and the rule is not changed.
- **Numbers shown on the validation page** are generated by `scripts/run_phase8_prep.py` into `reports/phase8_app_numbers.json`
  and read by the app, so no figure is typed by hand. The page leads with firms where Attr21 is present (PR AUC with the
  no-skill baseline, High precision, recall, all with bootstrap CIs, on validation and on train out-of-fold), shows the
  all-firms headline beside it, and states in one sentence that the all-firms figure is flattered by missing Attr21.
  The one-shot test result (all firms, no Attr21 breakdown possible) is shown with the same caveat.
- **Bands on firms with Attr21 present.** The app shows the historical bankrupt rate per band on that subset (train out-of-fold
  and validation), and states that the cuts were set on all firms, so High precision on present firms is under the 0.60
  floor. No re-cut, no recalibration. ECE and a reliability check on the present subset are reported only.
- **No accuracy figure anywhere** in the app (a test scans the app text). The app shows PR AUC, precision, recall and band rates.
- **Allowed claims.** A screening signal for firms like those in the Polish data, not lending advice, random split only,
  so nothing is claimed about future years. The user asked for "non-financial firms". The UCI page does not state the
  sector, so the sector is UNVERIFIED and the app says the sector composition is not documented.
- **Privacy wording.** "Uploaded data is processed in memory and is not written to disk by this app." Not "nothing stored".
- **License (checked on the UCI page, 2026-10-17).** The page states Creative Commons Attribution 4.0 International
  (CC BY 4.0). Citation as given on the page: Tomczak, S. (2016). Polish Companies Bankruptcy [Dataset]. UCI Machine
  Learning Repository. DOI 10.24432/C5F600. Attribution is added to the README and the app. The page says the bankrupt
  companies were analysed over 2000 to 2012 and the still operating ones over 2007 to 2013, and that the data comes from
  EMIS. This project changed the data (dedup, splits, derived scores), which the attribution says.
- **Explanations in the app** use LightGBM's native contributions times the Platt slope, so the app does not need `shap`
  or `numba`. A test pins them equal to `explain_firm`.
- **Frozen artifacts at app start.** The app checks SHA256 of `models/lightgbm.joblib`, `artifacts/calibration/calibrators.json`
  and `bands.json` against `artifacts/phase5_protocol.json` and stops on a mismatch. The model file is committed through a
  `.gitignore` exception and marked binary in `.gitattributes`. The two JSON files were hashed with Windows line endings,
  so for those two text files the check accepts the same bytes with LF or CRLF line endings, and never for the model.
- **Deployment** is the user's action (public-facing). This phase prepares `app/requirements.txt` with pinned versions and
  the commands, and leaves the live-app link as a TODO.

## 2026-10-17 (Phase 8, after the build)

- The app follows the protocol above: policy A for Attr21, the blank limit of 6 fixed before any count was seen, demo firms
  re-picked with the restricted rule, validation page led by firms with Attr21 present, no accuracy figure, privacy wording
  "not written to disk by this app", and the model, calibrator and cuts unchanged.
- **Finding after the numbers were generated, reported as a flag.** A blank Attr27 also carries a strong signal, like Attr21:
  the bankrupt rate is far higher when it is blank than when it is present (train and validation), and in train it adds more
  than 1.0 calibrated log-odds on average. One demo firm (the highest-scoring true positive among Attr21-present firms) has
  Attr27 blank, and that blank is its largest driver. The pre-stated blank rule was not changed. Instead the app's blank
  warning now also shows the train and validation history for each blank attribute, labelled descriptive and not a test.
  The warning still says the effect on scoring is untested.
- **Blank warning will show for about half of scorable firms** (mostly Attr37, which is blank in about 42% of train firms).
  Reported in the flags.
- Floor statement uses three decimals so it matches the report (0.540 validation, 0.585 out-of-fold against 0.60).
- **License.** CC BY 4.0 confirmed on the UCI page. The sector is not stated there, so the user's wording "non-financial
  firms" could not be confirmed. The app says the sector is not documented (UNVERIFIED).
- `models/lightgbm.joblib` is committed through a gitignore exception and marked binary. The hash check tolerates LF or CRLF
  line endings for the two calibration JSON files only, because they were hashed on Windows with CRLF.
- Not done: deployment (the repo owner's action), manual entry of attributes, any model change. The live-app link stays a TODO.

## 2026-10-17 (Phase 9 protocol: how blank attributes change the score, written before any result is generated)

Scope: train and validation only, test split locked. The frozen model, calibrator and cuts do not change. Only the app's
blank policy layer can change. Nothing in this protocol was run before it was written. User approved the plan with a coverage
ceiling and four amendments, all included below.

**Population.** Firms with Attr21 present (the ones the app would score). Candidate attributes: every attribute other than
Attr21 that is blank in at least 20 Attr21-present train firms (the gate Phase 8 used). Attr27 gets no special treatment.

**Criteria for "block scoring when attribute X is blank".** All three must hold, applied mechanically by code.
1. Train history: bankrupt rate when X is blank is at least 3 times the rate when X is present, and the 95% Wilson lower bound
   of the blank rate is above the present rate. Mirror case: at most one third, and the Wilson upper bound is below the
   present rate.
2. Model effect, same direction as (1): the mean contribution of X to the score when X is blank (train, calibrated log-odds,
   native contributions times the Platt slope) is at least +1.0 for the "up" case, at most -1.0 for the "down" case.
3. Validation agreement: the validation rate ratio (blank over present) is at least 2 for the "up" case, at most 0.5 for the
   "down" case. Minimum evidence: at least 10 blank firms in validation (Attr21 present). Below that, criterion 3 is "not
   evaluable" and the attribute is not blocked.
An attribute that meets all three is blocked when blank, like Attr21. Blanks that fail a criterion keep the warning only.

**Multiple-testing exposure.** The number of candidates m is logged by code. Expected false positives under the null (no link
between a blank and the label) is estimated by code: labels permuted within train and within validation (Attr21-present
firms, 1000 permutations, seed 42), the same criteria applied to each candidate, criterion 2 taken from the frozen model's
contributions (it does not depend on the labels). Validation was already used for model choice, band cuts and the Phase 8
flag, so criterion 3 is not independent evidence. A blocked attribute is described as "meets the pre-registered criteria",
never as validated.

**Coverage ceiling.** New refusals are Attr21-present firms with at most 6 other blanks and at least one blocked attribute
blank. The ceiling is crossed if, in train or in validation, new refusals exceed 10% of Attr21-present firms or contain more
than 30% of the Attr21-present bankrupt firms. If crossed, the phase stops after the report: no app change, and the user
decides. The cumulative share including the count-limit refusals is reported for information only.

**Count limit of 6.** Groups by number of other blanks: 0, 1 to 2, 3 to 6, more than 6. PR AUC and High precision with 95%
bootstrap CIs (1000 stratified resamples, seed 42) on train out-of-fold and on validation. The limit is a candidate for change
only if, for the 3 to 6 group, both CIs lie entirely below the 0-blank group's CIs on train out-of-fold, and both validation
point estimates are lower. Otherwise it stays at 6. If the condition holds, the report proposes a value and the phase stops
for the user, like the ceiling.

**Single-blank stress test (descriptive, does not feed the rule).** For each of the 64 attributes, blank it for the
Attr21-present firms that have it, with the frozen model. Report the share of firms that change band and the mean change in
calibrated log-odds, on train and validation. Train uses the model's in-sample scores here because this is a model-sensitivity
check, not a performance claim. It covers attributes with too few blanks for a history test.

**Attr27 deep dive (descriptive, does not feed the rule).**
- Profile of Attr27-blank firms: counts, bankrupt rates, mean predicted probability (train out-of-fold, validation) against the
  observed rate, and which other attributes are blank with it most often.
- Stratification: Attr22 (operating profit / assets) sign, with "<= 0", "> 0" and "blank" as strata, and Attr2 (total
  liabilities / total assets) as the leverage ratio, cut at the train median among Attr21-present firms, with "blank" its own
  stratum. Bankrupt rate blank vs present within each stratum, with Wilson intervals.
- One ridge logistic regression on train (C = 1.0, lbfgs, seed 42): an Attr27-blank indicator plus every other attribute with a
  blank rate below 1% among Attr21-present train firms. No imputation. The other ratios are signed-log transformed
  (sign(x) * log1p(|x|)) and standardised on train. Rows with a blank in any covariate are dropped, and the count is reported.
  If more than 5% of rows would be dropped, the regression is reported as not evaluable. Interval for the indicator
  coefficient: 1000 stratified bootstrap resamples, percentile. An association only, not a cause.

**Cost of a block, in counts.** For each blocked attribute: firms and bankrupt firms refused in train and validation, and the
union across blocked attributes.

**If the rule is applied.** The app reads the blocked set from `reports/phase9_blank_rule.json`, not from a typed list. Refusal
text per attribute comes from the dictionary and the table. For Attr27 it says the ratio is undefined when financial expenses
are zero (dictionary note), that the data does not show whether a blank means zero expenses or missing data (UNVERIFIED), and
that a blank was linked to a much higher bankrupt rate in the training data so the score would reflect the blank, not
distress. No banner path: refuse, per the user's decision. Demo firms: if a demo firm would be refused, the demo rule is
re-run with the refusal rule included and the firm is swapped. It is not exempted.

**Cannot show.** The test split is spent, so no rule change can be checked on untouched data. Real users' blanks are probably not
random in the way the dataset's are. Multiple blanks together are tested only through the count check. Attributes with fewer
than 20 blank firms cannot be judged by history. Train out-of-fold scores use a calibrator fitted on those same scores. A block
protects against trusting a score driven by a blank, it is not a claim that the model is wrong on those firms.

## 2026-10-17 (Phase 9, after the run)

- The probe ran as specified, train and validation only, nothing changed in the model, calibrator, cuts or app.
- **Coverage ceiling crossed, so the phase stopped after the report, as pre-registered.** The pre-registered criteria are met by
  two attributes (Attr27 as a raised-risk blank, Attr24 as a lowered-risk blank). New refusals would hold more than 30% of the
  Attr21-present bankrupt firms in train and in validation. The numbers are in `reports/phase9_blanks.md`. The ceiling was not
  adjusted by the assistant. Applying the rule needs the user's decision.
- The count limit of 6 stays: the 3 to 6 blank group does not meet the pre-registered condition.
- Expected false positives under the null were tiny, but validation is reused, so the two attributes are described as meeting
  the pre-registered criteria, not as validated.
- Among Attr27-blank firms the model's mean predicted probability is close to the observed bankrupt rate (train out-of-fold and
  validation), and the blank signal survives the Attr22-sign and leverage splits and the ridge regression. So a refusal is a
  guard against a score driven by a blank (Attr27 is undefined when financial expenses are zero, per the dictionary, and the data
  does not show whether a blank means zero expenses or missing data, UNVERIFIED), not a statement that the model is wrong on
  those firms.
- `reports/phase9_blank_rule.json` records the candidates, criteria results and the status "stopped". The app does not read it yet.

## 2026-10-17 (Phase 9, user decision after the report, written before the app change and regeneration)

- **User decision:** refuse scoring when Attr27 or Attr24 is blank, and the user overrides the coverage ceiling for this. The
  ceiling stays recorded as crossed in the report. The rule is the set of attributes that met the pre-registered criteria, taken
  from the code's output, not typed by hand. The count limit stays at 6. The model, calibrator and cuts do not change.
- **How it is applied.** `scripts/run_phase9_blanks.py --override-ceiling` writes `reports/phase9_blank_rule.json` with the
  blocked set, the override noted, and per-attribute detail used in the refusal text. The app reads the blocked set from that
  file and fails closed: a missing or unreadable rule file stops scoring. Without the flag the script behaves as before and the
  rule stays empty.
- **Refusal text** comes from the rule file and the feature dictionary. It gives the train history and the model's contribution
  for the blank. Where the dictionary says the ratio is undefined when the denominator is zero (Attr27, financial expenses), the
  text says so and says the data does not show whether a blank means zero or missing (UNVERIFIED).
- **Demo firms.** The demo rule is re-run restricted to validation firms the app would score under all rules (Attr21 present, no
  refused blank, at most 6 other blanks). A demo firm that would be refused is swapped, not exempted. This replaces the earlier
  restriction to Attr21-present firms only, because the earlier rule would now pick a firm the app refuses.
- **Validation page.** Adds, report only, the performance on the firms the app now scores (same metrics, bootstrap CIs, seed 42),
  beside the existing Attr21-present and all-firms numbers, with the counts refused. Cuts are not re-set, so the High band's
  precision on this subset is reported against the 0.60 floor without any change.
- **Cannot show.** The test split is spent, so the effect of the refusal on untouched data is unknown. The subset numbers use a
  rule chosen after seeing train and validation, so they are descriptive, not an independent check.

## 2026-10-17 (Phase 9, after the app change)

- Applied as decided: Attr27 blank and Attr24 blank are refused, ceiling overridden by the user and recorded in the rule file and
  report. The count limit is unchanged. Model, calibrator and cuts unchanged. Test split untouched.
- The app reads the blocked set and the refusal text ingredients from `reports/phase9_blank_rule.json` and fails closed if it is
  missing. Tests pin the file to the criteria results, the refusal text, the unchanged limit, and the demo swap.
- Demo firms re-picked with the refusal rules included: the highest-scoring true positive changed (the earlier pick had Attr27
  blank and is now refused). Nothing was exempted.
- **Finding, shown on the validation page.** On the firms the app now scores, the metrics are lower than on Attr21-present firms
  (see `reports/phase9_blanks.md` section 7), because the refused firms were among the easiest bankrupt firms to flag. The
  all-firms headline and the Attr21-present numbers therefore overstate what the app does on the firms it scores. The cuts were
  not moved. Report only.

## 2026-10-17 (Phase 10 protocol: 4-ratio Quick check model, written before any result)

User decision: a separate small model that takes four ratios and returns a 1 to 10 score, a named band and a probability with a
range, shown as the first tab of the app. The frozen 64-ratio model, calibrator and cuts are not changed.

- **The four ratios, mapped to dataset attributes.** The user's list named current ratio twice; it is treated as one ratio, so
  there are four. (1) Current ratio = Attr4, current assets / short-term liabilities, exact. (2) Debt-to-equity = total
  liabilities / book equity, the reciprocal of Attr8 (book equity / total liabilities). The user wrote "total debt"; the data has
  total liabilities, so total liabilities is used (UNVERIFIED that this is what the user means by debt). (3) Net profit margin =
  Attr23, net profit / sales, exact. (4) The user asked for operating cash flow ratio. The data has no operating cash flow, so
  the closest available ratio is used, Attr26 (net profit + depreciation) / total liabilities, labelled "cash earnings ratio,
  a proxy". It differs from operating cash flow / current liabilities in both numerator and denominator.
- **Pre-registered signs (risk goes up when the ratio goes up):** current ratio down, debt-to-equity up, net profit margin
  down, cash earnings ratio down. The model sees Attr8 (equity / liabilities, finite and negative for negative equity), so its
  expected coefficient sign is the opposite of debt-to-equity. If any fitted coefficient has the wrong sign, the run stops, the
  report says so, and nothing is added to the app.
- **Population and split.** Train and validation firms with all four attributes present. Fit on train only, seed 42. The test
  split is not read. Validation is used for reporting only, not for any choice.
- **Features.** Each ratio is winsorized at the train 1st and 99th percentiles, then standardized with train mean and standard
  deviation. These bounds are stored and the app applies them to anything the user types.
- **Model.** Logistic regression, L2 with C = 1, no class weights, so probabilities are calibrated by construction. Challenger:
  LightGBM with monotone constraints on the same four inputs, small trees. The challenger replaces logistic regression only if
  its train 5-fold cross-validated PR AUC is higher by at least 0.02. Otherwise logistic regression is used. No recalibration;
  ECE and calibration slope are reported on validation.
- **Range.** Refit logistic regression on 1000 stratified bootstrap resamples of train (seed 42). The range shown with a
  probability is the 2.5th to 97.5th percentile of the refits for the entered point. It reflects estimation noise only. It does not
  cover a wrong model, a different country, a different year or a ratio defined differently.
- **Scale.** Cut points are the 10th to 90th percentiles of train out-of-fold (5-fold) predicted probability. Score = 1 + number
  of cut points the probability reaches, so 1 to 10. The scale is a rank within the Polish training population, not an absolute
  risk. Observed bankruptcy rate per score is reported with Wilson 95% intervals on train out-of-fold and validation.
- **Bands fixed now.** 1 to 3 Low, 4 to 6 Watch, 7 to 8 Elevated, 9 to 10 High.
- **Report only, no gate beyond the sign check.** On the same validation firms: PR AUC, ROC AUC, Brier, ECE with stratified
  bootstrap 95% intervals (n = 1000, seed 42), beside the frozen 64-ratio model and the no-skill rate. Expected: the 4-ratio model
  ranks firms worse than the 64-ratio model. The size of the gap is reported, not hidden.
- **Cannot show.** Behaviour on non-Polish firms, on untouched data (test is spent), and whether the proxy for operating cash flow
  behaves like the real ratio.
- **Artifacts.** `artifacts/quick/quick_model.json` (pure numbers, no pickle), `reports/phase10_quick.{md,json}`. The app fails
  closed if the file is missing or inconsistent. No outside data, no model download.

## 2026-10-17 (Phase 10 amendment, written before any challenger result beyond the train cross-validation PR AUC)

The first run stopped on the pre-registered rule: train 5-fold PR AUC was 0.299 for logistic regression and 0.351 for the
monotone LightGBM challenger, a difference of +0.052 against the +0.02 margin. All four logistic signs were as expected, but the
current ratio coefficient was -0.001 (no weight once the other three are in). Under the protocol the challenger replaces
logistic regression. The protocol did not say how to finish that case, so before generating any further result:

- **Model.** Monotone LightGBM on the four ratios as registered (150 trees, 4 leaves, min 30 per leaf, monotone constraints
  all negative on the model-space features, seed 42), fitted on all train firms with the four ratios present. No winsorizing or
  standardizing is needed; inputs are clipped to the train minimum and maximum so zero debt-to-equity stays finite.
- **Calibration.** Platt scaling (a logistic fit of the label on the log-odds of the raw score), fitted on the train 5-fold
  out-of-fold raw scores, because boosted scores are not calibrated by construction. Validation is reported only.
- **Scale.** Cut points are the 10th to 90th percentiles of the train out-of-fold calibrated probability, as registered.
- **Range (replaces the bootstrap-refit range).** Storing 1000 boosted refits is too large. The range shown with a score is the
  Wilson 95% interval of the observed bankruptcy rate among the train out-of-fold firms that got the same score. This is an
  empirical range and says how often firms with that score failed in the Polish data. It does not cover another country, year or
  ratio definition. The point probability is the calibrated model output.
- **Contributions.** LightGBM native contributions times the Platt slope, in calibrated log-odds, as in the main app.
- **Artifact.** `artifacts/quick/quick_model.json` holds the LightGBM model as its text dump plus numbers. No pickle.
- The stopped first-run report is kept as `reports/phase10_quick_stopped_lr_vs_challenger.md`.

## 2026-10-17 (Phase 10, after the run)

- Run as amended: the challenger replaced logistic regression, Platt fitted on train out-of-fold raw scores, scale cuts from train
  out-of-fold percentiles, range from the observed rate per score. All figures are in `reports/phase10_quick.md`; none are typed here.
- The Quick check is the first tab of the app. Inputs are the four ratios the user chose; the repeated current ratio in the request was
  treated as one. The cash earnings ratio is a labelled proxy. A debt-to-equity of 0 is read as no debt, a negative value as
  negative equity.
- The tab states, on the page, that four ratios are much less informative than 64 and gives both validation PR AUCs with intervals
  read from the report. The frozen 64-ratio model, calibrator and cuts are unchanged. The test split was not read.
- If `artifacts/quick/quick_model.json` is missing or inconsistent, only the Quick check tab is switched off, with an error; the
  other tabs keep working. Tests pin the file to the report, the protocol margin, monotone direction and fail-closed loading.

## 2026-10-17 (App redesign, presentation only)

- Scope: `app/streamlit_app.py` rebuilt around a result-first layout. New `src/app_ui.py` (design tokens, CSS, HTML pieces) and
  `.streamlit/config.toml` (fixed light theme, toolbar hidden). Model, calibration, cuts, rules and every generated figure are
  unchanged. The test split was not read.
- Information architecture: five equal tabs became four, named by task and scope (Quick check: 4 ratios, Full model: 64 ratios,
  Evidence, Limits and data). An intro line says which tab fits which input. How it was checked and What drives the model now sit
  under Evidence behind one switch.
- Result first: each result leads with a coloured score or band block and one plain sentence (how many firms with this score or
  band went bankrupt in the Polish data). The raw probability is a receding detail line. Quick check shows inputs beside the result.
- States: the Quick check says when it is showing the example values, offers a reset, shows plain-language meaning and the usual range
  under each input, and flags an outside-range entry on that field. Full model upload has an empty state, a file-error state with a
  fix, and a batch summary before the table. Not scored results say what to do next. Fatal stops still use `st.error`.
- Formats: proportions (rates, precision, recall, probabilities, cuts) are shown as percentages everywhere. AUCs stay decimals.
  `floor_statement` in `src/app_logic.py` and its test were changed to match. Driver charts are HTML bars with the direction written
  out, so colour is never the only cue.
- Tests: `tests/test_app_ui.py` now reads result cards through data attributes instead of `st.metric`; new tests cover the reset,
  example-value and outside-range states. `tests/test_app_ui_helpers.py` covers the HTML helpers.
- Not changed: the reliability and SHAP figures are PNGs from `src/plotting` and keep their own style.

## 2026-10-17 (US model, validation only)

- Protocol: `docs/us_protocol.md`, committed before any fit. Separate model on the 18 raw US variables, time split 1999 to 2011,
  2012 to 2014, 2015 to 2018. Not pooled with Polish. The Taiwanese file is pre-scaled to 0 to 1, and there is no Indian file.
- Finding that changed the design: the provided `status_label` is company-level (every row of a failed company is labelled failed).
  The primary label is therefore derived: positive only on a failed company's last observed year. The provided label is reported as a
  diagnostic only.
- Results are in `reports/us_validation.md`, written by `scripts/run_us_model.py`. No headline model is chosen. The test years are
  read once by `scripts/us_final_test.py` (lock written first) and have not been read.
