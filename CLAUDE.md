# solvency-signal

Early-warning model for corporate financial distress. Trains logistic regression,
LightGBM and a PyTorch MLP (TremorNet) on the Polish companies bankruptcy data
(`5year.arff`, 1-year horizon, 5,850 firms after dedup, about 7% bankrupt),
calibrates their probabilities, and maps them to Low, Watch and High risk bands.
A Streamlit app and an Indian case study come in later phases.

## Hard rules
- Seed 42 everywhere. Fixed seeds in every script and test.
- The test split was read once in Phase 5 and is locked. Never read it again, not even for shape checks. Nothing is tuned on it.
- No performance number is written by hand. Every figure comes from code.
- No em dashes in docs. No phone number anywhere.
- Do not pick the headline model silently. Present the comparison and recommend; the user decides.

## Repo map
- `src/`: preprocess, polish_features (dictionary, categories), train, nn (TremorNet), evaluate, calibrate, explain, plotting
- `scripts/`: update_state.py (regenerates docs/STATE.md), run_phase4.py, run_phase6.py (SHAP), run_phase7_attr21.py, run_phase8_prep.py (app numbers, demo firms), run_phase9_blanks.py (blank-attribute probe), run_phase10_quick.py (4-ratio Quick check model), final_test_eval.py (one-shot, refuses a rerun)
- `app/`: streamlit_app.py, demo_firms.json, requirements.txt (logic in `src/app_logic.py`; Quick check logic in `src/quick.py`, fitted model in `artifacts/quick/quick_model.json`)
- `tests/`: pytest suite, includes the test-split guard
- `reports/`: generated reports and figures (validation only)
- `artifacts/calibration/`: fitted calibrators, band thresholds and config (JSON, committed); `phase5_protocol.json` and `phase5.lock`: frozen hashes and the one-shot lock
- `models/`: fitted models (gitignored, rebuild with `python -m src.train`)
- `data/`: raw zip and processed splits (gitignored), `data/processed/summary.json` is committed
- `docs/`: STATE.md (generated), decisions.md (log), data_card.md (when present)

## Commands (use `.venv/Scripts/python`)
- Tests: `python -m pytest -q`
- Data: `python -m src.preprocess`; train: `python -m src.train`; curves: `python -m src.evaluate`
- Phase 4: `python -m scripts.run_phase4`. Phase 5 is done and cannot be rerun (lock in `artifacts/phase5.lock`). Phase 6: `python -m scripts.run_phase6`. Phase 7: `python -m scripts.run_phase7_attr21`. Phase 8: `python -m scripts.run_phase8_prep`, `streamlit run app/streamlit_app.py`. Phase 9: `python -m scripts.run_phase9_blanks --override-ceiling` (user decision, writes the rule the app reads), then `python -m scripts.run_phase8_prep`. Phase 10: `python -m scripts.run_phase10_quick` (train and validation only).
- End of every phase: `python -m scripts.update_state --phase "<phase>"`, then commit

## Status
Headline model: LightGBM + Platt (user decision), frozen at fdef552. The neural network is the deep learning comparison, not deployed. Test result: reports/phase5_test_results.md.
Current phase: Phase 10: Quick check (4-ratio model) added to the app
Last phase commit: 47f4f98

## Pointers
For details read docs/STATE.md, docs/decisions.md, docs/data_card.md only when needed.

## Token rules
- Do not read data/raw/ or any dataset file unless the task requires it. Use saved summaries
  (`data/processed/summary.json`, `docs/STATE.md`) instead.
- Do not print full dataframes or long logs. Prefer head, shape and summary stats.
