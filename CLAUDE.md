# Early Tremor

Early-warning model for corporate financial distress. Trains logistic regression,
LightGBM and a PyTorch MLP (TremorNet) on the Polish companies bankruptcy data
(`5year.arff`, 1-year horizon, 5,850 firms after dedup, about 7% bankrupt),
calibrates their probabilities, and maps them to Low, Watch and High risk bands.
A Streamlit app and an Indian case study come in later phases.

## Hard rules
- Seed 42 everywhere. Fixed seeds in every script and test.
- The test split stays unread until the user approves. Never load it, not even for shape checks.
- No performance number is written by hand. Every figure comes from code.
- No em dashes in docs. No phone number anywhere.
- Do not pick the headline model silently. Present the comparison and recommend; the user decides.

## Repo map
- `src/`: preprocess, polish_features, train, nn (TremorNet), evaluate, calibrate, plotting
- `scripts/`: update_state.py (regenerates docs/STATE.md), run_phase4.py
- `tests/`: pytest suite, includes the test-split guard
- `reports/`: generated reports and figures (validation only)
- `artifacts/calibration/`: fitted calibrators, band thresholds and config (JSON, committed)
- `models/`: fitted models (gitignored, rebuild with `python -m src.train`)
- `data/`: raw zip and processed splits (gitignored), `data/processed/summary.json` is committed
- `docs/`: STATE.md (generated), decisions.md (log), data_card.md (when present)

## Commands (use `.venv/Scripts/python`)
- Tests: `python -m pytest -q`
- Data: `python -m src.preprocess`; train: `python -m src.train`; curves: `python -m src.evaluate`
- Phase 4: `python -m scripts.run_phase4`
- End of every phase: `python -m scripts.update_state --phase "<phase>"`, then commit

## Status
Current phase: Phase 4: calibration, thresholds and risk bands (complete, awaiting user decisions)
Last phase commit: 7d9c24f

## Pointers
For details read docs/STATE.md, docs/decisions.md, docs/data_card.md only when needed.

## Token rules
- Do not read data/raw/ or any dataset file unless the task requires it. Use saved summaries
  (`data/processed/summary.json`, `docs/STATE.md`) instead.
- Do not print full dataframes or long logs. Prefer head, shape and summary stats.
