# US model, one-shot test results (2015 to 2018)

Frozen artifact sha256 0b73477ba7606f7a. Read once. Protocol: docs/us_protocol.md.

No-skill PR AUC 0.0097, 119 positives in 12282 firm-years.

| Model | PR AUC (95% CI) | ROC AUC (95% CI) | Mean predicted | Observed |
|---|---|---|---|---|
| lightgbm | 0.099 (0.062 to 0.156) | 0.840 (0.806 to 0.875) | 0.79% | 0.97% |
| logistic | 0.120 (0.072 to 0.183) | 0.806 (0.763 to 0.846) | 0.71% | 0.97% |

| Model | Flag | Share flagged | Precision | Recall |
|---|---|---|---|---|
| lightgbm | top5 | 5.8% | 7.4% | 44.5% |
| lightgbm | top10 | 12.1% | 4.7% | 58.0% |
| logistic | top5 | 5.1% | 7.9% | 42.0% |
| logistic | top10 | 10.3% | 4.9% | 52.1% |
