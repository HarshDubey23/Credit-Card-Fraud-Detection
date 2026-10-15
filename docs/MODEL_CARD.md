# Model card: XGBoost fraud scorer

| Field | Value |
|---|---|
| Model | `xgboost.XGBClassifier`: 400 trees, depth 4, learning rate 0.05, subsample 0.8, colsample 0.8 |
| Selected by | Highest validation PR-AUC among 8 candidates ([results](../results/RESULTS.md)) |
| Input | `V1`–`V28` and `Amount` from the ULB credit-card dataset schema |
| Output | Fraud score in [0, 1]. Raw scores are reasonably calibrated on the test window (Brier 0.00042). |
| Intended use | Educational and portfolio demonstration of evaluation practice on a public benchmark |
| Not intended for | Real payment decisions. The features are anonymised PCA components that no other data source produces. |

## Evaluation

Temporal split (train on the first 60% of the two-day window, select thresholds on the
next 20%, test on the final 20%). Test set: 56,746 transactions, 74 frauds.

- PR-AUC 0.793 (95% bootstrap CI 0.702–0.875); mean 0.796 ± 0.003 over 5 seeds
- At the validation-F1 threshold: precision 0.942, recall 0.662 (95% CI 0.554–0.757)
- Not statistically distinguishable from class-weighted logistic regression (paired CI includes 0)

## Known failure modes

- Lower recall on high-value frauds (median missed amount €12.31 vs €1.10 caught; 1 of 4
  frauds ≥ €500 caught). The sample is small.
- All evidence comes from two consecutive days. Behaviour under drift is unknown.

## Ethical and practical considerations

Fraud models create false alarms that block legitimate customers. At the chosen threshold,
about 0.92 alerts are raised per 1,000 transactions. Any real use would need its own data,
fairness review across customer groups (impossible here, since there are no demographic
fields), and a human-review process.
