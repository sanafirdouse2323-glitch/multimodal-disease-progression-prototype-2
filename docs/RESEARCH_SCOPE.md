# Research scope and limitations

## Prototype task

The baseline predicts the stage label at a patient's **next observed visit**, using the current visit's multimodal feature vector. Follow-up intervals are intentionally irregular. The stage labels (0–3), feature vectors and patients are synthetic.

## Advertised directions: current status

| Research direction | Current prototype status | Work still needed |
|---|---|---|
| Multimodal representation learning | Separate feed-forward encoders for toy image-derived features, wearable summaries, and clinical/tabular variables | Real imaging pipeline, domain-appropriate encoders, unimodal and multimodal baselines |
| Sparse longitudinal progression | Next-observed-visit classification with irregular visit intervals | Clinically justified fixed horizon, censoring and follow-up bias analysis |
| Missing modalities | Synthetic missing values, missingness flags, training-only imputation and scaling | MCAR/MAR/MNAR stress tests and robust training under absent modalities |
| Causal treatment effects | Not implemented in this minimal prototype | A precise causal question, time ordering, confounding assumptions and sensitivity analysis |
| Fairness and bias | Not implemented in this minimal prototype | Predefined subgroups, adequate sample sizes, uncertainty intervals, external validation and mitigation experiments |

## Data and safety

- Do not commit identifiable health data or medical images to GitHub.
- Use only authorised datasets in approved research environments with appropriate governance and ethics review.
- Synthetic stage labels are not a clinical scale.
- Synthetic performance is only a software check, not medical evidence.
- Agree with a clinical/research supervisor on the target, time zero, forecast horizon, split unit, missingness strategy and baselines before real-data work.

## Evaluation in this prototype

Patients—not visits—are split into training, validation and test sets. Imputation and scaling statistics are estimated from training examples only. Test metrics go to `results/metrics.json`; predictions go to `results/test_predictions.csv`. Results describe synthetic data only.
