# Monet Classification Evaluation (BINARY: AI vs not-AI)

Generated: 2026-07-23T17:26:10
Real + Suspicious are merged into 'not-AI'. The score band [0.46, 0.65) is the display 'Suspicious' zone (reported as uncertain, never as an error).
Examples evaluated: 1521 (not-AI=959, AI=562)
AI decision threshold: score >= 0.65

## Deployed-threshold metrics

- Accuracy: 89.3%
- AI precision: 98.1%
- AI recall: 72.6%
- AI F1: 83.4%
- not-AI precision: 86.1%
- not-AI recall: 99.2%
- Uncertain band [0.46, 0.65): 162 examples

## Confusion matrix (rows=true, cols=pred)
```
           not-AI    AI
  not-AI     951     8
  AI         154   408
```

## Threshold sweep

| Threshold | Accuracy | AI Precision | AI Recall | AI F1 | not-AI Recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.40 | 88.7% | 78.8% | 95.0% | 86.1% | 85.0% |
| 0.45 | 90.9% | 85.7% | 90.6% | 88.1% | 91.1% |
| 0.50 | 91.1% | 89.2% | 86.3% | 87.7% | 93.8% |
| 0.52 | 91.3% | 90.4% | 85.4% | 87.8% | 94.7% |
| 0.55 | 90.6% | 92.0% | 81.7% | 86.5% | 95.8% |
| 0.58 | 90.7% | 94.7% | 79.2% | 86.2% | 97.4% |
| 0.60 | 90.4% | 95.4% | 77.8% | 85.7% | 97.8% |
| 0.63 | 90.1% | 97.2% | 75.3% | 84.9% | 98.7% |
| 0.65 | 89.3% | 98.1% | 72.6% | 83.4% | 99.2% |
| 0.68 | 88.4% | 98.7% | 69.4% | 81.5% | 99.5% |
| 0.70 | 87.2% | 99.2% | 66.0% | 79.3% | 99.7% |
| 0.75 | 84.5% | 99.7% | 58.2% | 73.5% | 99.9% |
| 0.80 | 81.3% | 99.6% | 49.5% | 66.1% | 99.9% |
| 0.84 | 79.9% | 100.0% | 45.7% | 62.8% | 100.0% |

## Feature Importance

- vit_mean: 0.271
- vit_max: 0.221
- vit_std: 0.115
- saturation_flicker: 0.053
- brightness_flicker: 0.052
- texture_std: 0.048
- temporal_diff_mean: 0.048
- texture_mean: 0.046
- temporal_diff_std: 0.042
- texture_max: 0.032
- semantic: 0.021
- digital_penalty: 0.019
- color_mean: 0.015
- color_std: 0.011
- color_max: 0.005
- metadata_score: 0.000
- metadata_hits: 0.000

## False Positives (not-AI flagged as AI)

- cache/54 | score 0.695
- cache/60 | score 0.673
- cache/428 | score 0.662
- cache/688 | score 0.722
- cache/760 | score 0.660
- cache/844 | score 0.705
- cache/866 | score 0.826
- cache/943 | score 0.687

## False Negatives (AI missed)

- cache/964 | score 0.486
- cache/965 | score 0.524
- cache/968 | score 0.567
- cache/969 | score 0.278
- cache/970 | score 0.354
- cache/982 | score 0.538
- cache/983 | score 0.406
- cache/985 | score 0.531
- cache/987 | score 0.470
- cache/990 | score 0.348
- cache/997 | score 0.622
- cache/1010 | score 0.516
- cache/1019 | score 0.403
- cache/1024 | score 0.446
- cache/1029 | score 0.420
- cache/1030 | score 0.649
- cache/1034 | score 0.400
- cache/1035 | score 0.488
- cache/1036 | score 0.283
- cache/1038 | score 0.597
- cache/1042 | score 0.416
- cache/1045 | score 0.507
- cache/1054 | score 0.318
- cache/1060 | score 0.627
- cache/1062 | score 0.650
