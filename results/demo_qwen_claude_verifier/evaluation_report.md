# QuScribe evaluation

| Configuration | Items | Accuracy | Mean fidelity | Verified precision | Verified recall | Flagged precision | Flagged recall | Mean repairs | Mean cost (USD) | Mean latency (s) | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| image_only | 16 | 0.250 | 0.291 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | n/a | 81.393 | 8 |
| image_and_text | 16 | 0.125 | 0.164 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | n/a | 117.758 | 10 |
| reference_free_repair | 16 | 0.375 | 0.375 | 1.000 | 0.833 | 0.909 | 1.000 | 0.062 | n/a | 225.689 | 11 |
| oracle_repair | 16 | 0.125 | 0.131 | n/a | n/a | n/a | n/a | 0.188 | n/a | 353.963 | 13 |
