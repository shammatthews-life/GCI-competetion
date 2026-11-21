# Hidden-screen product-key blend-weight supplement

- Source screen: `experiments\phase13_matched_baseline\runs\hidden_key_screen_20261105T091314Z`
- Cross-fit rate file SHA-256: `851b7c78ddd4ef6270d275d111daecc4cc0389291aa3175e61df565855cd6f5c`
- Candidate-results file SHA-256: `daff0d6cf684ddd46f32d8c7a7c7c40c60d81bf449f419d153d8c4c81343b862`
- Rows: 171,202; baseline C pooled OOF AUC: 0.759619220.
- Recomputed fixed blends for all eight positive-control keys from their saved, five-fold cross-fitted target-rate vectors. This fills the 0.05 and 0.20 weights not present for explicit keys in the initial screen output.
- These label-dependent encodings are exploratory and rule eligibility remains unresolved. The blend weights are descriptive checks on exposed OOF labels, not nested model selection.

| Key | Categories | Weight | Blend OOF AUC | Delta vs C |
|---|---:|---:|---:|---:|
| PRODUCT_KEY_Q3 | 96 | 0.05 | 0.760641 | +0.001021 |
| PRODUCT_KEY_Q3 | 96 | 0.10 | 0.761310 | +0.001691 |
| PRODUCT_KEY_Q3 | 96 | 0.20 | 0.761536 | +0.001917 |
| PRODUCT_KEY_Q4 | 797 | 0.05 | 0.761074 | +0.001454 |
| PRODUCT_KEY_Q4 | 797 | 0.10 | 0.762158 | +0.002539 |
| PRODUCT_KEY_Q4 | 797 | 0.20 | 0.763154 | +0.003535 |
| PRODUCT_KEY_Q5 | 5,102 | 0.05 | 0.760931 | +0.001311 |
| PRODUCT_KEY_Q5 | 5,102 | 0.10 | 0.761930 | +0.002310 |
| PRODUCT_KEY_Q5 | 5,102 | 0.20 | 0.762926 | +0.003307 |
| PRODUCT_KEY_KQ3 | 1,002 | 0.05 | 0.760633 | +0.001013 |
| PRODUCT_KEY_KQ3 | 1,002 | 0.10 | 0.761365 | +0.001746 |
| PRODUCT_KEY_KQ3 | 1,002 | 0.20 | 0.761925 | +0.002306 |
| PRODUCT_KEY_KQ4 | 3,576 | 0.05 | 0.760511 | +0.000892 |
| PRODUCT_KEY_KQ4 | 3,576 | 0.10 | 0.761168 | +0.001548 |
| PRODUCT_KEY_KQ4 | 3,576 | 0.20 | 0.761713 | +0.002094 |
| PRODUCT_KEY_P4 | 775 | 0.05 | 0.761376 | +0.001756 |
| PRODUCT_KEY_P4 | 775 | 0.10 | 0.762727 | +0.003107 |
| PRODUCT_KEY_P4 | 775 | 0.20 | 0.764171 | +0.004551 |
| PRODUCT_KEY_P5 | 4,079 | 0.05 | 0.761326 | +0.001707 |
| PRODUCT_KEY_P5 | 4,079 | 0.10 | 0.762656 | +0.003037 |
| PRODUCT_KEY_P5 | 4,079 | 0.20 | 0.764145 | +0.004525 |
| PRODUCT_KEY_AG3 | 131 | 0.05 | 0.759965 | +0.000346 |
| PRODUCT_KEY_AG3 | 131 | 0.10 | 0.760046 | +0.000427 |
| PRODUCT_KEY_AG3 | 131 | 0.20 | 0.759280 | -0.000339 |
