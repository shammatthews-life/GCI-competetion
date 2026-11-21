# Paired bootstrap report

Paired, stratified row bootstrap intervals are conditional on fitted OOF predictions and are not independent confirmation.

| Candidate | Reference | Delta | 95% low | 95% high |
|---|---|---:|---:|---:|
| B_raw_numeric_categories | A_raw_numeric | +0.008100 | +0.006634 | +0.009524 |
| C_complete_raw_rowwise | A_raw_numeric | +0.015492 | +0.013901 | +0.017208 |
| D_C_plus_keys_ohe | A_raw_numeric | +0.021143 | +0.019241 | +0.023264 |
| D_C_plus_keys_ohe | C_complete_raw_rowwise | +0.005651 | +0.004519 | +0.006779 |
| E_C_plus_keys_frequency_ohe | A_raw_numeric | +0.021910 | +0.019847 | +0.023835 |
| E_C_plus_keys_frequency_ohe | C_complete_raw_rowwise | +0.006418 | +0.005315 | +0.007590 |
| F_C_plus_supported_numeric_decompositions | A_raw_numeric | +0.016141 | +0.014570 | +0.017846 |
