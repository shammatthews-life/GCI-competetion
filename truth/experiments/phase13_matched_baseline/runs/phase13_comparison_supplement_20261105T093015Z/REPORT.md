# Phase 13 direct frequency-support comparison supplement

- Source Phase 13 run: `experiments\phase13_matched_baseline\runs\phase13_20261105T075946Z`
- OOF SHA-256: `775f11742e2be8f1bfdf75d4a34c7da7603788fb0201ec3052990cf471e6c488`
- Comparison: E (fit-fold frequency-supported one-hot) versus D (ordinary one-hot), evaluated on identical primary outer-fold rows.
- D pooled OOF AUC: 0.765270199
- E pooled OOF AUC: 0.766037689
- E-D: +0.000767490
- Paired stratified bootstrap 95% interval: [-0.000033246, +0.001512418] from 1,000 resamples.
- This interval is conditional on the fitted OOF predictions and the previously examined labels. It is not independent confirmation. The interval includes zero, so the nominal E-over-D gain is not conclusive evidence that frequency support is superior.
- The secondary-seed run compared selected E against C, not E against D; it therefore does not independently replicate the E-D contrast.
