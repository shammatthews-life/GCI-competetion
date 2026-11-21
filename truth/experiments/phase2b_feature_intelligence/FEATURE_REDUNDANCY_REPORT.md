# Feature redundancy report

- The previous formula experiment established that many ratios duplicate baseline features or differ only by a monotone scaling.
- This batch tests peer-normalized deviations and external-score interactions, not duplicate ratios.
- Peer features use fit-only medians and IQRs, with rare groups falling back to fit-only global statistics.
- Each included interaction combines an official external score with an existing loan-burden ratio; LightGBM may already learn some interactions, so the ablation tests incremental value.
- No univariate target ranking was used to select these features.
