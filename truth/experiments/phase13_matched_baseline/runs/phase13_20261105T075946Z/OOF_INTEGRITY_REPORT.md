# Phase 13 OOF integrity report

- Input: `D:\gci\inpuy\train.csv` SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Rows/unique IDs: 171,202/171,202; positives: 13,821.
- Primary seed `20261113` and secondary seed `20261114`; each has complete five-fold OOF coverage.
- Primary OOF columns: 6; ablation columns: 20; shuffled-control columns: 5.
- Missing or out-of-range primary/control predictions: 0; secondary prediction missing count: 0.
- IDs and targets align to original input row order; input SHA-256 was unchanged.
- Test, sample submission, external data, holdout, and submissions were not opened/used.
- Status: PASS.
