# Source-code and artifact audit

## V16 source identity

- Executed notebook: `D:\gci\pre info\oct22\GCI_v16_everything_merged_o_78222 score.ipynb`; SHA-256 `ca77c6a6dbf6f5d1ea80b4f0f23de1f0f0c73ea7b4ac6fab937fd32ed8881306`.
- Source-only copy: `D:\exp colab\GCI_v16_everything_merged.ipynb`; SHA-256 `a1b3aec64c6730d58ad2d22f1c3a49dd0d3a9b66f05ab22054514add61f9dec3`.
- Canonical code-cell source SHA-256 for both: `0d6a0cc7be62873b66a759ffcd05d3d64e0b8e6b4dc8550990dbd829f9d09ef1`. All code-cell sources match; execution metadata/output differs.
- The executed copy has 11/11 code cells with execution counts. Saved outputs are historical notebook state, not independent evidence of a current run.
- Both copies use data discovery by filenames/row-count/schema, not pinned hashes. The local raw datasets were separately verified by this audit.

## What V16 implements

The full feature and model inventory is in `V16_FEATURE_EXTRACTION.csv` and `V16_MODEL_CONFIGURATION.csv`. In brief: rowwise missingness, external-score moments/interactions, financial and age ratios, bureau summaries, pooled category/value frequencies, pooled group deviations, K_DEC/Q/P/community key hypotheses, heuristic repeat-applicant matching, nested target encodings, learned residual/view features, four LightGBM sets, a CatBoost product model, and a fixed rank-space blend.

`TARGET` is removed before the shared train/test predictor frame is created. Target labels are intentionally used later for supervised encodings and model fitting. The V16 target encodings are nested/cross-fitted in the source; the label-free frequency/group/fee/residual transformations are nevertheless computed over pooled train+test covariates and not isolated within each outer validation fold.

## Saved output claims

- Cell 10: `old_qf_shallow fold 1/5: AUC 0.77027 (61s)` (saved output only; not independently reproduced).
- Cell 10: `old_qf_shallow fold 2/5: AUC 0.77318 (122s)` (saved output only; not independently reproduced).
- Cell 10: `old_qf_shallow fold 3/5: AUC 0.77620 (181s)` (saved output only; not independently reproduced).
- Cell 10: `old_qf_shallow fold 4/5: AUC 0.76980 (241s)` (saved output only; not independently reproduced).
- Cell 10: `old_qf_shallow fold 5/5: AUC 0.77341 (279s)` (saved output only; not independently reproduced).
- Cell 10: `old_qf_shallow: held-out AUC 0.77252` (saved output only; not independently reproduced).
- Cell 10: `old_qf_reg fold 1/5: AUC 0.76994 (91s)` (saved output only; not independently reproduced).
- Cell 10: `old_qf_reg fold 2/5: AUC 0.77380 (150s)` (saved output only; not independently reproduced).
- Cell 10: `old_qf_reg fold 3/5: AUC 0.77646 (241s)` (saved output only; not independently reproduced).
- Cell 10: `old_qf_reg fold 4/5: AUC 0.76910 (331s)` (saved output only; not independently reproduced).
- Cell 10: `old_qf_reg fold 5/5: AUC 0.77209 (422s)` (saved output only; not independently reproduced).
- Cell 10: `old_qf_reg: held-out AUC 0.77223` (saved output only; not independently reproduced).
- Cell 10: `old_v1_allfam_oh fold 1/5: AUC 0.77040 (62s)` (saved output only; not independently reproduced).
- Cell 10: `old_v1_allfam_oh fold 2/5: AUC 0.77424 (98s)` (saved output only; not independently reproduced).
- Cell 10: `old_v1_allfam_oh fold 3/5: AUC 0.77767 (160s)` (saved output only; not independently reproduced).
- Cell 10: `old_v1_allfam_oh fold 4/5: AUC 0.77060 (223s)` (saved output only; not independently reproduced).
- Cell 10: `old_v1_allfam_oh fold 5/5: AUC 0.77193 (286s)` (saved output only; not independently reproduced).
- Cell 10: `old_v1_allfam_oh: held-out AUC 0.77289` (saved output only; not independently reproduced).
- Cell 10: `old_v2_allfam_oh fold 1/5: AUC 0.77004 (63s)` (saved output only; not independently reproduced).
- Cell 10: `old_v2_allfam_oh fold 2/5: AUC 0.77542 (125s)` (saved output only; not independently reproduced).
- Cell 10: `old_v2_allfam_oh fold 3/5: AUC 0.77704 (187s)` (saved output only; not independently reproduced).
- Cell 10: `old_v2_allfam_oh fold 4/5: AUC 0.77063 (249s)` (saved output only; not independently reproduced).
- Cell 10: `old_v2_allfam_oh fold 5/5: AUC 0.77200 (312s)` (saved output only; not independently reproduced).
- Cell 10: `old_v2_allfam_oh: held-out AUC 0.77298` (saved output only; not independently reproduced).
- Cell 10: `old_v2_mono fold 1/5: AUC 0.77059 (58s)` (saved output only; not independently reproduced).
- Cell 10: `old_v2_mono fold 2/5: AUC 0.77386 (115s)` (saved output only; not independently reproduced).
- Cell 10: `old_v2_mono fold 3/5: AUC 0.77623 (171s)` (saved output only; not independently reproduced).
- Cell 10: `old_v2_mono fold 4/5: AUC 0.77070 (228s)` (saved output only; not independently reproduced).
- Cell 10: `old_v2_mono fold 5/5: AUC 0.77241 (284s)` (saved output only; not independently reproduced).
- Cell 10: `old_v2_mono: held-out AUC 0.77272` (saved output only; not independently reproduced).
- Cell 10: `e1_qf_shallow fold 1/5: AUC 0.77227 (79s)` (saved output only; not independently reproduced).
- Cell 10: `e1_qf_shallow fold 2/5: AUC 0.77522 (124s)` (saved output only; not independently reproduced).
- Cell 10: `e1_qf_shallow fold 3/5: AUC 0.77807 (203s)` (saved output only; not independently reproduced).
- Cell 10: `e1_qf_shallow fold 4/5: AUC 0.77185 (282s)` (saved output only; not independently reproduced).
- Cell 10: `e1_qf_shallow fold 5/5: AUC 0.77503 (360s)` (saved output only; not independently reproduced).
- Cell 10: `e1_qf_shallow: held-out AUC 0.77444` (saved output only; not independently reproduced).
- Cell 10: `e1_qf_reg fold 1/5: AUC 0.77028 (117s)` (saved output only; not independently reproduced).
- Cell 10: `e1_qf_reg fold 2/5: AUC 0.77508 (186s)` (saved output only; not independently reproduced).
- Cell 10: `e1_qf_reg fold 3/5: AUC 0.77762 (302s)` (saved output only; not independently reproduced).
- Cell 10: `e1_qf_reg fold 4/5: AUC 0.77034 (417s)` (saved output only; not independently reproduced).
- Cell 10: `e1_qf_reg fold 5/5: AUC 0.77466 (486s)` (saved output only; not independently reproduced).
- Cell 10: `e1_qf_reg: held-out AUC 0.77355` (saved output only; not independently reproduced).
- Cell 10: `e1_v1_allfam_oh fold 1/5: AUC 0.77054 (75s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v1_allfam_oh fold 2/5: AUC 0.77637 (116s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v1_allfam_oh fold 3/5: AUC 0.77781 (156s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v1_allfam_oh fold 4/5: AUC 0.77161 (196s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v1_allfam_oh fold 5/5: AUC 0.77300 (235s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v1_allfam_oh: held-out AUC 0.77380` (saved output only; not independently reproduced).
- Cell 10: `e1_v2_allfam_oh fold 1/5: AUC 0.77167 (41s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v2_allfam_oh fold 2/5: AUC 0.77537 (82s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v2_allfam_oh fold 3/5: AUC 0.77801 (122s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v2_allfam_oh fold 4/5: AUC 0.77100 (161s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v2_allfam_oh fold 5/5: AUC 0.77328 (203s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v2_allfam_oh: held-out AUC 0.77382` (saved output only; not independently reproduced).
- Cell 10: `e1_v2_mono fold 1/5: AUC 0.77180 (68s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v2_mono fold 2/5: AUC 0.77501 (136s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v2_mono fold 3/5: AUC 0.77826 (205s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v2_mono fold 4/5: AUC 0.77206 (274s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v2_mono fold 5/5: AUC 0.77413 (343s)` (saved output only; not independently reproduced).
- Cell 10: `e1_v2_mono: held-out AUC 0.77419` (saved output only; not independently reproduced).
- Cell 10: `e1et_qf_shallow fold 1/5: AUC 0.77309 (79s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_qf_shallow fold 2/5: AUC 0.77619 (156s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_qf_shallow fold 3/5: AUC 0.77923 (235s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_qf_shallow fold 4/5: AUC 0.77362 (279s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_qf_shallow fold 5/5: AUC 0.77500 (324s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_qf_shallow: held-out AUC 0.77538` (saved output only; not independently reproduced).
- Cell 10: `e1et_qf_reg fold 1/5: AUC 0.77192 (120s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_qf_reg fold 2/5: AUC 0.77676 (238s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_qf_reg fold 3/5: AUC 0.77897 (356s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_qf_reg fold 4/5: AUC 0.77229 (424s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_qf_reg fold 5/5: AUC 0.77560 (543s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_qf_reg: held-out AUC 0.77507` (saved output only; not independently reproduced).
- Cell 10: `e1et_v1_allfam_oh fold 1/5: AUC 0.77223 (75s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v1_allfam_oh fold 2/5: AUC 0.77662 (115s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v1_allfam_oh fold 3/5: AUC 0.77877 (155s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v1_allfam_oh fold 4/5: AUC 0.77210 (195s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v1_allfam_oh fold 5/5: AUC 0.77495 (235s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v1_allfam_oh: held-out AUC 0.77489` (saved output only; not independently reproduced).
- Cell 10: `e1et_v2_allfam_oh fold 1/5: AUC 0.77253 (76s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v2_allfam_oh fold 2/5: AUC 0.77656 (124s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v2_allfam_oh fold 3/5: AUC 0.77954 (164s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v2_allfam_oh fold 4/5: AUC 0.77323 (204s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v2_allfam_oh fold 5/5: AUC 0.77483 (245s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v2_allfam_oh: held-out AUC 0.77529` (saved output only; not independently reproduced).
- Cell 10: `e1et_v2_mono fold 1/5: AUC 0.77257 (63s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v2_mono fold 2/5: AUC 0.77684 (125s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v2_mono fold 3/5: AUC 0.77964 (189s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v2_mono fold 4/5: AUC 0.77313 (252s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v2_mono fold 5/5: AUC 0.77496 (314s)` (saved output only; not independently reproduced).
- Cell 10: `e1et_v2_mono: held-out AUC 0.77538` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_qf_shallow fold 1/5: AUC 0.77312 (48s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_qf_shallow fold 2/5: AUC 0.77642 (92s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_qf_shallow fold 3/5: AUC 0.77950 (169s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_qf_shallow fold 4/5: AUC 0.77352 (246s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_qf_shallow fold 5/5: AUC 0.77522 (291s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_qf_shallow: held-out AUC 0.77551` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_qf_reg fold 1/5: AUC 0.77217 (117s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_qf_reg fold 2/5: AUC 0.77660 (235s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_qf_reg fold 3/5: AUC 0.77925 (303s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_qf_reg fold 4/5: AUC 0.77222 (421s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_qf_reg fold 5/5: AUC 0.77568 (539s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_qf_reg: held-out AUC 0.77515` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v1_allfam_oh fold 1/5: AUC 0.77231 (74s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v1_allfam_oh fold 2/5: AUC 0.77674 (114s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v1_allfam_oh fold 3/5: AUC 0.77907 (153s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v1_allfam_oh fold 4/5: AUC 0.77180 (193s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v1_allfam_oh fold 5/5: AUC 0.77516 (233s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v1_allfam_oh: held-out AUC 0.77496` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v2_allfam_oh fold 1/5: AUC 0.77272 (76s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v2_allfam_oh fold 2/5: AUC 0.77642 (115s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v2_allfam_oh fold 3/5: AUC 0.77995 (155s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v2_allfam_oh fold 4/5: AUC 0.77269 (194s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v2_allfam_oh fold 5/5: AUC 0.77519 (234s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v2_allfam_oh: held-out AUC 0.77534` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v2_mono fold 1/5: AUC 0.77279 (63s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v2_mono fold 2/5: AUC 0.77700 (125s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v2_mono fold 3/5: AUC 0.78003 (188s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v2_mono fold 4/5: AUC 0.77301 (250s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v2_mono fold 5/5: AUC 0.77529 (312s)` (saved output only; not independently reproduced).
- Cell 10: `e1Aet_v2_mono: held-out AUC 0.77557` (saved output only; not independently reproduced).
- Cell 12: `CatBoost-product fold 1/5: AUC 0.77035 (534s)` (saved output only; not independently reproduced).
- Cell 12: `CatBoost-product fold 2/5: AUC 0.77315 (1068s)` (saved output only; not independently reproduced).
- Cell 12: `CatBoost-product fold 3/5: AUC 0.77599 (1603s)` (saved output only; not independently reproduced).
- Cell 12: `CatBoost-product fold 4/5: AUC 0.76990 (2133s)` (saved output only; not independently reproduced).
- Cell 12: `CatBoost-product fold 5/5: AUC 0.77135 (2663s)` (saved output only; not independently reproduced).
- Cell 15: `held-out AUC of the champion formula (weights fixed in advance): 0.77649   [v15 reference without partner/community keys: 0.77404]` (saved output only; not independently reproduced).
- Cell 15: `old    blend held-out AUC 0.77412` (saved output only; not independently reproduced).
- Cell 15: `e1     blend held-out AUC 0.77512` (saved output only; not independently reproduced).
- Cell 15: `e1et   blend held-out AUC 0.77608` (saved output only; not independently reproduced).
- Cell 15: `e1Aet  blend held-out AUC 0.77618` (saved output only; not independently reproduced).
- Cell 17: `fold 1 {'base (no label blocks)': np.float64(0.75874), '+ v15 product keys': np.float64(0.76929), '+ community CT keys': np.float64(0.76955), '+ community CT keys + partner': np.float64(0.77098)}` (saved output only; not independently reproduced).
- Cell 17: `fold 2 {'base (no label blocks)': np.float64(0.76169), '+ v15 product keys': np.float64(0.77405), '+ community CT keys': np.float64(0.77401), '+ community CT keys + partner': np.float64(0.77438)}` (saved output only; not independently reproduced).
- Cell 17: `fold 3 {'base (no label blocks)': np.float64(0.76721), '+ v15 product keys': np.float64(0.77654), '+ community CT keys': np.float64(0.77674), '+ community CT keys + partner': np.float64(0.77755)}` (saved output only; not independently reproduced).
- Cell 17: `fold 4 {'base (no label blocks)': np.float64(0.75694), '+ v15 product keys': np.float64(0.76895), '+ community CT keys': np.float64(0.76919), '+ community CT keys + partner': np.float64(0.77092)}` (saved output only; not independently reproduced).
- Cell 17: `fold 5 {'base (no label blocks)': np.float64(0.76166), '+ v15 product keys': np.float64(0.77042), '+ community CT keys': np.float64(0.7708), '+ community CT keys + partner': np.float64(0.77153)}` (saved output only; not independently reproduced).
- Cell 17: `base (no label blocks)             mean AUC 0.76125 | delta vs base +0.00000 | folds improved by >=0.0002: 0/5` (saved output only; not independently reproduced).
- Cell 17: `+ v15 product keys                 mean AUC 0.77185 | delta vs base +0.01060 | folds improved by >=0.0002: 5/5` (saved output only; not independently reproduced).
- Cell 17: `+ community CT keys                mean AUC 0.77206 | delta vs base +0.01081 | folds improved by >=0.0002: 5/5` (saved output only; not independently reproduced).
- Cell 17: `+ community CT keys + partner      mean AUC 0.77307 | delta vs base +0.01182 | folds improved by >=0.0002: 5/5` (saved output only; not independently reproduced).
- Cell 19: `(a) permutation test held-out AUC = 0.5014  (expect ~0.50)` (saved output only; not independently reproduced).
- Cell 19: `(b) adversarial AUC = 0.6856  (0.50 = identical)` (saved output only; not independently reproduced).
- Cell 19: `(c) full champion held-out AUC 0.77649` (saved output only; not independently reproduced).
- Cell 21: `cross-fitted regime-aware blend OOF AUC 0.77648  vs fixed-weight 0.9/0.1 blend 0.77649` (saved output only; not independently reproduced).

The notebook output is not tied to input hashes, cache hashes, or a cache manifest. Therefore its displayed AUC is not reproduced here.

## V16 cache status

The configured cache folders and V15 fallback caches were absent from the audited local paths. V16 loads model and CatBoost OOF/test cache files by filename without verifying their hashes, data identity, feature schema, fold assignment, or model configuration. This prevents a faithful replay from those notebook cells as currently supplied.
