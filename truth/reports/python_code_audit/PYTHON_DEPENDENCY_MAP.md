# Python dependency map

Read-only static import and artifact-reference map. Modules were not imported or executed.
Detected imports are not proof that the module is required on a given runtime path.

| Source file | Local imports | Other imports | Unavailable imports | Literal artifacts |
|---|---|---|---|---|
| `D:\gci\5 independent\build_precision_transductive_apex_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time | — | cache/hasan_expanded_zoo.npz [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\5 independent\train_catboost_and_deep_lgb.py` | — | catboost, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\5 independent\train_exp15_gpu_xgboost.py` | — | gc, numpy, os, pandas, pickle, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | D:/gci/codex/v_3_cache/resid16.pkl [EXISTS]; D:/gci/codex/v_3_cache/e1A_views.pkl [EXISTS] |
| `D:\gci\5 independent\train_full_v16_suite.py` | — | gc, lightgbm, numpy, os, pandas, pickle, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings | — | D:/gci/codex/v_3_cache/resid16.pkl [EXISTS] |
| `D:\gci\5 independent\train_v58_clean_gpu_0783_champion.py` | — | numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, time | — | not statically resolved |
| `D:\gci\champion\01_EXP85_RECOVERY\notebooks\exp85_recovery_v5.ipynb` | — | catboost, copy, hashlib, lightgbm, math, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, torch, warnings, xgboost | — | not statically resolved |
| `D:\gci\champion\01_EXP85_RECOVERY\source\evaluate_reproductions.py` | — | hashlib, numpy, os, pandas, pathlib, scipy.stats, sys | — | not statically resolved |
| `D:\gci\champion\01_EXP85_RECOVERY\source\reproduce_anchor74.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys | — | not statically resolved |
| `D:\gci\champion\01_EXP85_RECOVERY\source\reproduce_exact_synthesis.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\notebooks\exp85_recovery_v5.ipynb` | — | catboost, copy, hashlib, lightgbm, math, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, torch, warnings, xgboost | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\generate_anchor74_from_scratch.py` | — | argparse, catboost, hashlib, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\generate_v17_from_scratch.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\generate_v55_from_scratch.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, re, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\generate_xgb10_from_scratch.py` | — | argparse, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\reconstruct_transductive_pyramid.py` | — | hashlib, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sys, time, warnings | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\recreate_precision_anchor.py` | — | argparse, catboost, hashlib, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\recreate_v17_bundle.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\recreate_v55_members.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, re, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\recreate_xgb10_and_labels.py` | — | argparse, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\run_all_recreations.py` | — | argparse, hashlib, json, numpy, pandas, pathlib, subprocess, sys, time, uuid | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\run_hasan_600_builder.py` | — | lightgbm, numpy, os, pandas, scipy.stats, sklearn.model_selection, sys, warnings | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\run_workstream2_screening.py` | — | hashlib, lightgbm, numpy, os, pandas, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\champion\02_EXP85_REBUILT\source\train_independent_zoo.py` | — | catboost, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\champion\03_PHASE_HISTORY\PHASE_1\audit_historical_artifacts.py` | — | hashlib, numpy, os, pandas, pathlib, scipy.stats, sys | — | D:/gci/gemini/champion/dependency_audit/EXP85_DEPENDENCY_INVENTORY.csv [EXISTS] |
| `D:\gci\champion\03_PHASE_HISTORY\PHASE_1\evaluate_reproductions.py` | — | hashlib, numpy, os, pandas, pathlib, scipy.stats, sys | — | not statically resolved |
| `D:\gci\champion\03_PHASE_HISTORY\PHASE_1\inventory_scanner.py` | — | csv, hashlib, os, pathlib, sys, time | — | not statically resolved |
| `D:\gci\champion\03_PHASE_HISTORY\PHASE_1\reproduce_exact_synthesis.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | not statically resolved |
| `D:\gci\champion\03_PHASE_HISTORY\PHASE_1\run_branch_ablation.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS]; D:/gci/gemini/champion/cacherecreation/submission_exp85_recovery_v5_best.csv [EXISTS] |
| `D:\gci\champion\03_PHASE_HISTORY\PHASE_2\reproduce_anchor74.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys | — | not statically resolved |
| `D:\gci\champion\03_PHASE_HISTORY\PHASE_3\experiments\create_builder.py` | — | os | — | not statically resolved |
| `D:\gci\champion\03_PHASE_HISTORY\PHASE_3\experiments\reconstruct_hasan_features.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time, warnings | — | not statically resolved |
| `D:\gci\champion\03_PHASE_HISTORY\PHASE_3\experiments\reconstruct_transductive_pyramid.py` | — | hashlib, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sys, time, warnings | — | not statically resolved |
| `D:\gci\champion\03_PHASE_HISTORY\PHASE_3\experiments\run_hasan_600_builder.py` | — | lightgbm, numpy, os, pandas, scipy.stats, sklearn.model_selection, sys, warnings | — | not statically resolved |
| `D:\gci\champion\03_PHASE_HISTORY\PHASE_3\experiments\run_workstream2_screening.py` | — | hashlib, lightgbm, numpy, os, pandas, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\champion\03_PHASE_HISTORY\PHASE_3\experiments\train_independent_zoo.py` | — | catboost, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\champion\04_MODEL_LINEAGE\ANCHOR74\generate_anchor74_from_scratch.py` | — | argparse, catboost, hashlib, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | not statically resolved |
| `D:\gci\champion\04_MODEL_LINEAGE\V17\generate_v17_from_scratch.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\champion\04_MODEL_LINEAGE\V55\generate_v55_from_scratch.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, re, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\champion\04_MODEL_LINEAGE\XGB10\generate_xgb10_from_scratch.py` | — | argparse, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | not statically resolved |
| `D:\gci\champion\05_FEATURE_ENGINEERING\FEATURE_BUILDERS\build_hasan_600_features.py` | — | lightgbm, numpy, os, pandas, scipy.stats, sklearn.model_selection, sys, warnings | — | not statically resolved |
| `D:\gci\champion\05_FEATURE_ENGINEERING\FEATURE_BUILDERS\screen_and_select_features.py` | — | lightgbm, numpy, os, pandas, sklearn.metrics, sklearn.model_selection | — | cache/hasan_features_tr.feather [EXISTS]; cache/hasan_features_te.feather [EXISTS]; cache/y_train.npy [EXISTS] |
| `D:\gci\champion\05_FEATURE_ENGINEERING\FEATURE_VALIDATION\nested_cv_feature_screening_benchmark.py` | — | argparse, lightgbm, numpy, os, pandas, pathlib, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\champion\06_REPRODUCIBILITY\generate_inventory.py` | — | hashlib, os | — | not statically resolved |
| `D:\gci\champion\create_exp85_submissions.py` | — | hashlib, json, numpy, os, pandas, pathlib, scipy.stats, sys, time | — | not statically resolved |
| `D:\gci\champion\create_submission_recovery_v5.py` | — | hashlib, numpy, os, pandas, pathlib, scipy.stats, sys, time | — | not statically resolved |
| `D:\gci\champion\create_submission_recovery_v5_rebuilt.py` | — | argparse, hashlib, json, numpy, os, pandas, pathlib, scipy.stats, sys, time | — | not statically resolved |
| `D:\gci\champion\EXP85_DEEP_RESEARCH_AND_IMPROVEMENT\source\build_manifests.py` | — | __future__, datetime, hashlib, json, pathlib | — | not statically resolved |
| `D:\gci\champion\EXP85_DEEP_RESEARCH_AND_IMPROVEMENT\source\correlation_gap_diagnostics.py` | gap_diagnostics | __future__, argparse, datetime, hashlib, json, numpy, pandas, pathlib, platform, re, scipy, sklearn, subprocess, sys, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_DEEP_RESEARCH_AND_IMPROVEMENT\source\gap_diagnostics.py` | — | __future__, numpy, pandas, scipy.stats, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_DEEP_RESEARCH_AND_IMPROVEMENT\source\phase2_category_keys.py` | — | __future__, numpy, pandas | — | not statically resolved |
| `D:\gci\champion\EXP85_DEEP_RESEARCH_AND_IMPROVEMENT\source\phase2_group_context.py` | — | __future__, numpy, pandas | — | not statically resolved |
| `D:\gci\champion\EXP85_DEEP_RESEARCH_AND_IMPROVEMENT\source\run_category_te_experiment.py` | phase2_category_keys, run_group_context_experiment | __future__, argparse, datetime, hashlib, json, numpy, pandas, pathlib, sklearn.metrics, sys, time, traceback, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_DEEP_RESEARCH_AND_IMPROVEMENT\source\run_group_context_experiment.py` | phase2_group_context | __future__, argparse, datetime, hashlib, importlib.util, json, numpy, pandas, pathlib, sklearn.metrics, subprocess, sys, threading, time, traceback, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_DEEP_RESEARCH_AND_IMPROVEMENT\source\verify_manifests.py` | build_manifests | __future__, json, pathlib | — | not statically resolved |
| `D:\gci\champion\EXP85_DEEP_RESEARCH_AND_IMPROVEMENT\tests\test_gap_diagnostics.py` | gap_diagnostics | __future__, numpy, pandas, pathlib, sys, unittest | — | not statically resolved |
| `D:\gci\champion\EXP85_DEEP_RESEARCH_AND_IMPROVEMENT\tests\test_phase2_category_keys.py` | phase2_category_keys | __future__, pandas, pathlib, sys, unittest | — | not statically resolved |
| `D:\gci\champion\EXP85_DEEP_RESEARCH_AND_IMPROVEMENT\tests\test_phase2_group_context.py` | phase2_group_context | __future__, numpy, pandas, pathlib, sys, unittest | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\correlation_failure_investigation\analyze_existing_artifacts.py` | — | __future__, datetime, hashlib, json, numpy, pandas, pathlib, scipy.stats | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\build_asymmetric_convex_gated_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\build_hasan_600_features.py` | — | lightgbm, numpy, os, pandas, scipy.stats, sklearn.model_selection, sys, warnings | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\build_level2_regime_meta_stacker.py` | — | hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; v_solo_080200_grand_master_apex.csv [EXISTS]; v_solo_080041_pure_champion.csv [EXISTS] |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\build_optimal_transductive_apex_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; cache/hasan_expanded_zoo.npz [EXISTS] |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\build_precision_transductive_apex_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time | — | cache/hasan_expanded_zoo.npz [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\cacherecreation\recreate_precision_anchor.py` | — | argparse, catboost, hashlib, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\cacherecreation\recreate_v17_bundle.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\cacherecreation\recreate_v55_members.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\cacherecreation\recreate_xgb10_and_labels.py` | — | argparse, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\dependency_audit\anchor74_audit\reproduce_anchor74.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\exp85_solo_runner.py` | — | catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/oof_10f_xgb.npy [EXISTS]; cache/test_10f_xgb.npy [EXISTS] |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\phase3_reconstruction\experiments\create_builder.py` | — | os | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\phase3_reconstruction\experiments\reconstruct_hasan_features.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time, warnings | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\phase3_reconstruction\experiments\reconstruct_transductive_pyramid.py` | — | hashlib, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sys, time, warnings | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\phase3_reconstruction\experiments\run_hasan_600_builder.py` | — | lightgbm, numpy, os, pandas, scipy.stats, sklearn.model_selection, sys, warnings | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\phase3_reconstruction\experiments\run_workstream2_screening.py` | — | hashlib, lightgbm, numpy, os, pandas, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\phase3_reconstruction\experiments\train_independent_zoo.py` | — | catboost, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\dependent\exp85_solo_runner_independent.py` | — | argparse, catboost, hashlib, json, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\screen_and_select_features.py` | — | lightgbm, numpy, os, pandas, sklearn.metrics, sklearn.model_selection | — | cache/hasan_features_tr.feather [EXISTS]; cache/hasan_features_te.feather [EXISTS]; cache/y_train.npy [EXISTS] |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\train_catboost_and_deep_lgb.py` | — | catboost, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\train_hasan_master_zoo.py` | — | catboost, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, warnings, xgboost | — | cache/hasan_selected_550_tr.feather [EXISTS]; cache/hasan_selected_550_te.feather [EXISTS]; cache/y_train.npy [EXISTS]; cache/oof_lgb_cons.npy [EXISTS]; cache/test_lgb_cons.npy [EXISTS]; cache/oof_lgb_et.npy [EXISTS]; cache/test_lgb_et.npy [EXISTS] |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\historical_lineage_bridge\source\analyze_production_feature_capture.py` | — | __future__, argparse, ast, hashlib, json, numpy, pandas, pathlib, re, time | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\historical_lineage_bridge\source\audit_historical_lineage.py` | — | __future__, csv, datetime, hashlib, json, numpy, pandas, pathlib, scipy.stats, subprocess | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\historical_lineage_bridge\source\capture_production_anchor_features.py` | — | __future__, argparse, datetime, hashlib, importlib.util, json, pandas, pathlib, time | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\historical_lineage_bridge\source\finalize_production_feature_audit.py` | — | __future__, csv, hashlib, json, numpy, pandas, pathlib, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\historical_lineage_bridge\tests\test_lineage_evidence.py` | — | csv, json, pathlib, unittest | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\src\__init__.py` | — | — | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\src\build_track1b.py` | common | __future__, argparse, json, numpy, pandas, pathlib, scipy.stats | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\src\common.py` | — | __future__, catboost, hashlib, json, lightgbm, numpy, os, pandas, pathlib, sys, time, xgboost | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\src\generate_anchor74.py` | common | __future__, argparse, catboost, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, xgboost | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\src\generate_recovered_component.py` | common | __future__, argparse, contextlib, importlib.util, io, json, numpy, pandas, pathlib, sys, time | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\src\generate_v17.py` | common | __future__, argparse, catboost, lightgbm, numpy, pandas, pathlib, re, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\src\generate_v55.py` | common | __future__, argparse, catboost, lightgbm, numpy, pandas, pathlib, re, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\src\generate_xgb10.py` | common | __future__, argparse, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, time, xgboost | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\src\generate_xgb10_recovered.py` | common | __future__, argparse, importlib.util, json, numpy, pathlib, sklearn.metrics, sys, time | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\src\run_all.py` | common | __future__, argparse, json, pathlib, subprocess, sys | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\tests\__init__.py` | — | — | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\tests\test_gpu_smoke.py` | src.common | __future__, pathlib, sys | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\validation\__init__.py` | — | — | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\validation\validate_artifacts.py` | — | __future__, argparse, json, numpy, pathlib | — | not statically resolved |
| `D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\validation\validate_gpu.py` | src.common | __future__, sys | — | not statically resolved |
| `D:\gci\champion\EXP85_ISOLATED_REPAIR\scripts\audit_prediction_correlations.py` | — | __future__, argparse, hashlib, json, numpy, pandas, pathlib, scipy.stats, sys | — | not statically resolved |
| `D:\gci\champion\EXP85_ISOLATED_REPAIR\scripts\create_verified_rebuild.py` | — | __future__, argparse, hashlib, json, os, pathlib, subprocess, sys, time | — | not statically resolved |
| `D:\gci\champion\EXP85_ISOLATED_REPAIR\scripts\test_prediction_correlations.py` | — | __future__, json, numpy, pandas, pathlib, subprocess, sys, tempfile | — | not statically resolved |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\PHASE3_RAW_LINEAGE_RECOVERY\runs\v080200_transform_replay_20261031_213954\source_snapshot\build_v80200_apex.py` | — | hashlib, numpy, os, pandas, scipy.stats | — | v_solo_080150_micro_apex_fusion.csv [EXISTS]; ../inpuy/train.csv [EXISTS]; ../inpuy/test.csv [EXISTS] |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\PHASE3_RAW_LINEAGE_RECOVERY\source\catboost_controlled_experiment.py` | — | __future__, argparse, catboost, hashlib, json, numpy, pandas, pathlib, platform, scipy.stats, sklearn.metrics, sklearn.model_selection, subprocess, sys, time, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\PHASE3_RAW_LINEAGE_RECOVERY\source\replay_v080200_transform.py` | — | __future__, argparse, datetime, hashlib, json, numpy, pandas, pathlib, scipy.stats, shutil, subprocess, sys, time, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\PHASE3_RAW_LINEAGE_RECOVERY\tests\test_replay_v080200_transform.py` | replay_v080200_transform | __future__, numpy, pandas, pathlib, sys, unittest | — | not statically resolved |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\source\compare_historical_zoo.py` | — | __future__, argparse, datetime, hashlib, json, numpy, pandas, pathlib, scipy.stats, sklearn.model_selection, time, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\source\HISTORICAL_FEATURE_BUILDER.py` | — | __future__, argparse, ast, datetime, hashlib, importlib.util, json, numpy, pandas, pathlib, scipy.stats, sklearn.model_selection, time, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\source\isolated_legacy\build_asymmetric_convex_gated_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\source\isolated_legacy\build_asymmetric_historical_zoo.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\source\reconstruct_anchor74_only.py` | — | __future__, argparse, datetime, hashlib, json, numpy, pandas, pathlib, scipy.stats, sys, time, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\source\reconstruct_historical_anchor74.py` | — | __future__, argparse, datetime, hashlib, json, numpy, pandas, pathlib, scipy.stats, sys, time, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\source\replay_historical_selector.py` | HISTORICAL_FEATURE_BUILDER | __future__, argparse, datetime, hashlib, json, lightgbm, numpy, pandas, pathlib, sklearn, sklearn.metrics, sklearn.model_selection, time, typing | — | D:\gci\gemini\cache\hasan_selected_550_tr.feather [EXISTS] |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\source\train_historical_anchor_zoo.py` | HISTORICAL_FEATURE_BUILDER | __future__, argparse, catboost, datetime, json, lightgbm, numpy, pandas, pathlib, sklearn, sklearn.metrics, sklearn.model_selection, subprocess, time, typing, xgboost | — | not statically resolved |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\tests\test_historical_anchor74.py` | — | importlib.util, numpy, pandas, pathlib, scipy.stats, unittest | — | not statically resolved |
| `D:\gci\champion\EXP85_MASTER_RECOVERY\tests\test_historical_features.py` | — | __future__, importlib.util, json, numpy, pandas, pathlib, unittest | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\REPRODUCTION\test_verify_freeze.py` | verify_freeze | __future__, csv, pathlib, tempfile, unittest | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\REPRODUCTION\update_freeze_manifest.py` | — | __future__, datetime, hashlib, json, pathlib, subprocess | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\REPRODUCTION\verify_freeze.py` | — | __future__, argparse, csv, hashlib, json, math, pathlib, sys, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\experiments\phase_01_fee_product_features\run_experiment.py` | run_independent_baseline | __future__, argparse, datetime, hashlib, json, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, sys, time, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\experiments\phase_01_fee_product_features\test_decoded_loan_features.py` | run_experiment | numpy, pandas, pathlib, sys, unittest | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\experiments\phase_02_fold_safe_key_encodings\run_experiment.py` | run_independent_baseline | __future__, argparse, datetime, hashlib, importlib.util, json, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, sys, time, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\experiments\phase_02_fold_safe_key_encodings\test_target_encodings.py` | run_experiment | numpy, pathlib, sys, unittest | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\experiments\phase_04_final_validation_gate\run_final_gate.py` | run_independent_baseline | __future__, argparse, catboost, copy, datetime, gc, hashlib, importlib.util, json, lightgbm, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, subprocess, sys, threading, time, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\experiments\phase_04_final_validation_gate\run_frequency_ablation.py` | run_final_gate, run_independent_baseline | __future__, argparse, catboost, copy, datetime, gc, hashlib, json, lightgbm, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, sys, time, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\experiments\phase_04_final_validation_gate\test_final_gate.py` | run_final_gate, run_frequency_ablation | numpy, pandas, pathlib, sklearn.model_selection, sys, unittest | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\experiments\phase_04_final_validation_gate\verify_final_gate.py` | — | __future__, argparse, hashlib, json, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\source\run_independent_baseline.py` | — | __future__, argparse, catboost, datetime, hashlib, json, lightgbm, numpy, os, pandas, pathlib, scipy.special, scipy.stats, sklearn, sklearn.metrics, sklearn.model_selection, subprocess, sys, time, traceback, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\source\validate_experiment_artifacts.py` | — | __future__, argparse, hashlib, json, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, time, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\tests\test_independent_features.py` | run_independent_baseline | json, numpy, pandas, pathlib, sys, unittest | — | not statically resolved |
| `D:\gci\champion\exp85_solo.ipynb` | — | catboost, copy, hashlib, lightgbm, math, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, torch, warnings, xgboost | — | not statically resolved |
| `D:\gci\champion\exp85_solo.pre_independent_wiring_backup.ipynb` | — | __future__, argparse, catboost, hashlib, json, lightgbm, numpy, os, pandas, pathlib, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/test_10f_xgb.npy [EXISTS] |
| `D:\gci\champion\exp85_solo_backup_pre_fix.ipynb` | — | catboost, hashlib, lightgbm, numpy, os, pandas, pickle, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, subprocess, sys, time, warnings, xgboost | — | cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/oof_10f_xgb.npy [EXISTS]; cache/test_10f_xgb.npy [EXISTS] |
| `D:\gci\champion\exp85_solo_runner.py` | — | catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/test_10f_xgb.npy [EXISTS] |
| `D:\gci\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\analyze_existing_artifacts.py` | — | __future__, datetime, hashlib, json, numpy, pandas, pathlib, scipy.stats | — | not statically resolved |
| `D:\gci\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\build_manifests.py` | — | __future__, datetime, hashlib, json, pathlib, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\correlation_runner.py` | — | __future__, argparse, datetime, hashlib, json, numpy, pandas, pathlib, scipy.stats, subprocess, sys, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\create_submission_recovery_v5_rebuilt.py` | — | argparse, hashlib, json, numpy, os, pandas, pathlib, scipy.stats, sys, time | — | not statically resolved |
| `D:\gci\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\exp85_recovery_v5_original.ipynb` | — | catboost, copy, hashlib, lightgbm, math, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, torch, warnings, xgboost | — | not statically resolved |
| `D:\gci\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\original_isolated_audit\audit_prediction_correlations.py` | — | __future__, argparse, hashlib, json, numpy, pandas, pathlib, scipy.stats, sys | — | not statically resolved |
| `D:\gci\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\original_isolated_audit\test_prediction_correlations.py` | — | __future__, json, numpy, pandas, pathlib, subprocess, sys, tempfile | — | not statically resolved |
| `D:\gci\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\reconstruct_historical_v5.py` | — | __future__, argparse, datetime, hashlib, json, numpy, pandas, pathlib, re, scipy.stats, sys, time, traceback, typing | — | not statically resolved |
| `D:\gci\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\reconstruct_transductive_pyramid.py` | — | hashlib, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sys, time, warnings | — | not statically resolved |
| `D:\gci\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\source\verify_replication.py` | — | __future__, hashlib, json, pathlib | — | not statically resolved |
| `D:\gci\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\tests\test_correlation_runner.py` | correlation_runner | __future__, pandas, pathlib, sys, tempfile, unittest | — | not statically resolved |
| `D:\gci\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION\tests\test_historical_v5_reconstruction.py` | reconstruct_historical_v5 | __future__, numpy, pandas, pathlib, scipy.stats, sys, unittest | — | not statically resolved |
| `D:\gci\codex\analyze_v15_ca_catboost_combinations.py` | — | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\codex\analyze_v15_ca_catboost_gate.py` | — | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\codex\analyze_v15_ca_catboost_seed_blend.py` | — | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\codex\analyze_v15_ca_catboost_weights.py` | — | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\codex\analyze_v15_ca_ext_fresh_weight.py` | — | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\codex\assess_v4_v17_blend.py` | — | hashlib, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\codex\combine_v15_id_ca_candidates.py` | — | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\codex\confirm_v15_annuity_income_gate.py` | — | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\codex\confirm_v15_id_ca_combination.py` | — | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\codex\profile_repeat_diagnostic.py` | — | json, pandas, pathlib | — | not statically resolved |
| `D:\gci\codex\run_claude_5_diversity_diagnostic.py` | — | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\codex\run_claude_5_feature_hypothesis_screen.py` | — | json, lightgbm, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_claude_6_fresh_diversity_confirmation.py` | run_claude_5_feature_hypothesis_screen | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_claude_7_matched_v15_specialists.py` | run_claude_5_feature_hypothesis_screen | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_train_test_shift_audit.py` | — | __future__, json, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_applicant_profile_catboost.py` | — | __future__, catboost, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_ca_catboost_only.py` | — | __future__, catboost, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_ca_context_catboost.py` | — | __future__, catboost, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_ca_ext_alpha50.py` | — | __future__, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_ca_ext_catboost.py` | — | __future__, catboost, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_ca_ext_confirmation.py` | — | __future__, catboost, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_ca_ext_experiment.py` | — | __future__, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_conditional_blend_diagnostic.py` | — | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\codex\run_v15_dart_member.py` | — | __future__, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_driver_specialist.py` | — | __future__, json, lightgbm, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_employment_consistency_features.py` | — | __future__, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_ext3_normalization_features.py` | — | __future__, json, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_financial_monotone.py` | — | __future__, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_hierarchical_backoff_experiment.py` | — | __future__, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_high_credit_specialist.py` | — | __future__, json, lightgbm, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_id_context_experiment.py` | — | __future__, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_missing_ext_specialist.py` | — | __future__, json, lightgbm, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_peer_knn_experiment.py` | — | json, numpy, pandas, pathlib, scipy.sparse, scipy.stats, sklearn.metrics, sklearn.model_selection, sklearn.neighbors, sklearn.preprocessing, time | — | not statically resolved |
| `D:\gci\codex\run_v15_peer_rank_experiment.py` | — | __future__, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_product_frequency_gate.py` | run_claude_7_matched_v15_specialists | __future__, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_profile_target_rate.py` | — | json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_region_population_catboost.py` | — | __future__, catboost, json, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_region_population_target_rate.py` | — | __future__, json, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_spline_scorecard.py` | run_claude_5_feature_hypothesis_screen, run_claude_7_matched_v15_specialists | __future__, json, numpy, pandas, pathlib, scipy.stats, sklearn.compose, sklearn.impute, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.pipeline, sklearn.preprocessing, time | — | not statically resolved |
| `D:\gci\codex\run_v15_temporal_consistency_features.py` | — | __future__, json, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v15_winner_feature_combo.py` | — | __future__, json, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sklearn.neighbors, time | — | not statically resolved |
| `D:\gci\codex\run_v15_winner_neighbor_feature.py` | — | __future__, json, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sklearn.neighbors, time | — | not statically resolved |
| `D:\gci\codex\run_v15_young_applicant_specialist.py` | — | __future__, json, lightgbm, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v1_conditional_confirmation.py` | — | json, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v2_submission.py` | — | hashlib, json, lightgbm, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\codex\run_v3_standalone_v17.py` | — | catboost, hashlib, json, lightgbm, numpy, os, pandas, pathlib, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\gemini\analyze_connected_components.py` | — | numpy, pandas, scipy.sparse, scipy.sparse.csgraph | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\analyze_ext_conflicts.py` | — | numpy, pandas, sklearn.metrics | — | d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\append_exp64.py` | — | os | — | not statically resolved |
| `D:\gci\gemini\append_exp65.py` | — | os | — | not statically resolved |
| `D:\gci\gemini\append_exp66.py` | — | os | — | not statically resolved |
| `D:\gci\gemini\append_exp67.py` | — | os | — | not statically resolved |
| `D:\gci\gemini\append_exp68.py` | — | os | — | not statically resolved |
| `D:\gci\gemini\append_exp69.py` | — | os | — | not statically resolved |
| `D:\gci\gemini\append_exp70.py` | — | csv | — | not statically resolved |
| `D:\gci\gemini\append_exp71.py` | — | csv | — | not statically resolved |
| `D:\gci\gemini\append_exp72.py` | — | csv | — | not statically resolved |
| `D:\gci\gemini\append_exp73.py` | — | csv | — | not statically resolved |
| `D:\gci\gemini\append_exp74.py` | — | csv | — | not statically resolved |
| `D:\gci\gemini\apply_audit_fixes_nb.py` | — | hashlib, json | — | not statically resolved |
| `D:\gci\gemini\apply_chatgpt_final_audit.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\audit_notebook.py` | — | json, re | — | not statically resolved |
| `D:\gci\gemini\autonomous_loc_auc_optimization_cycle.py` | — | hashlib, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/exp15_xgb_bundle.npz [EXISTS]; cache/hasan_expanded_zoo.npz [EXISTS] |
| `D:\gci\gemini\benchmark_gpu_deep.py` | — | catboost, numpy, pandas, sklearn.metrics, sklearn.model_selection, time, xgboost | — | cache/hasan_feature_importance.csv [EXISTS]; cache/hasan_features_tr.feather [EXISTS]; d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\build_asymmetric_convex_gated_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\build_clean_champion_notebook.py` | — | json, os | — | not statically resolved |
| `D:\gci\gemini\build_contract_risk_contradiction.py` | — | numpy, os, pandas, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\gemini\build_dae_regime_metastack_champion.py` | — | catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\build_dae_regime_synthesis_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time | — | not statically resolved |
| `D:\gci\gemini\build_domain_signal_pack.py` | — | itertools, numpy, os, pandas, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\gemini\build_dual_apex_harmonic_consensus_v83.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time | — | not statically resolved |
| `D:\gci\gemini\build_exp84_chronological_product_apex.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | not statically resolved |
| `D:\gci\gemini\build_exp85_optimal_macro_apex.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | not statically resolved |
| `D:\gci\gemini\build_exp86_quadratic_apex_macro.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | not statically resolved |
| `D:\gci\gemini\build_global_honors_apex_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time | — | not statically resolved |
| `D:\gci\gemini\build_grand_master_macro_stacker_apex.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | not statically resolved |
| `D:\gci\gemini\build_grand_master_ultra_stacker_v82.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sys, time | — | not statically resolved |
| `D:\gci\gemini\build_harmonic_consensus_apex_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; cache/hasan_expanded_zoo.npz [EXISTS] |
| `D:\gci\gemini\build_hasan_600_features.py` | — | lightgbm, numpy, os, pandas, scipy.stats, sklearn.model_selection, sys, warnings | — | not statically resolved |
| `D:\gci\gemini\build_hierarchical_q4_features.py` | — | gc, numpy, os, pandas, sklearn.model_selection, time, warnings | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\build_level2_chain_reaction_apex_stacker.py` | — | hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; d:/gci/gemini/v_solo_080300_level2_regime_stacker.csv [EXISTS] |
| `D:\gci\gemini\build_level2_regime_meta_stacker.py` | — | hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; v_solo_080200_grand_master_apex.csv [EXISTS]; v_solo_080041_pure_champion.csv [EXISTS] |
| `D:\gci\gemini\build_major_gpu_macro_synthesis.py` | — | catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\build_meta_top10_features.py` | — | numpy, os, pandas, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\gemini\build_optimal_transductive_apex_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; cache/hasan_expanded_zoo.npz [EXISTS] |
| `D:\gci\gemini\build_origin_exp1.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\build_origin_exp2.py` | — | hashlib, json, numpy, os, pandas, sys, time | — | not statically resolved |
| `D:\gci\gemini\build_origin_exp3.py` | — | hashlib, json, numpy, os, pandas, sys, time | — | not statically resolved |
| `D:\gci\gemini\build_origin_exp4.py` | — | hashlib, json, numpy, os, pandas, sys, time | — | not statically resolved |
| `D:\gci\gemini\build_precision_transductive_apex_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time | — | cache/hasan_expanded_zoo.npz [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\build_pseudolabel_gpu_superlearner.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | d:/gci/gemini/cache/v_17_full_suite_bundle.npz [EXISTS]; d:/gci/gemini/cache/v55_members.npz [EXISTS]; d:/gci/gemini/cache/test_10f_xgb.npy [EXISTS]; d:/gci/gemini/cache/gpu_tri_engine_550.npz [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\build_pure_champion.py` | — | numpy, os, pandas, scipy.stats | — | not statically resolved |
| `D:\gci\gemini\build_reconstructed_ext_features.py` | — | numpy, os, pandas, sklearn.metrics | — | not statically resolved |
| `D:\gci\gemini\build_regime_adaptive_macro_apex.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time | — | not statically resolved |
| `D:\gci\gemini\build_residual_product_champion.py` | — | hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\build_standalone_package.py` | — | base64, json, numpy, os, pandas, zlib | — | not statically resolved |
| `D:\gci\gemini\build_top_domain_features.py` | — | numpy, os, pandas, sklearn.model_selection, time, warnings | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\build_transductive_equilibrium_apex_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; cache/hasan_expanded_zoo.npz [EXISTS] |
| `D:\gci\gemini\build_transductive_tail_apex_champion.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\build_uid_and_ctg_blocks.py` | — | numpy, os, pandas, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\gemini\build_v16_encodings.py` | — | numpy, os, pandas, pickle, sklearn.model_selection, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\build_v16_features.py` | — | numpy, os, pandas, scipy.stats, sklearn.model_selection, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\build_v16_views.py` | — | numpy, os, pandas, pickle, sklearn.ensemble, sklearn.model_selection, sys, time, warnings | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; D:/gci/codex/v_3_cache/resid16.pkl [EXISTS] |
| `D:\gci\gemini\build_v24_v25_submissions.py` | — | numpy, os, pandas, scipy.stats | — | not statically resolved |
| `D:\gci\gemini\build_v38_views.py` | — | numpy, os, pandas, pickle, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, warnings | — | D:/gci/codex/v_3_cache/resid16.pkl [EXISTS] |
| `D:\gci\gemini\build_v65_solo_notebook.py` | — | json, os | — | not statically resolved |
| `D:\gci\gemini\build_v80200_apex.py` | — | hashlib, numpy, os, pandas, scipy.stats | — | v_solo_080150_micro_apex_fusion.csv [EXISTS]; ../inpuy/train.csv [EXISTS]; ../inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\build_v80250_power_apex.py` | — | hashlib, numpy, os, pandas, scipy.stats | — | cache/hasan_zoo_predictions.npz [EXISTS]; ../inpuy/train.csv [EXISTS]; ../inpuy/test.csv [EXISTS]; v_solo_080200_grand_master_apex.csv [EXISTS]; v_solo_080041_pure_champion.csv [EXISTS]; v_20_apex_grand_champion_submission.csv [EXISTS] |
| `D:\gci\gemini\build_woe_scorecard.py` | — | numpy, os, pandas, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\gemini\cascade_feature_distillation_pipeline.py` | — | lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\apply_patch_v5_02.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\apply_patch_v5_03.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\apply_patch_v5_04.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\apply_patch_v5_05.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\apply_patch_v5_06.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\build_exp85_recovery_nb.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\build_exp85_recovery_v2_nb.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\build_exp85_recovery_v3_nb.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\build_exp85_recovery_v4_nb.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\check_cache_equivalence.py` | — | numpy, pandas, pathlib, scipy.stats | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\check_cols.py` | — | pandas, re | — | D:\gci\inpuy\train.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\check_historical_corrs.py` | — | numpy, pandas, pathlib, scipy.stats | — | D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\dump_nb.py` | — | — | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\execute_exp85_recovery.py` | — | builtins, io, json, os, pathlib, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\execute_exp85_recovery_v2.py` | — | builtins, io, json, os, pathlib, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\execute_exp85_recovery_v3.py` | — | builtins, io, json, os, pathlib, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\execute_exp85_recovery_v4.py` | — | builtins, io, json, os, pathlib, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\execute_exp85updated.py` | — | builtins, io, json, os, pathlib, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\execute_notebook.py` | — | builtins, io, json, numpy, os, pandas, pathlib, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\exp85_recovery.ipynb` | — | catboost, gc, hashlib, lightgbm, math, numpy, os, pandas, pathlib, scipy.optimize, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.optim, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\exp85_recovery_v2.ipynb` | — | catboost, gc, hashlib, lightgbm, math, numpy, os, pandas, pathlib, scipy.optimize, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, tabm, time, torch, torch.nn, torch.optim, torch.utils.data, warnings, xgboost | tabm | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\exp85_recovery_v3.ipynb` | — | catboost, hashlib, lightgbm, math, numpy, os, pandas, pathlib, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, tabm, time, torch, torch.nn, torch.nn.functional, torch.utils.data, warnings, xgboost | tabm | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\exp85_recovery_v4.ipynb` | — | catboost, copy, hashlib, lightgbm, math, numpy, os, pandas, pathlib, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, tabm, time, torch, torch.nn, torch.utils.data, warnings, xgboost | tabm | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\exp85_recovery_v4_backup.ipynb` | — | catboost, copy, hashlib, lightgbm, math, numpy, os, pandas, pathlib, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, tabm, time, torch, torch.nn, torch.utils.data, warnings, xgboost | tabm | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\exp85_recovery_v5.ipynb` | — | catboost, copy, hashlib, lightgbm, math, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, torch, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\exp85_recovery_v5_backup.ipynb` | — | catboost, copy, hashlib, lightgbm, math, numpy, os, pandas, pathlib, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, tabm, time, torch, torch.nn, torch.utils.data, warnings, xgboost | tabm | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\exp85_solo.ipynb` | recreate_precision_anchor, recreate_v17_bundle, recreate_v55_members, recreate_xgb10_and_labels | __future__, argparse, catboost, hashlib, json, lightgbm, numpy, os, pandas, pathlib, scipy.stats, shutil, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\exp85updated.ipynb` | — | catboost, hashlib, lightgbm, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\fix_c10.py` | — | — | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\fix_c11.py` | — | — | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\fix_c14_15_16.py` | — | hashlib, json, numpy, pandas, pathlib, scipy.stats | — | D:\gci\gemini\champion\cacherecreation\submission_exp85_recovery.csv [EXISTS]; D:\gci\gemini\v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\fix_cell5.py` | — | — | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\inspect_cell3.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\inspect_cell3_pt2.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\inspect_cell3_zchamp.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\inspect_cell4.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\inspect_nb.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\list_exp85_cells.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\patch_nb_builder.py` | — | — | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\patch_v5_cells.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\read_cells12_13.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\read_cells2_3.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\read_exp85_cells.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\recreate_precision_anchor.py` | — | argparse, catboost, hashlib, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\recreate_v17_bundle.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\recreate_v55_members.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\recreate_xgb10_and_labels.py` | — | argparse, numpy, pandas, pathlib, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\run_all_recreations.py` | — | argparse, pathlib, subprocess, sys, time | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\run_v5_pipeline.py` | — | hashlib, io, json, numpy, os, pandas, pathlib, scipy.stats, shutil, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\scratch_audit.py` | — | numpy, pandas, scipy.stats | — | D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS]; D:/gci/gemini/champion/cacherecreation/submission_exp85_recovery_v4.csv [EXISTS]; D:/gci/inpuy/train.csv [EXISTS]; D:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_audit_cells.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\scratch_cb.py` | — | catboost, numpy, pandas, pathlib, sklearn.metrics, time | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\scratch_cb_gpu.py` | — | catboost, numpy, pandas, sklearn.metrics, sklearn.model_selection, time | — | D:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_check_signatures.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\scratch_corr.py` | — | numpy, pandas, scipy.stats | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS]; D:/gci/gemini/v_20261027_precision_transductive_apex.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_dump_cells_part1.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\scratch_dump_cells_part2.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\scratch_et.py` | — | lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_exact_historical.py` | — | numpy, pandas, scipy.stats | — | D:/gci/gemini/champion/cacherecreation/submission_exp85_recovery_v4.csv [EXISTS]; D:/gci/inpuy/train.csv [EXISTS]; D:/gci/inpuy/test.csv [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_fine.py` | — | numpy, pandas, scipy.stats | — | D:/gci/gemini/champion/cacherecreation/submission_exp85_recovery_v4.csv [EXISTS]; D:/gci/inpuy/train.csv [EXISTS]; D:/gci/inpuy/test.csv [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_forensic_gap_analysis.py` | — | hashlib, numpy, pandas, pathlib, scipy.stats | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS]; D:/gci/gemini/v_20261027_precision_transductive_apex.csv [EXISTS]; D:/gci/gemini/champion/cacherecreation/v_20261027_precision_transductive_apex.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_hypotheses.py` | — | numpy, pandas, scipy.stats, sklearn.metrics | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS]; D:/gci/gemini/champion/cacherecreation/submission_exp85_recovery_v5_best.csv [EXISTS]; D:/gci/inpuy/test.csv [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_inter_corr.py` | — | numpy, pandas, scipy.stats | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_joint.py` | — | numpy, pandas, scipy.stats | — | D:/gci/gemini/champion/cacherecreation/submission_exp85_recovery_v4.csv [EXISTS]; D:/gci/inpuy/train.csv [EXISTS]; D:/gci/inpuy/test.csv [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_meta_interactions.py` | — | numpy, pandas, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS]; D:/gci/gemini/champion/cacherecreation/submission_exp85_recovery_v5_best.csv [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_meta_log.py` | — | numpy, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_refine_calibration.py` | — | numpy, pandas, scipy.stats | — | D:/gci/gemini/champion/cacherecreation/submission_exp85_recovery_v5_best.csv [EXISTS]; D:/gci/inpuy/train.csv [EXISTS]; D:/gci/inpuy/test.csv [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_refine_calibration_ext.py` | — | numpy, pandas, scipy.stats | — | D:/gci/gemini/champion/cacherecreation/submission_exp85_recovery_v5_best.csv [EXISTS]; D:/gci/inpuy/train.csv [EXISTS]; D:/gci/inpuy/test.csv [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_suite.py` | — | numpy, pandas, scipy.stats | — | D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS]; D:/gci/gemini/v_20261027_precision_transductive_apex.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_tail.py` | — | numpy, pandas, scipy.stats | — | D:/gci/gemini/champion/cacherecreation/submission_exp85_recovery_v5_best.csv [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_test_cb_inner_es.py` | — | catboost, numpy, pandas, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\scratch_test_ensemble_weights.py` | — | numpy, pandas, scipy.optimize, scipy.stats, sklearn.metrics | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_test_et_ensemble.py` | — | lightgbm, numpy, pandas, pathlib, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS]; D:/gci/inpuy/train.csv [EXISTS]; D:/gci/inpuy/test.csv [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_test_full_vs_inner.py` | — | lightgbm, numpy, pandas, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\scratch_test_inner_es.py` | — | lightgbm, numpy, pandas, sklearn.metrics, sklearn.model_selection, time | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\scratch_test_nested_stacker_dryrun.py` | — | catboost, lightgbm, numpy, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, time, xgboost | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\scratch_test_xgb_tuning.py` | — | numpy, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, time, xgboost | — | D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_tier.py` | — | numpy, pandas, scipy.stats | — | D:/gci/gemini/champion/cacherecreation/submission_exp85_recovery_v4.csv [EXISTS]; D:/gci/inpuy/train.csv [EXISTS]; D:/gci/inpuy/test.csv [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\scratch_verify_baseline.py` | — | hashlib, numpy, pandas, scipy.stats | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\scratch_xgb10.py` | — | numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, time, xgboost | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS]; D:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\test_cell30.py` | — | json, numpy, pandas, pathlib, scipy.stats | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\test_cell30_safe.py` | — | numpy, pandas, pathlib, scipy.stats | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS] |
| `D:\gci\gemini\champion\cacherecreation\test_clamp.py` | — | numpy, scipy.stats | — | not statically resolved |
| `D:\gci\gemini\champion\cacherecreation\update_nb.py` | — | json, pathlib | — | not statically resolved |
| `D:\gci\gemini\champion\dependency_audit\anchor74_audit\reproduce_anchor74.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys | — | not statically resolved |
| `D:\gci\gemini\champion\dependency_audit\audit_historical_artifacts.py` | — | hashlib, numpy, os, pandas, pathlib, scipy.stats, sys | — | D:/gci/gemini/champion/dependency_audit/EXP85_DEPENDENCY_INVENTORY.csv [EXISTS] |
| `D:\gci\gemini\champion\dependency_audit\evaluate_reproductions.py` | — | hashlib, numpy, os, pandas, pathlib, scipy.stats, sys | — | not statically resolved |
| `D:\gci\gemini\champion\dependency_audit\inventory_scanner.py` | — | csv, hashlib, os, pathlib, sys, time | — | not statically resolved |
| `D:\gci\gemini\champion\dependency_audit\reproduce_exact_synthesis.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | not statically resolved |
| `D:\gci\gemini\champion\dependency_audit\run_branch_ablation.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys | — | D:/gci/gemini/champion/cacherecreation/base_predictions_v5.npz [EXISTS]; D:/gci/gemini/champion/cacherecreation/submission_exp85_recovery_v5_best.csv [EXISTS] |
| `D:\gci\gemini\champion\exp85_solo.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\champion\exp85_solo_runner.py` | — | catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/oof_10f_xgb.npy [EXISTS]; cache/test_10f_xgb.npy [EXISTS] |
| `D:\gci\gemini\champion\phase3_reconstruction\experiments\create_builder.py` | — | os | — | not statically resolved |
| `D:\gci\gemini\champion\phase3_reconstruction\experiments\reconstruct_hasan_features.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\champion\phase3_reconstruction\experiments\reconstruct_transductive_pyramid.py` | — | hashlib, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\champion\phase3_reconstruction\experiments\run_hasan_600_builder.py` | — | lightgbm, numpy, os, pandas, scipy.stats, sklearn.model_selection, sys, warnings | — | not statically resolved |
| `D:\gci\gemini\champion\phase3_reconstruction\experiments\run_workstream2_screening.py` | — | hashlib, lightgbm, numpy, os, pandas, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\gemini\champion\phase3_reconstruction\experiments\train_independent_zoo.py` | — | catboost, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\champion\run_pipeline.py` | — | os, subprocess, sys, time | — | not statically resolved |
| `D:\gci\gemini\check_id_properties.py` | — | numpy, pandas | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\clean_cell8_nb.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\colab_champion_solution_080.ipynb` | — | base64, catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost, zlib | — | not statically resolved |
| `D:\gci\gemini\colab_champion_solution_080.py` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\colab_champion_solution_080_backup.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\colab_champion_solution_080_with_embedded_anchor.py` | — | base64, catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost, zlib | — | not statically resolved |
| `D:\gci\gemini\colab_champion_solution_v88_audited.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\colab_champion_solution_v88_reproducible.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\compute_residuals.py` | — | numpy, os, pandas, sklearn.ensemble, sklearn.model_selection, time, warnings | — | not statically resolved |
| `D:\gci\gemini\deep_applicant_linkage_analysis.py` | — | numpy, pandas | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\deep_search_leak.py` | — | re | — | d:/gci/gemini/triage_text_extracted.txt [NOT_FOUND_OR_DYNAMIC] |
| `D:\gci\gemini\dependent\build_precision_transductive_apex_independent.py` | — | __future__, argparse, catboost, hashlib, lightgbm, numpy, os, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | not statically resolved |
| `D:\gci\gemini\dependent\exp85_solo_runner_independent.py` | — | argparse, catboost, hashlib, json, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\dependent\train_full_v16_suite_independent.py` | — | argparse, catboost, json, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\diagnose_20pct_failure_categories.py` | — | numpy, os, pandas, scipy.stats, sklearn.metrics, sys | — | d:/gci/inpuy/train.csv [EXISTS]; cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/exp15_xgb_bundle.npz [EXISTS] |
| `D:\gci\gemini\distill_and_fuse_champion.py` | — | hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; d:/gci/gemini/v_solo_080300_level2_regime_stacker.csv [EXISTS] |
| `D:\gci\gemini\eda_and_verify.py` | — | numpy, os, pandas, sys, time | — | not statically resolved |
| `D:\gci\gemini\embed_creation_code_nb.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\eval_cache_and_post_v77.py` | — | numpy, os, pandas, sklearn.metrics | — | not statically resolved |
| `D:\gci\gemini\eval_top_submissions.py` | — | numpy, os, pandas, scipy.stats | — | d:/gci/pre info/oct22/GCI_submission_v17.csv [EXISTS]; d:/gci/pre info/oct22/GCI_submission_v18.csv [EXISTS]; d:/gci/pre info/oct22/GCI_submission_v21_candidate.csv [EXISTS]; d:/gci/gemini/v_20_apex_grand_champion_submission.csv [EXISTS]; d:/gci/gemini/v_16_grand_triumphant_submission.csv [EXISTS] |
| `D:\gci\gemini\execute_exp85_solo_clean.py` | — | contextlib, io, json, os, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\execute_origin_exp2.py` | — | contextlib, io, json, os, shutil, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\execute_origin_exp3.py` | — | contextlib, io, json, os, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\execute_origin_exp4.py` | — | contextlib, io, json, os, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\exp09_build_and_validate_k1.py` | — | numpy, os, pandas, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\exp09_fingerprint_analysis.py` | — | numpy, os, pandas, sklearn.metrics, sys | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\exp09_test_combo_tes.py` | — | numpy, os, pandas, sklearn.metrics, sklearn.model_selection, sys | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\exp10_grand_meta_blend.py` | — | numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/sample_submission.csv [EXISTS] |
| `D:\gci\gemini\exp11_v16_grand_fusion.py` | — | numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/sample_submission.csv [EXISTS]; d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\exp14_partner_expansion_features.py` | — | numpy, os, pandas, sklearn.metrics, sklearn.model_selection, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\exp16_grand_stacking_optimizer.py` | — | numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings | — | d:/gci/inpuy/sample_submission.csv [EXISTS]; d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\EXP85_LEGITIMATE_RECONSTRUCTION.ipynb` | — | — | — | not statically resolved |
| `D:\gci\gemini\exp85_rebuilt\EXP85_LEGITIMATE_RECONSTRUCTION.ipynb` | — | — | — | not statically resolved |
| `D:\gci\gemini\exp85_rebuilt\scripts\feature_pipeline.py` | — | lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.model_selection, time, warnings | — | not statically resolved |
| `D:\gci\gemini\exp85_rebuilt\scripts\macro_synthesizer.py` | train_eight_engine_suite | hashlib, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sys, time | — | not statically resolved |
| `D:\gci\gemini\exp85_rebuilt\scripts\train_eight_engine_suite.py` | feature_pipeline | catboost, json, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\exp85_rebuilt\tests\test_exp85_pipeline.py` | — | numpy, os, pandas, scipy.stats, sys, unittest | — | not statically resolved |
| `D:\gci\gemini\experiment_chain_reaction_deep_suite.py` | — | lightgbm, numpy, os, pandas, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\explore_transductive_tail_apex.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | cache/hasan_expanded_zoo.npz [EXISTS]; d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\extract_resolve_paths.py` | — | os, re | — | not statically resolved |
| `D:\gci\gemini\extract_solution.py` | — | re | — | d:/gci/gemini/triage_text_extracted.txt [NOT_FOUND_OR_DYNAMIC] |
| `D:\gci\gemini\extract_v15.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\extracted_community_solution.py` | — | — | — | not statically resolved |
| `D:\gci\gemini\extracted_v15.py` | — | catboost, glob, google.colab, lightgbm, numpy, os, pandas, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, warnings | google | not statically resolved |
| `D:\gci\gemini\extracted_v26\run_gci_v26_optimize_five.py` | — | json, lightgbm, numpy, os, pandas, sklearn.cluster, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, time, warnings | — | not statically resolved |
| `D:\gci\gemini\extracted_v29\GCI_v29_package\run_gci_v29_fresh_confirmation.py` | — | json, lightgbm, numpy, os, pandas, sklearn.cluster, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, time, warnings | — | not statically resolved |
| `D:\gci\gemini\feature_builder.py` | — | gc, numpy, os, pandas, sklearn.ensemble, sklearn.model_selection, time, warnings | — | not statically resolved |
| `D:\gci\gemini\feature_builder_v2.py` | — | gc, numpy, os, pandas, sklearn.model_selection, time, warnings | — | not statically resolved |
| `D:\gci\gemini\final_submission\reproduce_final_submission.py` | — | hashlib, json, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys | — | not statically resolved |
| `D:\gci\gemini\find_dataset_ordering.py` | — | numpy, pandas | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb` | — | base64, catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost, zlib | — | not statically resolved |
| `D:\gci\gemini\GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\gci_champion_solution_clean.ipynb` | — | catboost, gc, google.colab, hashlib, json, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.decomposition, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, torch, warnings, xgboost | google | not statically resolved |
| `D:\gci\gemini\GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb` | — | base64, catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost, zlib | — | not statically resolved |
| `D:\gci\gemini\grid_search_calibration.py` | — | numpy, pandas, scipy.stats, sklearn.metrics | — | d:/gci/inpuy/train.csv [EXISTS]; cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/exp15_xgb_bundle.npz [EXISTS] |
| `D:\gci\gemini\idependentexperiments\origin.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\independentexperiments\origin.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\independentexperiments\origin_exp1.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\independentexperiments\origin_exp2.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\independentexperiments\origin_exp3.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\independentexperiments\origin_exp4.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\inspect_5_artifacts_deep.py` | — | numpy, pandas, scipy.stats, sklearn.metrics | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/gemini/cache/v_17_full_suite_bundle.npz [EXISTS]; d:/gci/gemini/cache/v55_members.npz [EXISTS]; d:/gci/gemini/cache/oof_10f_xgb.npy [EXISTS]; d:/gci/gemini/cache/test_10f_xgb.npy [EXISTS]; d:/gci/gemini/v_20261027_precision_transductive_apex.csv [EXISTS]; d:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\inspect_cache_candidates.py` | — | glob, numpy, os, pandas, scipy.stats, sklearn.metrics | — | d:/gci/gemini/cache/v_17_full_suite_bundle.npz [EXISTS]; d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\inspect_gold.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\inspect_notebook.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\inspect_preinfo_nbs.py` | — | json, os | — | not statically resolved |
| `D:\gci\gemini\inspect_test_pair_discrepancies.py` | — | numpy, pandas | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; v_20261027_optimal_transductive_apex.csv [EXISTS] |
| `D:\gci\gemini\patch_cell6_helpers.py` | — | contextlib, io, json, os, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\populate_clean_outputs.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\populate_final_clean_outputs.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\populate_final_outputs.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\prepare_execution.py` | — | contextlib, io, json, os, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\read_v16_notebook.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\rebuild_exp85_artifacts.py` | — | catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\reproduce_honors_champion.py` | — | argparse, lightgbm, numpy, os, pandas, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\research_exp48_0783_breakthrough.py` | — | catboost, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\research_feature_formulas.py` | — | gc, lightgbm, numpy, os, pandas, sklearn.metrics, sklearn.model_selection, time, warnings | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; d:/gci/gemini/cache/v16_Xa.feather [EXISTS]; d:/gci/gemini/cache/new_4block_train.feather [EXISTS]; D:/gci/codex/v_3_cache/resid16.pkl [EXISTS] |
| `D:\gci\gemini\run_90min_honors_breakthrough.py` | — | gc, json, lightgbm, numpy, os, pandas, scipy.special, scipy.stats, sklearn.decomposition, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\run_origin_exp1.py` | — | contextlib, io, json, os, shutil, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\scratch\append_exp90.py` | — | — | — | not statically resolved |
| `D:\gci\gemini\scratch\apply_champion_to_scripts.py` | — | base64, numpy, pandas, zlib | — | d:/gci/gemini/scratch/b64_ranks.txt [EXISTS] |
| `D:\gci\gemini\scratch\audit_and_sync_notebook.py` | — | ast, hashlib, json, os, re | — | not statically resolved |
| `D:\gci\gemini\scratch\audit_champion_comparison.py` | — | hashlib, numpy, os, pandas, scipy.stats | — | not statically resolved |
| `D:\gci\gemini\scratch\audit_clean_ensemble.py` | — | numpy, scipy.stats, sklearn.metrics | — | d:/gci/gemini/cache/y_train.npy [EXISTS]; d:/gci/gemini/cache/v_17_full_suite_bundle.npz [EXISTS]; d:/gci/gemini/cache/v55_members.npz [EXISTS]; d:/gci/gemini/cache/oof_10f_xgb.npy [EXISTS] |
| `D:\gci\gemini\scratch\audit_exp85_files.py` | — | hashlib, numpy, os, pandas | — | not statically resolved |
| `D:\gci\gemini\scratch\build_100pct_corr_nb.py` | — | ast, json, os | — | not statically resolved |
| `D:\gci\gemini\scratch\build_18cell_master_notebook.py` | — | ast, json, os | — | not statically resolved |
| `D:\gci\gemini\scratch\build_clean_exp85_solo_notebook.py` | — | json, os | — | not statically resolved |
| `D:\gci\gemini\scratch\build_clean_standalone_notebook.py` | — | ast, json, os | — | not statically resolved |
| `D:\gci\gemini\scratch\build_clean_suite_submission.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics | — | d:/gci/gemini/cache/y_train.npy [EXISTS]; d:/gci/gemini/cache/v_17_full_suite_bundle.npz [EXISTS]; d:/gci/gemini/cache/v55_members.npz [EXISTS]; d:/gci/gemini/cache/oof_10f_xgb.npy [EXISTS]; d:/gci/gemini/cache/test_10f_xgb.npy [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\scratch\build_exp85_legitimate_notebook.py` | — | json, os | — | not statically resolved |
| `D:\gci\gemini\scratch\build_gci_champion_self_contained_nb.py` | — | ast, hashlib, json | — | not statically resolved |
| `D:\gci\gemini\scratch\build_master_colab_champion_nb.py` | — | ast, json, os | — | not statically resolved |
| `D:\gci\gemini\scratch\check_b64.py` | — | base64, numpy, zlib | — | not statically resolved |
| `D:\gci\gemini\scratch\check_colab_ipynb_ast.py` | — | ast, json | — | not statically resolved |
| `D:\gci\gemini\scratch\check_column_access.py` | — | json, pandas, re | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\scratch\check_device.py` | — | catboost, lightgbm, torch, xgboost | — | not statically resolved |
| `D:\gci\gemini\scratch\check_diff.py` | — | hashlib, numpy, pandas, scipy.stats | — | d:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS]; d:/gci/gemini/scratch/check_match.csv [EXISTS] |
| `D:\gci\gemini\scratch\check_log_code.py` | — | — | — | competition_log_extracted_full.txt [EXISTS] |
| `D:\gci\gemini\scratch\check_model_corrs.py` | — | hashlib, numpy, os, pandas, scipy.stats | — | not statically resolved |
| `D:\gci\gemini\scratch\check_nb_mapping.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\check_v85_code.py` | — | numpy, pandas | — | d:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\scratch\check_world_nb.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\clean_cell18.py` | — | hashlib, json | — | not statically resolved |
| `D:\gci\gemini\scratch\compare_nb_to_script.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\create_audited_notebook.py` | — | ast, json | — | not statically resolved |
| `D:\gci\gemini\scratch\create_champion_notebook.py` | — | base64, json, numpy, pandas, zlib | — | d:/gci/gemini/scratch/b64_ranks.txt [EXISTS] |
| `D:\gci\gemini\scratch\deep_search_pillar.py` | — | os, zipfile | — | not statically resolved |
| `D:\gci\gemini\scratch\embed_notebook_outputs.py` | — | contextlib, io, json, time | — | not statically resolved |
| `D:\gci\gemini\scratch\final_sub\final_submission\reproduce_final_submission.py` | — | hashlib, json, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys | — | not statically resolved |
| `D:\gci\gemini\scratch\finalize_exp85_rebuilt.py` | macro_synthesizer | hashlib, json, numpy, os, pandas, scipy.stats, sys | — | not statically resolved |
| `D:\gci\gemini\scratch\generate_audit_and_manifest.py` | — | hashlib, json, numpy, os, pandas | — | not statically resolved |
| `D:\gci\gemini\scratch\generate_exp85_solo_notebook.py` | — | base64, json, numpy, os, pandas, zlib | — | not statically resolved |
| `D:\gci\gemini\scratch\generate_manifest.py` | — | hashlib, json, os | — | not statically resolved |
| `D:\gci\gemini\scratch\generate_notebooks.py` | — | base64, json, os, shutil, zlib | — | not statically resolved |
| `D:\gci\gemini\scratch\greedy_ensemble_search.py` | — | numpy, os, pandas, scipy.stats, sklearn.metrics | — | d:/gci/gemini/cache/y_train.npy [EXISTS]; d:/gci/gemini/cache/v_17_full_suite_bundle.npz [EXISTS]; d:/gci/gemini/cache/v55_members.npz [EXISTS]; d:/gci/gemini/cache/gpu_tri_engine_550.npz [EXISTS]; d:/gci/gemini/cache/v35_quad_bundle.npz [EXISTS]; d:/gci/gemini/cache/oof_10f_xgb.npy [EXISTS]; d:/gci/gemini/cache/oof_deep_xgb_d6.npy [EXISTS]; d:/gci/gemini/cache/oof_mv_xgb.npy [EXISTS]; d:/gci/gemini/cache/cb_upgraded_native_oof.npy [EXISTS]; d:/gci/gemini/cache/knn_500_tr.npy [EXISTS]; d:/gci/gemini/cache/woe_scorecard_tr.npy [EXISTS] |
| `D:\gci\gemini\scratch\inspect_all_cells.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\inspect_c4.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\inspect_cell18.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\inspect_colab_080.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\inspect_exp60.py` | — | csv | — | experminetbygeminioct22.csv [EXISTS] |
| `D:\gci\gemini\scratch\inspect_nb.py` | — | json, os | — | not statically resolved |
| `D:\gci\gemini\scratch\inspect_nb_detail.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\inspect_notebooks.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\inspect_py_cells.py` | — | re | — | not statically resolved |
| `D:\gci\gemini\scratch\inspect_recent_files.py` | — | datetime, os | — | not statically resolved |
| `D:\gci\gemini\scratch\inspect_self_contained_nb.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\inspect_v17_nb.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\inspect_v21_nb.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\inspect_world_nb.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\make_colab_ipynb.py` | — | ast, json, os | — | not statically resolved |
| `D:\gci\gemini\scratch\populate_notebook_outputs.py` | — | hashlib, io, json, numpy, os, pandas, scipy.stats, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\scratch\print_c2_c5.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\remove_b64.py` | — | — | — | not statically resolved |
| `D:\gci\gemini\scratch\run_and_audit_ipynb.py` | — | hashlib, json, numpy, os, pandas, scipy.stats, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\scratch\run_controlled_experiments.py` | — | catboost, hashlib, json, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\scratch\run_v85_dry.py` | — | hashlib | — | d:/gci/gemini/scratch/v85_dry_run.csv [EXISTS]; d:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\scratch\scan_anchor_sections.py` | — | — | — | not statically resolved |
| `D:\gci\gemini\scratch\search_brain.py` | — | os | — | not statically resolved |
| `D:\gci\gemini\scratch\search_files.py` | — | os | — | not statically resolved |
| `D:\gci\gemini\scratch\static_smoke_test_and_audit.py` | — | ast, hashlib, importlib, json, os, re, sys | — | not statically resolved |
| `D:\gci\gemini\scratch\sync_py_from_nb.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\test_b64_unpack.py` | — | base64, hashlib, numpy, pandas, scipy.stats, zlib | — | d:/gci/gemini/scratch/test_rec_match.csv [EXISTS]; d:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\scratch\test_complete_8engine_pipeline.py` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\scratch\test_decomp_hash.py` | — | base64, hashlib, json, numpy, pandas, scipy.stats, zlib | — | d:/gci/gemini/scratch/test_sub.csv [EXISTS] |
| `D:\gci\gemini\scratch\test_exact_v85.py` | — | base64, numpy, pandas, scipy.stats, zlib | — | d:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv [EXISTS] |
| `D:\gci\gemini\scratch\test_exp85_direct.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | not statically resolved |
| `D:\gci\gemini\scratch\test_exp85_solo_nb.py` | — | json, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\scratch\test_model_vs_champ.py` | — | hashlib, numpy, pandas, scipy.stats | — | not statically resolved |
| `D:\gci\gemini\scratch\test_real_reproducible_pipeline.py` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\scratch\test_run_notebook.py` | — | json, sys, time | — | not statically resolved |
| `D:\gci\gemini\scratch\test_run_option_b_nb.py` | — | json, time, traceback | — | not statically resolved |
| `D:\gci\gemini\scratch\test_stepwise_exp85.py` | — | hashlib, numpy, os, pandas, scipy.stats, sklearn.metrics, sys, time | — | d:/gci/gemini/scratch/test_exp85_stepwise.csv [EXISTS] |
| `D:\gci\gemini\scratch\trace_code_creators.py` | — | os, re | — | not statically resolved |
| `D:\gci\gemini\scratch\trace_oct22_progression.py` | — | csv | — | experminetbygeminioct22.csv [EXISTS] |
| `D:\gci\gemini\scratch\trace_variable_dependencies.py` | — | ast, builtins, json | — | not statically resolved |
| `D:\gci\gemini\scratch\update_manifest.py` | — | hashlib, json, os | — | not statically resolved |
| `D:\gci\gemini\scratch\verify_anchor_match.py` | — | base64, hashlib, numpy, pandas, zlib | — | d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\scratch\verify_ipynb.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch\verify_submission.py` | — | hashlib, numpy, pandas | — | submission_self_contained.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\scratch\view_cell_contents.py` | — | re | — | not statically resolved |
| `D:\gci\gemini\scratch\view_fe_cells.py` | — | re | — | not statically resolved |
| `D:\gci\gemini\scratch\view_log_pillar.py` | — | — | — | competition_log_extracted.txt [EXISTS] |
| `D:\gci\gemini\scratch_inspect_cells.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch_inspect_cells_15_17.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch_inspect_features_v17_v18.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch_inspect_v17_v18.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch_inspect_v19.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch_inspect_v22.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\scratch_meta_weight_optimizer.py` | — | numpy, pandas, scipy.optimize, scipy.stats, sklearn.metrics | — | d:/gci/gemini/cache/v23_meta_models_bundle.npz [EXISTS]; d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\screen_and_select_features.py` | — | lightgbm, numpy, os, pandas, sklearn.metrics, sklearn.model_selection | — | cache/hasan_features_tr.feather [EXISTS]; cache/hasan_features_te.feather [EXISTS]; cache/y_train.npy [EXISTS] |
| `D:\gci\gemini\screen_chain_reaction_features.py` | — | lightgbm, numpy, os, pandas, sklearn.metrics, sklearn.model_selection, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\search_hasan_secrets.py` | — | re | — | d:/gci/gemini/triage_text_extracted.txt [NOT_FOUND_OR_DYNAMIC] |
| `D:\gci\gemini\search_triage.py` | — | re | — | d:/gci/gemini/triage_text_extracted.txt [NOT_FOUND_OR_DYNAMIC] |
| `D:\gci\gemini\soften_leakage_wording.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\solo_grand_master_apex_080.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys | — | ../inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\standalone_champion_pipeline.py` | — | base64, catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost, zlib | — | not statically resolved |
| `D:\gci\gemini\synthesize_and_submit.py` | — | numpy, os, pandas, scipy.stats, sklearn.metrics, sys | — | cache/y_train.npy [EXISTS]; ../inpuy/test.csv [EXISTS]; cache/hasan_zoo_predictions.npz [EXISTS]; v_20_apex_grand_champion_submission.csv [EXISTS]; scratch/final_sub/final_submission/submission_final.csv [EXISTS]; ../inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\test_calibrated_harmonic_gamma.py` | — | numpy, pandas, scipy.stats, sklearn.metrics | — | d:/gci/inpuy/train.csv [EXISTS]; cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/exp15_xgb_bundle.npz [EXISTS] |
| `D:\gci\gemini\test_community_breakthrough_features.py` | — | lightgbm, numpy, os, pandas, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\test_conflict_correction.py` | — | numpy, pandas, scipy.stats, sklearn.metrics | — | d:/gci/inpuy/train.csv [EXISTS]; cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/exp15_xgb_bundle.npz [EXISTS] |
| `D:\gci\gemini\test_exp2_features.py` | — | numpy, os, pandas, sklearn.model_selection, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\test_exp4_specialists.py` | — | lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\test_expanded_partner_leakage.py` | — | numpy, pandas, sklearn.metrics, sklearn.model_selection | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\test_intra_person_consensus.py` | — | numpy, pandas, scipy.stats, sklearn.metrics | — | d:/gci/inpuy/train.csv [EXISTS]; cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/exp15_xgb_bundle.npz [EXISTS] |
| `D:\gci\gemini\test_joint_synthesis_oof.py` | — | numpy, pandas, scipy.stats, sklearn.metrics | — | d:/gci/inpuy/train.csv [EXISTS]; cache/hasan_expanded_zoo.npz [EXISTS]; cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/exp15_xgb_bundle.npz [EXISTS] |
| `D:\gci\gemini\test_meta_models.py` | — | catboost, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, time, xgboost | — | d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\test_partner_enc.py` | — | numpy, pandas, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\test_partner_features.py` | — | numpy, os, pandas, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\test_regime2_repair_weight.py` | — | numpy, pandas, scipy.stats, sklearn.metrics | — | d:/gci/inpuy/train.csv [EXISTS]; cache/hasan_expanded_zoo.npz [EXISTS]; cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/exp15_xgb_bundle.npz [EXISTS] |
| `D:\gci\gemini\test_regime2_solvency.py` | — | numpy, pandas, sklearn.metrics | — | d:/gci/inpuy/train.csv [EXISTS]; cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/exp15_xgb_bundle.npz [EXISTS] |
| `D:\gci\gemini\test_run_clean_notebook.py` | — | hashlib, json, numpy, os, pandas, scipy.stats, sys, time, traceback | — | not statically resolved |
| `D:\gci\gemini\test_true_champion_pipeline.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\test_views_and_resid.py` | — | catboost, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\track_pipeline_and_score_gap_analysis.py` | — | hashlib, numpy, os, pandas, scipy.stats, sys | — | not statically resolved |
| `D:\gci\gemini\train_and_evaluate_combinations.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, time, warnings | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; D:/gci/codex/v_3_cache/resid16.pkl [EXISTS] |
| `D:\gci\gemini\train_catboost_and_deep_lgb.py` | — | catboost, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\train_clean_honors.py` | — | argparse, catboost, gc, json, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.decomposition, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\train_clean_honors_champion.py` | — | catboost, lightgbm, numpy, os, pandas, pickle, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | not statically resolved |
| `D:\gci\gemini\train_exp15_gpu_xgboost.py` | — | gc, numpy, os, pandas, pickle, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | D:/gci/codex/v_3_cache/resid16.pkl [EXISTS]; D:/gci/codex/v_3_cache/e1A_views.pkl [EXISTS] |
| `D:\gci\gemini\train_full_v16_suite.py` | — | gc, lightgbm, numpy, os, pandas, pickle, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings | — | D:/gci/codex/v_3_cache/resid16.pkl [EXISTS] |
| `D:\gci\gemini\train_gpu_product_specialists.py` | — | catboost, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, xgboost | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; d:/gci/gemini/v_20261027_grand_master_macro_stacker_apex.csv [EXISTS] |
| `D:\gci\gemini\train_hasan_master_zoo.py` | — | catboost, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, warnings, xgboost | — | cache/hasan_selected_550_tr.feather [EXISTS]; cache/hasan_selected_550_te.feather [EXISTS]; cache/y_train.npy [EXISTS]; cache/oof_lgb_cons.npy [EXISTS]; cache/test_lgb_cons.npy [EXISTS]; cache/oof_lgb_et.npy [EXISTS]; cache/test_lgb_et.npy [EXISTS] |
| `D:\gci\gemini\train_meta_top10_models.py` | — | catboost, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, time, xgboost | — | not statically resolved |
| `D:\gci\gemini\train_models.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | d:/gci/inpuy/sample_submission.csv [EXISTS] |
| `D:\gci\gemini\train_models_v2.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | d:/gci/inpuy/sample_submission.csv [EXISTS] |
| `D:\gci\gemini\train_models_v3_gpu.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | d:/gci/inpuy/sample_submission.csv [EXISTS] |
| `D:\gci\gemini\train_models_v4_pseudolabel.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | d:/gci/inpuy/sample_submission.csv [EXISTS] |
| `D:\gci\gemini\train_models_v5_stacking.py` | — | gc, lightgbm, numpy, os, pandas, scipy.special, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/sample_submission.csv [EXISTS] |
| `D:\gci\gemini\train_models_v6_multiseed.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; d:/gci/inpuy/sample_submission.csv [EXISTS] |
| `D:\gci\gemini\train_models_v7_signal_purified.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; d:/gci/inpuy/sample_submission.csv [EXISTS] |
| `D:\gci\gemini\train_models_v8_knn.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; d:/gci/inpuy/sample_submission.csv [EXISTS] |
| `D:\gci\gemini\train_models_v9_combos.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; d:/gci/inpuy/sample_submission.csv [EXISTS] |
| `D:\gci\gemini\train_product_synergy_specialist.py` | — | hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\gemini\train_regime2_specialist.py` | — | lightgbm, numpy, pandas, sklearn.metrics, sklearn.model_selection | — | d:/gci/inpuy/train.csv [EXISTS] |
| `D:\gci\gemini\train_self_contained_zoo_and_transductive_apex.py` | — | __future__, argparse, catboost, hashlib, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, xgboost | — | not statically resolved |
| `D:\gci\gemini\train_v16_and_optimized.py` | — | catboost, gc, lightgbm, numpy, os, pandas, pickle, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | D:/gci/codex/v_3_cache/resid16.pkl [EXISTS]; D:/gci/codex/v_3_cache/e1A_views.pkl [EXISTS] |
| `D:\gci\gemini\train_v29_full_synergy.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, time, warnings, xgboost | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; D:/gci/codex/v_3_cache/resid16.pkl [EXISTS] |
| `D:\gci\gemini\train_v30_formula_surrogate.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, time, warnings, xgboost | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; D:/gci/codex/v_3_cache/resid16.pkl [EXISTS] |
| `D:\gci\gemini\train_v31_hierarchical_product_champion.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, time, warnings, xgboost | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS]; D:/gci/codex/v_3_cache/resid16.pkl [EXISTS] |
| `D:\gci\gemini\train_v35_apex_master.py` | — | catboost, gc, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\train_v58_clean_gpu_0783_champion.py` | — | numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, time | — | not statically resolved |
| `D:\gci\gemini\train_v59_clean_gpu_micro_apex.py` | — | numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, time | — | not statically resolved |
| `D:\gci\gemini\train_v61_clean_contract_gap_apex.py` | — | numpy, os, pandas, scipy.stats, time | — | not statically resolved |
| `D:\gci\gemini\train_v62_clean_contract_gap_apex_from_scratch.py` | — | argparse, numpy, os, pandas, scipy.stats, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v63_breakthrough_0783.py` | — | lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\gemini\train_v64_micro_gated_margin_calibration.py` | — | numpy, os, pandas, scipy.stats, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v65_double_winning_gradient_apex.py` | — | numpy, os, pandas, scipy.stats, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v68_clean_multiview_apex.py` | — | argparse, catboost, gc, json, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v69_invariant_linkage_apex.py` | — | json, numpy, os, pandas, scipy.stats, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v70_multiview_decomposition_apex.py` | — | gc, json, lightgbm, numpy, os, pandas, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v71_income_tier_ordinal_apex.py` | — | lightgbm, numpy, os, pandas, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v72_tri_anchor_convex_apex.py` | — | numpy, os, pandas, scipy.stats, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v73_extreme_precision_invariant_apex.py` | — | numpy, os, pandas, scipy.stats, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v74_omnibus_calibration_apex.py` | — | numpy, os, pandas, scipy.stats, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v75_expanded_identity_apex.py` | — | itertools, numpy, os, pandas, scipy.stats, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v76_multiview_expanded_identity_apex.py` | — | numpy, os, pandas, scipy.stats, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v77_quad_anchor_consensus_apex.py` | — | numpy, os, pandas, scipy.stats, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v78_standardized_cohort_multiview_apex.py` | — | gc, json, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.ensemble, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v79_discrepancy_repair_multiview_apex.py` | — | numpy, os, pandas, scipy.stats, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v80_grand_synthesis_multiview_apex.py` | — | numpy, os, pandas, scipy.stats, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v81_pure_ascent_apex.py` | — | numpy, os, pandas, scipy.stats, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\train_v_gold_grand_master_apex.py` | — | json, numpy, os, pandas, scipy.stats, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\update_cell8_cell9_auditor.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\update_cell9_assert.py` | — | json | — | not statically resolved |
| `D:\gci\gemini\update_exp69.py` | — | — | — | experminetbygeminioct22.csv [EXISTS] |
| `D:\gci\gemini\v17_cells_dump.py` | — | — | — | not statically resolved |
| `D:\gci\gemini\v62_champion_standalone.ipynb` | — | base64, glob, lightgbm, numpy, os, pandas, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, warnings, zlib | — | not statically resolved |
| `D:\gci\gemini\v62_champion_standalone.py` | — | base64, glob, lightgbm, numpy, os, pandas, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, warnings, zlib | — | not statically resolved |
| `D:\gci\gemini\v65_multiview_apex_champion_submission.ipynb` | — | catboost, gc, json, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.ensemble, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\v65_multiview_apex_champion_submission.py` | — | catboost, gc, json, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.ensemble, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, warnings | — | not statically resolved |
| `D:\gci\gemini\verify_grand_master.py` | — | numpy, os, pandas, scipy.stats, sklearn.metrics | — | not statically resolved |
| `D:\gci\name5_dependent_code\build_precision_transductive_apex_champion.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, xgboost | — | not statically resolved |
| `D:\gci\name5_dependent_code\train_catboost_and_deep_lgb.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\name5_dependent_code\train_exp15_gpu_xgboost.py` | — | argparse, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection, xgboost | — | not statically resolved |
| `D:\gci\name5_dependent_code\train_full_v16_suite.py` | — | argparse, lightgbm, numpy, pandas, pathlib, scipy.stats, sklearn.metrics, sklearn.model_selection | — | not statically resolved |
| `D:\gci\name5_dependent_code\train_v58_clean_gpu_0783_champion.py` | — | argparse, catboost, lightgbm, numpy, pandas, pathlib, scipy.optimize, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, xgboost | — | not statically resolved |
| `D:\gci\past_85\colab_champion_solution_080.ipynb` | — | base64, catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost, zlib | — | not statically resolved |
| `D:\gci\past_85\colab_champion_solution_080_with_embedded_anchor.py` | — | base64, catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost, zlib | — | not statically resolved |
| `D:\gci\past_85\exp85_solo with 90 percent correlation .ipynb` | — | __future__, argparse, catboost, hashlib, json, lightgbm, numpy, os, pandas, pathlib, scipy.stats, shutil, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\past_85\exp85_solo(2).ipynb` | — | catboost, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/test_10f_xgb.npy [EXISTS] |
| `D:\gci\past_85\exp85_solo(3).ipynb` | — | catboost, hashlib, lightgbm, numpy, os, pandas, pickle, scipy.stats, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, subprocess, sys, time, warnings, xgboost | — | cache/v_17_full_suite_bundle.npz [EXISTS]; cache/v55_members.npz [EXISTS]; cache/oof_10f_xgb.npy [EXISTS]; cache/test_10f_xgb.npy [EXISTS] |
| `D:\gci\past_85\exp85_solo.ipynb` | — | __future__, argparse, catboost, hashlib, json, lightgbm, numpy, os, pandas, pathlib, scipy.stats, shutil, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\past_85\GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb` | — | catboost, gc, hashlib, lightgbm, numpy, os, pandas, random, scipy.optimize, scipy.stats, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.utils.data, warnings, xgboost | — | not statically resolved |
| `D:\gci\pre info\GCI_FINAL_v15_end_to_end_colab.ipynb` | — | catboost, glob, google.colab, lightgbm, numpy, os, pandas, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, warnings | google | not statically resolved |
| `D:\gci\pre info\GCI_v17_catboost_E1_stack_colab.ipynb` | — | catboost, glob, google.colab, lightgbm, numpy, os, pandas, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, warnings | google | not statically resolved |
| `D:\gci\pre info\GCI_v18_alt_direction_probe.ipynb` | — | lightgbm, numpy, os, pandas, scipy.stats, time, warnings | — | not statically resolved |
| `D:\gci\pre info\GCI_v33_product_conditional_experiment.py` | — | hashlib, json, lightgbm, numpy, os, pandas, scipy.stats, sklearn.metrics, sklearn.model_selection, time, warnings | — | not statically resolved |
| `D:\gci\pre info\oct22\extracted_files\fast_screen.py` | — | argparse, itertools, json, os, sklearn.ensemble, sklearn.linear_model, sys, time, warnings, xgboost | — | not statically resolved |
| `D:\gci\pre info\oct22\extracted_files\gci_core.py` | — | glob, hashlib, lightgbm, numpy, os, pandas, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sys, time, warnings | — | not statically resolved |
| `D:\gci\pre info\oct22\GCI_v16_everything_merged_o.ipynb` | — | catboost, glob, google.colab, lightgbm, numpy, os, pandas, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, subprocess, sys, time, warnings | google | not statically resolved |
| `D:\gci\pre info\oct22\GCI_v16_everything_merged_o_78222 score.ipynb` | — | catboost, glob, google.colab, lightgbm, numpy, os, pandas, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, subprocess, sys, time, warnings | google | not statically resolved |
| `D:\gci\pre info\oct22\GCI_v17_optimised_O 78237 score.ipynb` | — | catboost, glob, google.colab, hashlib, lightgbm, numpy, os, pandas, scipy.optimize, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, subprocess, sys, time, warnings | google | not statically resolved |
| `D:\gci\pre info\oct22\GCI_v18_evidence_driven_o 78235 score.ipynb` | — | catboost, glob, google.colab, hashlib, lightgbm, numpy, optuna, os, pandas, scipy.optimize, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, subprocess, sys, time, warnings | google, optuna | not statically resolved |
| `D:\gci\pre info\oct22\GCI_v19_NN_entity_embeddings_colab_0.ipynb` | — | glob, google.colab, lightgbm, numpy, os, pandas, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing, sys, time, torch, torch.nn, torch.nn.functional, warnings | google | not statically resolved |
| `D:\gci\pre info\oct22\GCI_v21_gated_research_candidate_o 78197 score.ipynb` | — | catboost, glob, google.colab, hashlib, lightgbm, numpy, optuna, os, pandas, scipy.optimize, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, subprocess, sys, time, warnings | google, optuna | not statically resolved |
| `D:\gci\pre info\oct22\GCI_v22_research_expansion_gated crashed.ipynb` | — | catboost, glob, google.colab, hashlib, lightgbm, numpy, optuna, os, pandas, scipy.optimize, scipy.stats, sklearn.ensemble, sklearn.metrics, sklearn.model_selection, subprocess, sys, time, warnings | google, optuna | not statically resolved |
| `D:\gci\pre info\oct22\GCI_v25_block_tester.ipynb` | — | catboost, gc, glob, google.colab, hashlib, itertools, json, lightgbm, numpy, os, pandas, psutil, scipy.sparse, scipy.stats, sklearn.ensemble, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, subprocess, sys, time, warnings | google, psutil | not statically resolved |
| `D:\gci\pre info\oct22\gci_v25_raw_dump.py` | — | catboost, gc, glob, google.colab, hashlib, itertools, json, lightgbm, numpy, os, pandas, psutil, scipy.sparse, scipy.stats, sklearn.ensemble, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, subprocess, sys, time, warnings | google, psutil | not statically resolved |
| `D:\gci\pre info\oct22\run_gci_v25_block_tester.py` | — | gc, itertools, json, lightgbm, numpy, os, pandas, sklearn.metrics, sklearn.model_selection, time, warnings | — | d:/gci/inpuy/train.csv [EXISTS]; d:/gci/inpuy/test.csv [EXISTS] |
| `D:\gci\pre info\oct22\solution_v5_gci_research.py` | — | catboost, functools, glob, hashlib, json, lightgbm, numpy, optuna, os, pandas, re, scipy, scipy.optimize, scipy.signal, scipy.sparse, scipy.sparse.csgraph, scipy.special, scipy.stats, shutil, sklearn.ensemble, sklearn.impute, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.neighbors, sklearn.neural_network, sklearn.preprocessing, sys, time, torch, torch.nn, traceback, warnings, xgboost | optuna | not statically resolved |
