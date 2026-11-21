from pipeline_core_v1 import main


if __name__ == "__main__":
    main(
        candidate_id="v01_clean_raw",
        feature_set="raw",
        models=("catboost_gpu", "lightgbm_cpu", "xgboost_gpu"),
    )
