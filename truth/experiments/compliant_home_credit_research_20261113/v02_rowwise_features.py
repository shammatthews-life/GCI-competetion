from pipeline_core_v1 import main


if __name__ == "__main__":
    main(
        candidate_id="v02_rowwise_features",
        feature_set="rowwise_v1",
        models=("catboost_gpu", "lightgbm_cpu", "xgboost_gpu"),
    )
