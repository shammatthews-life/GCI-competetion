from pipeline_core_v2 import main


if __name__ == "__main__":
    main(
        candidate_id="v03_nested_target_encoding",
        feature_set="rowwise_v1",
        models=("catboost_gpu", "lightgbm_cpu", "xgboost_gpu"),
        target_encoding=True,
    )
