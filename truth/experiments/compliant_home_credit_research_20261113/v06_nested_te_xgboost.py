from pipeline_core_v2 import main


if __name__ == "__main__":
    main(
        candidate_id="v06_nested_te_xgboost",
        feature_set="rowwise_v1",
        models=("xgboost_gpu",),
        target_encoding=True,
    )
