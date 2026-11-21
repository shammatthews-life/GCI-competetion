from pipeline_core_v3 import main


if __name__ == "__main__":
    main(
        candidate_id="v05_fold_local_group_features",
        feature_set="rowwise_v1",
        models=("xgboost_gpu",),
        group_features=True,
    )
