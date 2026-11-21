from pipeline_core_v2 import main


if __name__ == "__main__":
    main(
        candidate_id="v04_pairwise_ranker",
        feature_set="rowwise_v1",
        models=("xgboost_ranker_gpu",),
        target_encoding=False,
    )
