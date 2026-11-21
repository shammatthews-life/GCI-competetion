# Executed runner snapshot

`run_experiment_seed42_source.py` is a byte-for-byte copy of the experiment runner
used for the completed `phase2_inner_stop_seed42` run. It is preserved because
the current runner's report now includes independently named fold-AUC
summaries; that reporting-only correction was made after training.

The model-training code, data inputs, fold assignments, and selected
iterations are unchanged by that reporting correction. The run manifest's
`source_sha256.runner` is the hash of this executed snapshot.
