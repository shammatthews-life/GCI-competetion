# Residual analysis status

Residual-driven feature discovery was not performed in this batch. The current OOF predictions are available in `OOF_PREDICTIONS.csv`, but inspecting outer-fold errors and then developing features against those same rows would create a second layer of selection bias.

Any next residual analysis must generate hypotheses on development-only predictions or perform nested feature discovery, then evaluate the predeclared hypotheses on untouched folds. No claims about under-ranked applicant groups are made here.
