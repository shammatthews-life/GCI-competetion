# Execution and reproducibility audit

## Safe checks performed

- Parsed and compiled **637** Python/notebook code sources; 633 passed and 4 failed.
- Notebook cells were compiled independently after removing IPython magic lines to avoid treating normal cell boundaries as syntax errors.
- Extracted functions/classes, imports, candidate local imports, data/artifact string references, seeds, CV, GPU, ensemble and leakage review signals.
- Checked third-party import availability with `find_spec`; no source module was imported or executed.
- Discovered **49** existing test-like Python files; none were run because imports may trigger legacy side effects and many require historical data/artifacts.
- No dependencies were installed, no model was trained, no GPU was used, and no preserved source or input data was modified.

## Syntax failures

| File | Failure |
|---|---|
| `D:\gci\gemini\EXP85_LEGITIMATE_RECONSTRUCTION.ipynb` | SyntaxError: unterminated string literal (detected at line 67) (EXP85_LEGITIMATE_RECONSTRUCTION.ipynb#cell7, line 67) |
| `D:\gci\gemini\exp85_rebuilt\EXP85_LEGITIMATE_RECONSTRUCTION.ipynb` | SyntaxError: unterminated string literal (detected at line 67) (EXP85_LEGITIMATE_RECONSTRUCTION.ipynb#cell7, line 67) |
| `D:\gci\gemini\extracted_community_solution.py` | SyntaxError: unterminated triple-quoted string literal (detected at line 583) (extracted_community_solution.py, line 579) |
| `D:\gci\gemini\v17_cells_dump.py` | SyntaxError: unterminated string literal (detected at line 6) (v17_cells_dump.py, line 6) |

Notebook parse failures may prevent those notebooks from executing as written. The 2 `.py` failures are archival/incomplete extracts, not evidence that the underlying project source is invalid.

## Environment import check

| Detected module/status | Number of source files |
|---|---:|
| `__future__: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 92 |
| `argparse: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 79 |
| `ast: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 13 |
| `base64: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 19 |
| `builtins: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 7 |
| `catboost: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 142 |
| `contextlib: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 9 |
| `copy: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 10 |
| `csv: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 14 |
| `datetime: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 25 |
| `functools: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 1 |
| `gc: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 55 |
| `glob: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 17 |
| `google: NOT_FOUND_IN_AUDIT_ENVIRONMENT` | 13 |
| `hashlib: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 209 |
| `importlib: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 10 |
| `io: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 17 |
| `itertools: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 6 |
| `json: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 241 |
| `lightgbm: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 193 |
| `math: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 12 |
| `numpy: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 465 |
| `optuna: NOT_FOUND_IN_AUDIT_ENVIRONMENT` | 4 |
| `os: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 340 |
| `pandas: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 465 |
| `pathlib: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 201 |
| `pickle: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 11 |
| `platform: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 2 |
| `psutil: NOT_FOUND_IN_AUDIT_ENVIRONMENT` | 2 |
| `random: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 14 |
| `re: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 23 |
| `scipy: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 363 |
| `shutil: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 9 |
| `sklearn: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 330 |
| `subprocess: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 27 |
| `sys: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 295 |
| `tabm: NOT_FOUND_IN_AUDIT_ENVIRONMENT` | 5 |
| `tempfile: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 4 |
| `threading: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 2 |
| `time: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 338 |
| `torch: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 27 |
| `traceback: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 24 |
| `typing: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 23 |
| `unittest: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 15 |
| `uuid: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 1 |
| `warnings: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 156 |
| `xgboost: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 111 |
| `zipfile: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 1 |
| `zlib: AVAILABLE_IN_AUDIT_ENVIRONMENT` | 19 |

Module availability describes only the current audit Python environment. It does not substitute for each project's pinned requirements or prove version compatibility.

## Blockers and artifact dependencies

- Unresolved literal input references are listed in `BLOCKED_COMPONENTS.csv`. A path is checked relative to its source directory and the top level of each preserved source root; `NOT_FOUND_OR_DYNAMIC` is not proof of global absence, so verify the runtime working directory and artifact store.
- Historical synthesis/recovery scripts depend on saved model prediction arrays and other generated artifacts; they do not train their entire lineage from raw data.
- Direct inspection of the five scripts in `D:\gci\5 independent` shows reads of cached Feather/NPZ/PKL inputs and/or an anchor CSV (including v16/v17/v55 bundles); these are artifact-dependent consumers, not standalone raw-data reproductions. The separate `D:\gci\name5_dependent_code` scripts are distinct raw-data replacement approximations.
- `gemini/final_submission/reproduce_final_submission.py` specifically references multiple neural/synthesis NPZ/NPY streams documented as absent in the preserved dependency audit. It is blocked as a complete historical synthesis route.
- `gemini/dependent/exp85_notebook_run/ARTIFACT_REBUILD_AUDIT.md` says the exact 274-feature monotonic 10-fold EXP42 training creator is not present; its missing filename is not stated there.
- `name5_dependent_code/README.md` says the exact V43 submission writer was not found. These are documented gaps, not names of the two user-referenced missing files.

## Missing-file exclusion

The two exact filenames previously identified were not discoverable from the permitted source documentation. I therefore did not guess or silently exclude files by name. All 637 available `.py`/`.ipynb` files across the seven stated preserved source roots were inventoried and scanned. The documented gaps above are separately recorded, and any organizer/user-supplied exclusion names should be added to the report without removing the current all-available-file inventory.
