# Exploratory test scripts (frozen)

These scripts were used during sub-projects 1 and 1.5 to diagnose and
characterise the pipeline.  They reference legacy modules (`backend.inpaint`,
`backend.colour_match`, etc.) that no longer exist on `main` after the FLUX
Kontext engine swap.  Most will not run successfully today.

They are preserved for archaeology: looking back at what was tested, how
quality changed, and what experiments led to current design decisions.

For the live acceptance gate, use `tests/run_acceptance_test.py`.

For ongoing unit tests, use `tests/test_*.py` (pytest collects these).
