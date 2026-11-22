# Grid color / stroke contrast

Experimental independent extraction method for reddish grids. It is not selected by the production entrypoint. Visual validation is incomplete: V4 gaps and grid artifacts remain on the evaluation family.

Run locally with explicit layout/calibration:

    python -m research_methods.run_grid_color IMAGE --config CONFIG.json --out NEW_OUTPUT_DIRECTORY

Requires the root requirements plus scipy. Uses only RGB appearance and explicit configuration, never source IDs. The activation gate is provisional and has not been validated for generalization. Outputs measured masks and an overlay; no clinical-accuracy claim.

Focused checks:

    python -m unittest tests.test_chromatic_residual -v
