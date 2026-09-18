# config.py

# ============================================================
# RUN LABEL
# ============================================================
# Identifies this pipeline run and becomes the top-level folder name
# under outputs/. Every dataset listed in DATASETS below is processed
# as part of this one run. Rename this whenever you want a fresh run
# (different dataset selection, different limits/sizes, etc.) without
# overwriting a previous run's outputs.

RUN_NAME = "chestmnist_stl10"
#RUN_NAME = "chestmnist"


# ============================================================
# DATASETS TO RUN
# ============================================================
# One entry per dataset this run will process. Add, remove, or edit
# entries here to control which datasets run and with what settings.
# To run a single dataset, just leave one entry in this dict.
#
#   limit : number of images to use from this dataset
#   sizes : resolutions (px) to evaluate, listed from base/clean
#           resolution down to the most degraded. sizes[0] is treated
#           as that dataset's base resolution.

DATASETS = {

    "chestmnist": {
        "limit": 6188,
        "sizes": [224, 192, 160, 128, 112, 96, 80, 64, 56, 48, 40, 32, 30, 28, 26, 24, 16, 12, 8],
    },

    "stl10": {
        "limit": 8000,
        "sizes": [96, 80, 64, 56, 48, 40, 32, 30, 28, 26, 24, 16, 12, 8],
    },

}


# ============================================================
# DERIVED SETTINGS — do not edit directly
# ============================================================
# Base resolution per dataset = first (highest) entry in its sizes list.

for _cfg in DATASETS.values():
    _cfg["base_resolution"] = _cfg["sizes"][0]