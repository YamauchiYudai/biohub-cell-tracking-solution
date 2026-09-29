"""Every module imports with the core dependencies only (torch / tracking_cellmot are imported lazily)."""

import importlib

import pytest

MODULES = [
    "biohub_tracking",
    "biohub_tracking.io", "biohub_tracking.io.graph", "biohub_tracking.io.submission", "biohub_tracking.io.geff",
    "biohub_tracking.metrics.official",
    "biohub_tracking.division", "biohub_tracking.division.candidates", "biohub_tracking.division.select",
    "biohub_tracking.division.processor", "biohub_tracking.division.features", "biohub_tracking.division.scoring",
    "biohub_tracking.division.t3", "biohub_tracking.division.crops", "biohub_tracking.division.labels",
    "biohub_tracking.division.thresholds",
    "biohub_tracking.postprocessing.relink", "biohub_tracking.postprocessing.linefit",
    "biohub_tracking.postprocessing.consensus",
    "biohub_tracking.detection.coordinate_head",
    "biohub_tracking.kaggle.notebook", "biohub_tracking.kaggle.variants", "biohub_tracking.kaggle.blocks",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_constants():
    import biohub_tracking

    assert biohub_tracking.VOXEL_SCALE_UM == (1.625, 0.40625, 0.40625)
    assert biohub_tracking.MATCH_RADIUS_UM == 7.0
