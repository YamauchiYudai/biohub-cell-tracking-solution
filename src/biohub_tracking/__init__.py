"""Biohub cell tracking solution: the components this team added on top of the public tracking stack.

Subpackages
  io              track-graph container and the competition's submission CSV schema
  metrics         thin wrapper around the organisers' official scorer (optional ``metric`` extra)
  division        learned division recovery: candidate forks, T3 3D-CNN scorer, gated add / remove
  postprocessing  jump-aware relink (b1c), jump-aware line fit (J2), frozen-frame coordinate consensus
  detection       V1284 coordinate-refinement head fine-tuning (final-day candidate component)
  kaggle          reading, checking and preparing the production Kaggle notebook
"""

__version__ = "1.0.0"

VOXEL_SCALE_UM: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
"""Voxel size (z, y, x) in micrometres of every competition movie."""

MATCH_RADIUS_UM = 7.0
"""Node-matching radius of the official metric."""
