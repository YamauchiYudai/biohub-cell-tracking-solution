"""Detection-side component: fine-tuning of the upstream V1284 coordinate-refinement head (recipe F03).

The detector itself (dual-seed TemporalUNet3D with 8-view TTA) and the DeepCenter centre prior are public
pretrained models used unchanged; see README "Architecture".
"""
