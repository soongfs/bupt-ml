"""Point cloud classification models."""

from pointcls.models.dgcnn import DGCNN
from pointcls.models.pointmlp import PointMLP

__all__ = ["DGCNN", "PointMLP"]
