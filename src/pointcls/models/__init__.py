"""Point cloud classification models."""

from pointcls.models.dgcnn import DGCNN
from pointcls.models.pointmlp import PointMLP
from pointcls.models.pointnext import PointNeXt

__all__ = ["DGCNN", "PointMLP", "PointNeXt"]
