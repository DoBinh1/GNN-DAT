"""Module data — sinh dữ liệu, xây dựng graph, đóng gói thành Dataset."""

from .synthetic_panel import generate_panel_design, generate_pseudo_stress_field
from .graph_builder import build_graph_from_design
from .dataset import StiffenedPanelDataset

__all__ = [
    "generate_panel_design",
    "generate_pseudo_stress_field",
    "build_graph_from_design",
    "StiffenedPanelDataset",
]
