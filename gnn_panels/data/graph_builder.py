"""
graph_builder.py — Chuyển một thiết kế stiffened panel thành đối tượng Graph.

Đây là TIM của paper Cai & Jelovica 2024:
    Mỗi ĐƠN VỊ CẤU TRÚC (plate-span / web / flange) → một NODE trong graph.
    Mỗi MỐI HÀN giữa 2 đơn vị → một EDGE.

So với cách "mỗi finite-element = 1 node" của các paper trước, cách này:
    - Số node giảm vài trăm lần (16-30 thay vì 10.000+).
    - Tôn trọng cấu trúc rời rạc tự nhiên của panel.
    - Tốc độ huấn luyện nhanh hơn ~27 lần, GPU memory giảm ~98%.

Quy ước index trong graph:
    [0 .. n_plate-1]                          → các plate-span
    [n_plate .. n_plate + n_stiff - 1]        → các web gân
    [n_plate + n_stiff .. n_plate + 2*n_stiff - 1] → các flange gân

với n_plate = n_stiffeners + 1.
"""

from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict

from .synthetic_panel import generate_pseudo_stress_field


def build_graph_from_design(design: Dict, generate_labels: bool = True) -> Data:
    """
    Xây dựng PyTorch Geometric Data object từ một thiết kế panel.

    Args:
        design: dict trả về từ generate_panel_design()
        generate_labels: nếu True, tự sinh nhãn stress (cho training).
                        Nếu False (vd inference), không sinh y.

    Returns:
        torch_geometric.data.Data với:
            x          : (N, 8) — node feature, 8 chiều mỗi node
            edge_index : (2, E) — danh sách cạnh (đã làm đối xứng cho undirected)
            y          : (N, 200) — nhãn stress, mỗi node 200 giá trị (10x20)
                         Chỉ có nếu generate_labels=True.
            num_nodes  : N
            design     : dict gốc (lưu lại để debug/visualize)
    """
    n_stiff = design["n_stiffeners"]
    n_plate = n_stiff + 1

    # ---- 1) Tạo node features ------------------------------------------------
    nodes = []   # list các vector 8-chiều
    labels = []  # list các vector 200-chiều
    rng = np.random.default_rng(design["seed"])

    # Plate-spans: width = total_width / (n_stiff + 1)
    plate_span_width = design["plate_width"] / n_plate
    for i in range(n_plate):
        feat = _make_plate_feature(design, plate_span_width)
        nodes.append(feat)
        if generate_labels:
            labels.append(generate_pseudo_stress_field("plate", design, i, rng))

    # Stiffener webs
    for i in range(n_stiff):
        feat = _make_web_feature(design)
        nodes.append(feat)
        if generate_labels:
            labels.append(generate_pseudo_stress_field("web", design, i, rng))

    # Stiffener flanges
    for i in range(n_stiff):
        feat = _make_flange_feature(design)
        nodes.append(feat)
        if generate_labels:
            labels.append(generate_pseudo_stress_field("flange", design, i, rng))

    # ---- 2) Tạo edges --------------------------------------------------------
    edges = _build_edges(n_plate, n_stiff)

    # ---- 3) Đóng gói thành PyG Data -----------------------------------------
    x = torch.tensor(np.array(nodes), dtype=torch.float)

    # edge_index shape (2, E*2): mỗi edge xuất hiện 2 chiều cho undirected graph
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    data = Data(x=x, edge_index=edge_index, num_nodes=x.size(0))
    if generate_labels:
        data.y = torch.tensor(np.array(labels), dtype=torch.float)
    data.design = design
    return data


# ---------------------------------------------------------------------------
# Helper functions để tạo feature vector cho mỗi loại unit
# ---------------------------------------------------------------------------
def _make_plate_feature(design: Dict, plate_span_width: float) -> list:
    """
    Tạo feature vector 8-chiều cho một plate-span.

    Thứ tự feature: [width, length, thickness, bc1, bc2, bc3, bc4, pressure]
    """
    return [
        plate_span_width,           # bề rộng của plate-span (mm)
        design["plate_length"],     # chiều dài (mm)
        design["plate_thickness"],  # độ dày (mm)
        float(design["bc_plate"][0]),
        float(design["bc_plate"][1]),
        float(design["bc_plate"][2]),
        float(design["bc_plate"][3]),
        design["pressure"],         # áp suất (MPa)
    ]


def _make_web_feature(design: Dict) -> list:
    """Feature 8-chiều cho web gân."""
    return [
        design["stiff_thickness"],      # bề rộng "hiệu dụng" (= độ dày web)
        design["stiff_height"],         # chiều cao web
        design["plate_length"],         # chiều dài web (= chiều dài panel)
        float(design["bc_stiff"][0]),
        float(design["bc_stiff"][1]),
        float(design["bc_stiff"][2]),
        float(design["bc_stiff"][3]),
        design["pressure"],
    ]


def _make_flange_feature(design: Dict) -> list:
    """Feature 8-chiều cho flange gân."""
    return [
        design["flange_width"],
        design["flange_thickness"],
        design["plate_length"],
        float(design["bc_stiff"][0]),
        float(design["bc_stiff"][1]),
        float(design["bc_stiff"][2]),
        float(design["bc_stiff"][3]),
        design["pressure"],
    ]


def _build_edges(n_plate: int, n_stiff: int) -> list:
    """
    Xây dựng danh sách cạnh dựa trên topology vật lý của panel.

    Quy ước:
        - Plate-span thứ i nằm giữa stiffener (i-1) và stiffener i.
          Vì vậy plate i nối với web (i-1) (nếu tồn tại) và web i (nếu tồn tại).
        - Mỗi web nối với flange của chính nó.

    Index:
        plate_idx  = i,                      i ∈ [0, n_plate)
        web_idx    = n_plate + j,            j ∈ [0, n_stiff)
        flange_idx = n_plate + n_stiff + j,  j ∈ [0, n_stiff)
    """
    edges = []

    # Plate ↔ Web
    for i in range(n_plate):
        web_left  = i - 1               # web nằm bên trái plate i
        web_right = i                   # web nằm bên phải plate i
        if 0 <= web_left < n_stiff:
            edges.append((i, n_plate + web_left))
        if 0 <= web_right < n_stiff:
            edges.append((i, n_plate + web_right))

    # Web ↔ Flange (mỗi web có 1 flange của riêng nó)
    for j in range(n_stiff):
        web_idx    = n_plate + j
        flange_idx = n_plate + n_stiff + j
        edges.append((web_idx, flange_idx))

    return edges
