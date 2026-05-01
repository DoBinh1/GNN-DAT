"""
graphsage.py — Mô hình GraphSAGE cho dự đoán ứng suất.

Kiến trúc khớp với paper Cai & Jelovica 2024:

    [Node feat 8-dim]
            │
            ▼
    SAGEConv(8 → 64) → BatchNorm → tanh
            │
            ▼
    SAGEConv(64 → 64) → BatchNorm → tanh    × (n_layers - 1) lần
            │
            ▼
    Linear(64 → 200)
            │
            ▼
    [Node stress 200-dim — reshape về (10, 20)]

Công thức GraphSAGE với aggregator 'sum':

    h_v^l = σ( W^l ( h_v^{l-1} + Σ_{u ∈ N(v)} h_u^{l-1} ) )

trong đó:
    σ           — hàm kích hoạt phi tuyến (ta dùng tanh)
    W^l         — ma trận trọng số học được tại layer l
    h_v^{l-1}   — vector trạng thái của node v tại layer trước
    N(v)        — tập các node láng giềng của v (theo edge_index)

Lý do chọn 'sum' (không phải 'mean'):
    - 'sum' phân biệt được số lượng láng giềng → phù hợp cơ học (số gân
      ảnh hưởng độ cứng).
    - Xu et al. (2019) chứng minh 'sum' là aggregator có sức biểu diễn
      mạnh nhất trong các hàm bất biến hoán vị.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class StiffenedPanelGNN(nn.Module):
    """
    Mạng GraphSAGE nhiều lớp cho dự đoán trường ứng suất trên từng node.

    Output shape: (num_nodes, out_dim).
        Sau khi predict, ta reshape mỗi vector out_dim thành lưới
        (rows, cols) — chính là trường ứng suất 2D trên đơn vị tương ứng.
    """

    def __init__(
        self,
        in_dim: int = 8,
        hidden: int = 64,
        n_layers: int = 8,
        out_dim: int = 200,
        aggr: str = "sum",
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers phải >= 1")

        self.in_dim   = in_dim
        self.hidden   = hidden
        self.n_layers = n_layers
        self.out_dim  = out_dim
        self.aggr     = aggr

        # Stack của các lớp SAGEConv + BatchNorm
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Layer đầu: in_dim → hidden
        self.convs.append(SAGEConv(in_dim, hidden, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden))

        # Các layer tiếp theo: hidden → hidden
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden))

        # Output head: linear projection từ hidden → out_dim (200)
        self.head = nn.Linear(hidden, out_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier init cho head — ổn định hơn với output là regression range rộng."""
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) — node features
            edge_index: (2, E) — danh sách cạnh

        Returns:
            (N, out_dim) — vector dự đoán cho mỗi node
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)   # message passing
            x = bn(x)                  # ổn định gradient & chống over-smoothing
            x = torch.tanh(x)          # phi tuyến
        return self.head(x)            # projection ra stress dim
