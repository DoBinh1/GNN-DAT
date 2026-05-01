"""
dataset.py — PyTorch Geometric Dataset wrapper.

Đóng gói pipeline: seed → design → graph thành một Dataset duy nhất, để
có thể dùng với DataLoader của PyG.

Lý do tách Dataset class riêng (thay vì dùng list of Data):
    - Có thể tích hợp lazy loading (đọc từ file thay vì tạo trong memory).
    - Tương thích chuẩn với PyG ecosystem.
    - Dễ mở rộng sau này (thêm transform, augmentation, ...).
"""

from __future__ import annotations
from torch_geometric.data import Dataset
from typing import List, Optional

from .synthetic_panel import generate_panel_design
from .graph_builder import build_graph_from_design


class StiffenedPanelDataset(Dataset):
    """
    Dataset các stiffened panel sinh giả lập theo seed.

    Mỗi item trong dataset là một PyG Data object đại diện cho 1 panel.
    """

    def __init__(self, seeds: List[int], transform=None, pre_transform=None):
        """
        Args:
            seeds: danh sách seed dùng để sinh panel. Số lượng seed = số panel.
            transform: optional, hàm transform áp dụng mỗi lần get().
            pre_transform: optional, hàm transform áp dụng 1 lần lúc init.
        """
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self._seeds = list(seeds)
        # Cache các graph đã build (vì sinh giả lập rẻ, ta build trước hết)
        self._graphs = []
        for s in self._seeds:
            design = generate_panel_design(seed=s)
            graph  = build_graph_from_design(design)
            if pre_transform is not None:
                graph = pre_transform(graph)
            self._graphs.append(graph)

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int):
        return self._graphs[idx]


def make_train_val_test(
    n_train: int,
    n_val: int,
    n_test: int,
    seed_offset: int = 0,
    pre_transform=None,
):
    """
    Tiện ích tạo 3 tập dataset (train/val/test) với seed không trùng nhau.

    Quan trọng: chia theo SEED → mỗi panel chỉ thuộc đúng 1 tập, không có
    rò rỉ dữ liệu.
    """
    s = seed_offset
    train_ds = StiffenedPanelDataset(range(s, s + n_train), pre_transform=pre_transform)
    s += n_train
    val_ds   = StiffenedPanelDataset(range(s, s + n_val), pre_transform=pre_transform)
    s += n_val
    test_ds  = StiffenedPanelDataset(range(s, s + n_test), pre_transform=pre_transform)
    return train_ds, val_ds, test_ds
