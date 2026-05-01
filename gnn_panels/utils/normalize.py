"""
normalize.py — Chuẩn hóa node features và labels.

Vì sao cần chuẩn hóa?
    Các feature có thang đo CỰC khác nhau:
        - chiều dài (mm): 100 - 3000
        - độ dày (mm):    5 - 20
        - áp suất (MPa):  0.05 - 0.3

    Nếu KHÔNG normalize, gradient của feature có biên độ lớn (length) sẽ
    DOMINATE gradient của feature có biên độ nhỏ (pressure). Mạng "phớt
    lờ" pressure → predict sai.

Chiến lược: Min-Max scaling về [0, 1]. Ưu điểm: giữ được ý nghĩa "0 = không
có / nhỏ nhất" và "1 = lớn nhất", phù hợp với hàm kích hoạt tanh.

Lưu ý quan trọng:
    Phải fit normalizer CHỈ trên tập TRAIN, sau đó áp dụng cho val/test.
    Fit trên cả tập sẽ gây "data leakage" — model thấy được phân phối
    test trong khi học, làm test accuracy ảo cao.
"""

from __future__ import annotations
import torch
from typing import List
from torch_geometric.data import Data


class FeatureNormalizer:
    """Chuẩn hóa Min-Max cho node features và labels."""

    def __init__(self):
        self.x_min: torch.Tensor | None = None
        self.x_max: torch.Tensor | None = None
        self.y_min: torch.Tensor | None = None
        self.y_max: torch.Tensor | None = None

    def fit(self, dataset: List[Data]):
        """Tính min/max trên toàn dataset."""
        all_x = torch.cat([d.x for d in dataset], dim=0)        # (sum_N, F)
        all_y = torch.cat([d.y for d in dataset], dim=0)        # (sum_N, out_dim)
        self.x_min = all_x.min(dim=0).values
        self.x_max = all_x.max(dim=0).values
        self.y_min = all_y.min()
        self.y_max = all_y.max()
        return self

    def transform(self, data: Data) -> Data:
        """
        Áp dụng chuẩn hóa lên 1 Data object.

        x → (x - x_min) / (x_max - x_min) → range [0, 1]
        y → (y - y_min) / (y_max - y_min) → range [0, 1]

        Vì transform sửa data tại chỗ → trả về data luôn (PyG convention).
        """
        eps = 1e-8
        data.x = (data.x - self.x_min) / (self.x_max - self.x_min + eps)
        if hasattr(data, "y") and data.y is not None:
            data.y = (data.y - self.y_min) / (self.y_max - self.y_min + eps)
        return data

    def inverse_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Đảo ngược chuẩn hóa cho label/prediction.
        Cần khi muốn xem giá trị stress ở đơn vị MPa thực.
        """
        return y_norm * (self.y_max - self.y_min) + self.y_min


def fit_normalizer(train_dataset) -> FeatureNormalizer:
    """
    Fit normalizer trên tập train và trả về.
    """
    norm = FeatureNormalizer()
    norm.fit([train_dataset.get(i) for i in range(train_dataset.len())])
    return norm
