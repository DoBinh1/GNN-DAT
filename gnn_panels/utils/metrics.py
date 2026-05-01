"""
metrics.py — Các metric đánh giá mô hình.

Paper báo cáo "max stress accuracy" — chỉ số cực kỳ quan trọng trong cơ
học vì failure xảy ra tại vùng ứng suất cực đại.
"""

from __future__ import annotations
import torch


def max_stress_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Tính độ chính xác dự đoán ứng suất cực đại.

    accuracy = 1 - |pred_max - target_max| / target_max

    Args:
        pred:   (N, out_dim) hoặc (out_dim,) — dự đoán stress
        target: same shape   — ground truth

    Returns:
        float trong [0, 1] (cao là tốt). Có thể âm nếu sai cực mạnh.
    """
    pred_max   = pred.max()
    target_max = target.max()
    if target_max.abs() < 1e-8:
        return 1.0
    rel_err = (pred_max - target_max).abs() / target_max
    return float(1.0 - rel_err.item())


def mean_relative_error(pred: torch.Tensor, target: torch.Tensor,
                        eps: float = 1e-3) -> float:
    """
    Sai số tương đối trung bình toàn trường ứng suất.

    Bỏ qua các điểm có |target| < eps để tránh chia 0.
    """
    mask = target.abs() > eps
    if mask.sum() == 0:
        return 0.0
    rel = (pred[mask] - target[mask]).abs() / target[mask].abs()
    return float(rel.mean().item())


def per_panel_max_accuracy(pred: torch.Tensor, target: torch.Tensor,
                           batch: torch.Tensor) -> float:
    """
    Tính max stress accuracy TRUNG BÌNH theo panel (không theo node).

    Trong PyG, batch.batch là tensor (N,) chỉ ra mỗi node thuộc panel nào.
    Ta cần group theo batch.batch rồi lấy max trong từng group.

    Args:
        pred:   (N, out_dim)
        target: (N, out_dim)
        batch:  (N,) — batch index của mỗi node

    Returns:
        Trung bình accuracy trên các panel.
    """
    accs = []
    for graph_idx in batch.unique():
        mask = batch == graph_idx
        accs.append(max_stress_accuracy(pred[mask], target[mask]))
    return sum(accs) / len(accs) if accs else 0.0
