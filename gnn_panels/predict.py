"""
predict.py — Inference với một checkpoint đã train.

Chạy:
    python predict.py

Sẽ load best.pt, sinh 1 panel ngẫu nhiên, predict, và vẽ so sánh
predicted vs ground-truth (nếu matplotlib có sẵn).
"""

from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import torch
import numpy as np

from . import config as cfg
from .data import generate_panel_design, build_graph_from_design
from .models import StiffenedPanelGNN
from .utils import FeatureNormalizer


def load_model_from_checkpoint(ckpt_path, device):
    """Load model + normalizer từ checkpoint đã save."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = StiffenedPanelGNN(**ckpt["config_model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    norm = FeatureNormalizer()
    norm.__dict__.update(ckpt["normalizer"])
    # GIỮ normalizer ở CPU — vì transform được gọi trên graph chưa move sang device.
    # Sau khi transform xong, ta mới move graph sang device cho forward.
    for k in ("x_min", "x_max", "y_min", "y_max"):
        v = getattr(norm, k)
        if isinstance(v, torch.Tensor):
            setattr(norm, k, v.cpu())
    return model, norm


def predict_panel(model, normalizer, design, device):
    """
    Dự đoán stress field cho 1 panel.

    Returns:
        pred_grids: numpy (num_nodes, rows, cols) — stress 2D từng node
        true_grids: numpy (num_nodes, rows, cols) — ground truth (giả lập)
    """
    graph = build_graph_from_design(design, generate_labels=True)
    true_y = graph.y.clone()  # giữ ground truth gốc
    normalizer.transform(graph)
    graph = graph.to(device)

    with torch.no_grad():
        pred_norm = model(graph.x, graph.edge_index)

    # Đảo chuẩn hóa để có MPa thật
    pred_real = normalizer.inverse_y(pred_norm).cpu()

    rows, cols = cfg.STRESS_GRID
    pred_grids = pred_real.view(-1, rows, cols).numpy()
    true_grids = true_y.view(-1, rows, cols).numpy()
    return pred_grids, true_grids


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg.CHECKPOINT_DIR / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {ckpt_path}. Hãy chạy train.py trước."
        )

    print(f"[INFO] Loading checkpoint từ {ckpt_path}")
    model, normalizer = load_model_from_checkpoint(ckpt_path, device)

    # Sinh 1 panel test (seed nằm ngoài tập train)
    test_seed = 99999
    design = generate_panel_design(seed=test_seed)
    print(f"[INFO] Test panel: {design['n_stiffeners']} stiffeners, "
          f"plate {design['plate_thickness']:.1f} mm dày, "
          f"pressure {design['pressure']:.3f} MPa")

    pred_grids, true_grids = predict_panel(model, normalizer, design, device)

    # In kết quả từng node
    print(f"\n{'Node idx':<10}{'Type':<10}{'Pred max (MPa)':>18}{'True max (MPa)':>18}{'Acc':>10}")
    print("-" * 66)

    n_plate = design["n_stiffeners"] + 1
    n_stiff = design["n_stiffeners"]
    for i in range(pred_grids.shape[0]):
        if i < n_plate:
            t = "plate"
        elif i < n_plate + n_stiff:
            t = "web"
        else:
            t = "flange"
        p_max = pred_grids[i].max()
        gt_max = true_grids[i].max()
        acc = 1.0 - abs(p_max - gt_max) / max(abs(gt_max), 1e-6)
        print(f"{i:<10}{t:<10}{p_max:>18.2f}{gt_max:>18.2f}{acc*100:>9.1f}%")

    # Tổng hợp
    overall_pred_max = pred_grids.max()
    overall_true_max = true_grids.max()
    print(f"\n[OVERALL] Pred panel max = {overall_pred_max:.2f} MPa")
    print(f"[OVERALL] True panel max = {overall_true_max:.2f} MPa")
    overall_acc = 1.0 - abs(overall_pred_max - overall_true_max) / max(abs(overall_true_max), 1e-6)
    print(f"[OVERALL] Max stress accuracy = {overall_acc*100:.2f}%")

    # Optional: vẽ
    try:
        import matplotlib.pyplot as plt
        n_show = min(pred_grids.shape[0], 6)
        fig, axes = plt.subplots(2, n_show, figsize=(3*n_show, 6))
        for i in range(n_show):
            axes[0, i].imshow(true_grids[i], cmap="hot", aspect="auto")
            axes[0, i].set_title(f"GT node {i}")
            axes[0, i].axis("off")
            axes[1, i].imshow(pred_grids[i], cmap="hot", aspect="auto")
            axes[1, i].set_title(f"Pred node {i}")
            axes[1, i].axis("off")
        plt.tight_layout()
        out_path = cfg.LOG_DIR / "prediction_demo.png"
        plt.savefig(out_path, dpi=100)
        print(f"\n[INFO] Đã lưu hình so sánh tại: {out_path}")
    except ImportError:
        print("\n[INFO] matplotlib chưa cài → bỏ qua phần visualize.")


if __name__ == "__main__":
    main()
