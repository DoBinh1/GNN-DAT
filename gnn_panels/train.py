"""
train.py — Training pipeline hoàn chỉnh.

Chạy độc lập:
    python train.py

Sẽ:
    1. Sinh dataset (train/val/test) từ seed.
    2. Fit normalizer trên train, áp lên cả 3.
    3. Khởi tạo model GraphSAGE.
    4. Vòng lặp huấn luyện với early stopping.
    5. Lưu checkpoint tốt nhất vào outputs/checkpoints/.
"""

from __future__ import annotations
import sys
# Đảm bảo console Windows in được tiếng Việt
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from . import config as cfg
from .data.dataset import make_train_val_test
from .models import StiffenedPanelGNN
from .utils import fit_normalizer, max_stress_accuracy
from .utils.metrics import per_panel_max_accuracy


# ---------------------------------------------------------------------------
# Helper: chọn device
# ---------------------------------------------------------------------------
def get_device():
    pref = cfg.TRAIN["device"]
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


# ---------------------------------------------------------------------------
# Một epoch huấn luyện
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_n    = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        # Gradient clipping — bảo vệ chống explosion trong mạng sâu
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_n    += batch.num_graphs
    return total_loss / total_n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Chạy validation/test, trả về (loss, max_stress_accuracy)."""
    model.eval()
    total_loss = 0.0
    total_n    = 0
    accs       = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index)
        loss = loss_fn(pred, batch.y)
        total_loss += loss.item() * batch.num_graphs
        total_n    += batch.num_graphs
        accs.append(per_panel_max_accuracy(pred, batch.y, batch.batch))
    avg_loss = total_loss / total_n
    avg_acc  = sum(accs) / len(accs) if accs else 0.0
    return avg_loss, avg_acc


# ---------------------------------------------------------------------------
# Pipeline chính
# ---------------------------------------------------------------------------
def main():
    device = get_device()
    print(f"[INFO] device = {device}")

    # 1) Dataset --------------------------------------------------------------
    print("[INFO] Đang sinh dataset...")
    train_ds, val_ds, test_ds = make_train_val_test(
        n_train=cfg.TRAIN["n_samples_train"],
        n_val=cfg.TRAIN["n_samples_val"],
        n_test=cfg.TRAIN["n_samples_test"],
    )

    # 2) Fit normalizer trên train, áp lên TẤT CẢ ----------------------------
    print("[INFO] Đang fit normalizer trên train set...")
    normalizer = fit_normalizer(train_ds)

    def apply_norm(ds):
        for i in range(ds.len()):
            normalizer.transform(ds.get(i))

    apply_norm(train_ds)
    apply_norm(val_ds)
    apply_norm(test_ds)

    # 3) DataLoader -----------------------------------------------------------
    train_loader = DataLoader(train_ds, batch_size=cfg.TRAIN["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.TRAIN["batch_size"])
    test_loader  = DataLoader(test_ds,  batch_size=cfg.TRAIN["batch_size"])

    # 4) Model ----------------------------------------------------------------
    model = StiffenedPanelGNN(**cfg.MODEL).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model có {n_params:,} tham số.")

    # 5) Loss + Optimizer -----------------------------------------------------
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN["learning_rate"],
        weight_decay=cfg.TRAIN["weight_decay"],
    )

    # 6) Vòng lặp train -------------------------------------------------------
    best_val_loss  = float("inf")
    no_improve     = 0
    best_ckpt_path = cfg.CHECKPOINT_DIR / "best.pt"

    for epoch in range(1, cfg.TRAIN["n_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            torch.save({
                "model_state":   model.state_dict(),
                "normalizer":    normalizer.__dict__,
                "config_model":  cfg.MODEL,
                "epoch":         epoch,
                "val_loss":      val_loss,
                "val_acc":       val_acc,
            }, best_ckpt_path)
            no_improve = 0
        else:
            no_improve += 1

        if epoch % cfg.LOG_INTERVAL == 0 or epoch == 1 or improved:
            tag = "  ★" if improved else ""
            print(f"  Epoch {epoch:3d} | train MSE={train_loss:.4f} | "
                  f"val MSE={val_loss:.4f} | val acc={val_acc*100:5.2f}%{tag}")

        if no_improve >= cfg.TRAIN["early_stop_patience"]:
            print(f"[INFO] Early stop tại epoch {epoch} (không cải thiện {no_improve} epoch)")
            break

    # 7) Test với checkpoint tốt nhất ----------------------------------------
    print(f"\n[INFO] Đang load best checkpoint từ {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"[RESULT] Test MSE = {test_loss:.4f}")
    print(f"[RESULT] Test max-stress accuracy = {test_acc*100:.2f}%")
    print(f"[INFO] Best checkpoint: epoch {ckpt['epoch']}, val MSE {ckpt['val_loss']:.4f}")


if __name__ == "__main__":
    main()
