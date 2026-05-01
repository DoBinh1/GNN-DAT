"""
run_demo.py — Script end-to-end cho người mới: chạy 1 lệnh, xem kết quả.

Chạy:
    python -m gnn_panels.run_demo
hoặc:
    cd gnn_panels && python run_demo.py

Sẽ:
    1. Sinh dataset nhỏ.
    2. Train model trong vài chục epoch.
    3. Predict 1 panel test.
    4. In bảng so sánh và (optional) lưu hình.

Mục đích: kiểm tra pipeline hoạt động end-to-end. Chạy nhanh, dùng
cấu hình nhỏ.
"""

from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from pathlib import Path

# Cho phép chạy trực tiếp `python run_demo.py` từ trong gnn_panels/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Override một số config cho demo nhanh -------------------------------------
import gnn_panels.config as cfg
cfg.TRAIN["n_samples_train"] = 200
cfg.TRAIN["n_samples_val"]   = 50
cfg.TRAIN["n_samples_test"]  = 50
cfg.TRAIN["batch_size"]      = 16
cfg.TRAIN["n_epochs"]        = 50
cfg.TRAIN["early_stop_patience"] = 15
cfg.MODEL["n_layers"]        = 6   # giảm còn 6 cho demo

# ---------------------------------------------------------------------------
from gnn_panels.train import main as train_main
from gnn_panels.predict import main as predict_main


if __name__ == "__main__":
    print("=" * 70)
    print(" DEMO: GNN cho stiffened panel — pipeline end-to-end")
    print("=" * 70)
    print()
    print(">>> BƯỚC 1: TRAINING")
    print("-" * 70)
    train_main()
    print()
    print(">>> BƯỚC 2: INFERENCE & SO SÁNH")
    print("-" * 70)
    predict_main()
    print()
    print("=" * 70)
    print(" DEMO HOÀN TẤT.")
    print(" Xem outputs/checkpoints/ và outputs/logs/ để xem kết quả.")
    print("=" * 70)
