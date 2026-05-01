"""
config.py — Tập trung mọi siêu tham số và đường dẫn của project.

Triết lý: KHÔNG để siêu tham số rải rác trong code. Mọi thứ thay đổi giữa các
thí nghiệm (learning rate, số layer, batch size, ...) đều ở đây. Khi muốn
đổi thí nghiệm, chỉ sửa file này.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "outputs" / "dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"

# Tự tạo nếu chưa có
for d in (DATA_DIR, CHECKPOINT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hằng số bài toán (khớp với paper Cai & Jelovica 2024, Bảng 2)
# ---------------------------------------------------------------------------
PANEL_LENGTH = 3000.0   # mm — kích thước cố định 3 m × 3 m
PANEL_WIDTH  = 3000.0

# Miền lấy mẫu cho dataset giả lập
PARAM_RANGES = {
    "plate_thickness":  (10.0, 20.0),    # mm
    "stiff_thickness":  (5.0, 20.0),
    "stiff_height":     (100.0, 400.0),
    "flange_thickness": (5.0, 20.0),
    "flange_width":     (50.0, 150.0),
    "n_stiffeners":     (2, 8),          # int
    "pressure":         (0.05, 0.10),    # MPa
}

# Vật liệu (cố định cho dataset cơ bản)
YOUNG_MODULUS = 200_000.0   # MPa = 200 GPa
POISSON       = 0.3

# ---------------------------------------------------------------------------
# Cấu hình mô hình
# ---------------------------------------------------------------------------
MODEL = {
    "in_dim":   8,        # số feature mỗi node (matching paper)
    "hidden":   64,       # số nơ-ron mỗi lớp ẩn
    "n_layers": 8,        # paper dùng 32; ta dùng 8 cho demo nhanh
    "out_dim":  200,      # 10 × 20 = 200 stress values mỗi node
    "aggr":     "sum",    # 'sum' | 'mean' | 'max' — paper chọn 'sum'
}

# Kích thước lưới sampling stress (output)
STRESS_GRID = (10, 20)    # (rows, cols) — paper chọn 10x20

# ---------------------------------------------------------------------------
# Cấu hình huấn luyện
# ---------------------------------------------------------------------------
TRAIN = {
    "n_samples_train":  800,    # số panel cho train
    "n_samples_val":    100,    # validation
    "n_samples_test":   100,    # test
    "batch_size":       32,     # paper dùng 512; ta dùng 32 cho dataset nhỏ
    "learning_rate":    0.005,  # paper dùng 0.02 với batch 512
    "weight_decay":     1e-4,   # L2 regularization
    "n_epochs":         100,
    "early_stop_patience": 20,  # dừng nếu val không cải thiện sau N epoch
    "seed":             42,
    "device":           "auto", # 'cuda' | 'cpu' | 'auto'
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_INTERVAL = 5   # in log mỗi N epoch
