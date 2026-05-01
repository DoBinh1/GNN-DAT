"""
synthetic_panel.py — Sinh dữ liệu giả lập có ý nghĩa vật lý.

Vì không có ABAQUS, ta tạo dữ liệu nhân tạo mà vẫn TUÂN THỦ các quy luật
cơ học cơ bản:

  1. Ứng suất tỷ lệ thuận với áp suất tải.
  2. Ứng suất tỷ lệ nghịch với độ dày (gần đúng: t^2 cho plate bending).
  3. Pattern phân bố stress phụ thuộc điều kiện biên:
        - Fixed boundary  → ứng suất cao ở MÉP, thấp ở GIỮA.
        - Simply supported → ứng suất cao ở GIỮA, thấp ở MÉP.
  4. Ứng suất tại web/flange thấp hơn plate ở các thiết kế thông thường.

Mục đích: model GNN học được các quy luật này. Khi sau này thay bằng dữ liệu
ABAQUS thật, KIẾN TRÚC code không thay đổi — chỉ đổi nguồn data.
"""

from __future__ import annotations
import numpy as np
from typing import Dict
from .. import config as cfg


# ---------------------------------------------------------------------------
# 1) Sinh tham số thiết kế ngẫu nhiên (uniform trong miền cho phép)
# ---------------------------------------------------------------------------
def generate_panel_design(seed: int) -> Dict:
    """
    Sinh một thiết kế panel ngẫu nhiên, deterministic theo seed.

    Trả về dict gồm:
        - kích thước plate, gân, flange
        - số gân
        - áp suất
        - điều kiện biên cho plate và stiffener (mã hóa thành số nguyên)

    Mã hóa BC:
        Plate edges:    0 = simply supported, 1 = fixed
        Stiffener edges: 0 = free, 1 = simply supported, 2 = fixed
    """
    rng = np.random.default_rng(seed)
    R = cfg.PARAM_RANGES

    return {
        "seed":             int(seed),
        "plate_length":     cfg.PANEL_LENGTH,
        "plate_width":      cfg.PANEL_WIDTH,
        "plate_thickness":  float(rng.uniform(*R["plate_thickness"])),
        "n_stiffeners":     int(rng.integers(R["n_stiffeners"][0], R["n_stiffeners"][1] + 1)),
        "stiff_height":     float(rng.uniform(*R["stiff_height"])),
        "stiff_thickness":  float(rng.uniform(*R["stiff_thickness"])),
        "flange_width":     float(rng.uniform(*R["flange_width"])),
        "flange_thickness": float(rng.uniform(*R["flange_thickness"])),
        "pressure":         float(rng.uniform(*R["pressure"])),
        # 4 cạnh plate, mỗi cạnh nhận một BC
        "bc_plate":         rng.integers(0, 2, size=4).tolist(),
        # 4 cạnh stiffener (web + flange), mỗi cạnh nhận một BC
        "bc_stiff":         rng.integers(0, 3, size=4).tolist(),
    }


# ---------------------------------------------------------------------------
# 2) Sinh trường ứng suất giả lập cho từng loại đơn vị cấu trúc
# ---------------------------------------------------------------------------
def _base_stress(pressure: float, span_mm: float, thickness_mm: float,
                 support_factor: float = 1.0) -> float:
    """
    Biên độ ứng suất uốn tấm xấp xỉ: σ ≈ k · p · (L/t)² · support_factor.

    Đây là công thức RÚT GỌN cho ứng suất uốn cực đại trong một tấm chữ
    nhật chịu áp suất phân bố đều. Hằng số k = 0.5 được hiệu chỉnh để giá
    trị stress nằm trong khoảng vật lý hợp lý (vài chục đến vài trăm MPa).

    Đơn vị: pressure (MPa), span (mm), thickness (mm). Trả về MPa.
    """
    L_m = span_mm / 1000.0
    t_m = max(thickness_mm, 1.0) / 1000.0
    sigma = 0.5 * pressure * (L_m / t_m) ** 2 * support_factor
    return float(np.clip(sigma, 5.0, 600.0))


def _stress_pattern(rows: int, cols: int, bc_avg: float, sigma_max: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Sinh pattern phân bố ứng suất 2D theo điều kiện biên.

    Triết lý: pattern phải SMOOTH và UNIMODAL — chỉ có 1 đỉnh, để max-stress
    metric không bị nhiễu bởi spike. Nếu bc_avg = 0 (simply supported), đỉnh
    nằm ở GIỮA. Nếu bc_avg = 1 (fixed), đỉnh nằm ở MÉP. Mix tuyến tính giữa
    hai trạng thái sao cho luôn smooth.

    Noise được giảm xuống 1% (so với 5% cũ) — thực tế FEM solutions cũng
    rất smooth, không có spike random.
    """
    x = np.linspace(-1, 1, rows)
    y = np.linspace(-1, 1, cols)
    X, Y = np.meshgrid(x, y, indexing="ij")
    R2 = X ** 2 + Y ** 2

    # Hai pattern smooth & unimodal:
    #   - center_pattern: max ở giữa (R²=0), giảm theo Gaussian
    #   - edge_pattern: max ở góc (R²=2), tăng theo R²
    center_pattern = np.exp(-1.5 * R2)        # max=1 ở (0,0)
    edge_pattern   = R2 / 2.0                  # max=1 ở góc (1,1)

    # Trộn — bc_avg=0 → toàn center, bc_avg=1 → toàn edge
    grid = (1 - bc_avg) * center_pattern + bc_avg * edge_pattern

    # Chuẩn hóa max về 1 rồi nhân biên độ
    grid = grid / (grid.max() + 1e-8) * sigma_max

    # Noise rất nhỏ ~1% — đủ để pattern không hoàn toàn deterministic
    grid = grid * (1.0 + 0.01 * rng.standard_normal(grid.shape))
    return grid.astype(np.float32)


def generate_pseudo_stress_field(unit_type: str, design: Dict, unit_idx: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    Sinh trường ứng suất 2D (lưới 10x20) cho một đơn vị cấu trúc.

    Triết lý: tất cả 3 loại unit phải có stress cùng order of magnitude
    (vài chục đến vài trăm MPa) — đây là sự thật vật lý: trong một panel
    thực, plate, web và flange đều chịu stress comparable, không chênh
    lệch hàng nghìn lần như công thức cũ.

    Để đảm bảo điều này:
        - PLATE dùng span = khoảng cách giữa 2 stiffener (= width/n).
          Đây mới là "bending span" thực sự của plate, KHÔNG phải toàn
          bộ chiều dài 3m của panel.
        - WEB và FLANGE dùng span tương ứng với kích thước hình học của
          chúng, kèm hệ số amplification mô phỏng "load transfer từ plate".

    Args:
        unit_type: 'plate' | 'web' | 'flange'
        design:    dict tham số panel
        unit_idx:  chỉ số của đơn vị
        rng:       random generator để noise

    Returns:
        np.ndarray shape (rows*cols,) — đã flatten thành vector 200 chiều
    """
    rows, cols = cfg.STRESS_GRID
    p = design["pressure"]

    if unit_type == "plate":
        # Span đúng là khoảng cách giữa 2 gân (chiều ngắn của plate-span).
        # KHÔNG dùng plate_length = 3000mm vì điều đó cho stress phi thực tế.
        span = design["plate_width"] / (design["n_stiffeners"] + 1)
        t    = design["plate_thickness"]
        bc_avg = float(np.mean(design["bc_plate"]))
        support_factor = 0.6 + 0.6 * bc_avg
        sigma_max = _base_stress(p, span, t, support_factor)

    elif unit_type == "web":
        # Web là tấm đứng cao stiff_height, dày stiff_thickness.
        # Stress trong web chủ yếu do load transfer từ plate qua mối hàn.
        span = design["stiff_height"]
        t    = design["stiff_thickness"]
        bc_avg = float(np.mean(design["bc_stiff"])) / 2.0   # range [0,1]
        support_factor = 0.4 + 0.4 * bc_avg
        # Hệ số khuếch đại = 4.0 mô phỏng load transfer từ plate vào web
        sigma_max = _base_stress(p, span, t, support_factor) * 4.0
        sigma_max = float(np.clip(sigma_max, 5.0, 600.0))

    elif unit_type == "flange":
        # Flange là tấm ngang trên đỉnh web, chịu kéo/nén do uốn của web.
        span = design["flange_width"]
        t    = design["flange_thickness"]
        bc_avg = float(np.mean(design["bc_stiff"])) / 2.0
        support_factor = 0.3 + 0.3 * bc_avg
        # Hệ số 3.0 < hệ số web vì flange xa load source hơn
        sigma_max = _base_stress(p, span, t, support_factor) * 3.0
        sigma_max = float(np.clip(sigma_max, 5.0, 500.0))

    else:
        raise ValueError(f"unknown unit_type: {unit_type}")

    # Biến thiên nhỏ giữa các đơn vị cùng loại (mô phỏng vị trí ở mép vs giữa)
    sigma_max *= (1.0 + 0.05 * np.sin(unit_idx * 0.7))

    grid = _stress_pattern(rows, cols, bc_avg, sigma_max, rng)
    return grid.flatten()
