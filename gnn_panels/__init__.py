"""
gnn_panels — Mạng nơ-ron đồ thị cho dự đoán ứng suất trong stiffened panel.

Reimplementation của Cai & Jelovica (2024), Thin-Walled Structures 203, 112157.

Cấu trúc package:
    config.py        — siêu tham số tập trung
    data/            — sinh data + xây graph
    models/          — kiến trúc GraphSAGE
    utils/           — chuẩn hóa + metrics
    train.py         — pipeline huấn luyện
    predict.py       — pipeline inference
    run_demo.py      — demo end-to-end
"""

__version__ = "0.1.0"
