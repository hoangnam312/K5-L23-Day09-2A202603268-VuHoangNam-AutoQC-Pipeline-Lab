# Báo cáo Day 09 — <Họ và tên>, <MSHV>

> Xoá mọi `TODO(học viên)` và dòng gợi ý. Mỗi ý trả lời bằng **số lấy từ `submission/reports/`**, kèm mẫu số. Cả báo cáo ~1–2 trang.

## 1. Kết luận trước
TODO(học viên): pipeline kết luận **Release / Hold / Rework**? Bao nhiêu ảnh vào, bao nhiêu ảnh ra dataset? Lý do chính (1–3 câu).

## 2. Lỗi và phân việc
Lấy từ `02_error_list_v1` (gom theo mã lỗi). Cột "ai sửa": **máy** (autofix) hay **người** (hàng đợi); chỉ ghi những mã có mặt trong lô của bạn.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| TODO(học viên) | | | |

TODO(học viên): vì sao `05_review_queue.csv` xếp ảnh đầu hàng như vậy? Với ngân sách 80 ảnh, người xem được bao nhiêu lỗi ngữ nghĩa?

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 16 | `DUP_IMAGE_NEAR`: 1.000 / 0.500 → 1.000 / 0.833 | TP tăng 3/6 → 5/6, FP vẫn 0; trùng byte vẫn bắt đủ 3/3. Đây là ngưỡng nhỏ nhất trong sweep đạt kết quả này; tăng lên 20 làm recall giảm. |
| `second_opinion.conf` | 0.25 | 0.6 | `WRONG_CLASS`: 0.556 / 0.833 → 0.625 / 0.833; `MISSING`: 0.500 / 1.000 → 0.571 / 1.000 | Giữ TP nhầm lớp 5/6 và thiếu box 8/8, giảm FP tương ứng 4 → 3 và 8 → 6. Chấp nhận recall `LOOSE_BOX` giảm 2/5 → 1/5; không tăng lên 0.7 vì bắt đầu bỏ sót lỗi thiếu box. |

Nguồn: `submission/calibration/calibration.json` và `calibration_report.md` (2 sweep); kết quả kiểm tra lại cấu hình đã chọn nằm trong `submission/calibration/selected/`. Với `conf=0.6`, `POSSIBLE_LOOSE_BOX` có precision 0.200, recall 0.200; `POSSIBLE_FALSE_POS` vẫn có precision 0.312, recall 1.000 (TP=5, FP=11), vì nhánh này dùng `fp_conf` cố định.

Calibration có thể đánh lừa vì chỉ có 39 ảnh và 6 lỗi gần trùng: bắt thêm 1 ảnh đã làm recall tăng khoảng 16.7 điểm phần trăm. Precision 1.000 tại `phash_dist=16` chỉ có nghĩa không có FP trên bộ calib, chưa chứng minh không xóa nhầm trên raw. Vẫn bỏ sót 1/6 ảnh gần trùng và 4/5 lỗi box lỏng ở cấu hình đã chọn. Ở CP4 cần mở xem các cặp ảnh raw bị báo trùng trước khi kết luận ngưỡng phù hợp với dữ liệu thật.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
TODO(học viên): (a) chứng minh gate chặn đúng: chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt nên ảnh thiếu file nhãn còn nguyên), chỉ ra kết luận **Rework**, mã thoát 20 và việc **không xuất** dataset. (Đừng sửa tay `work/run/ds`: pipeline dựng lại thư mục này từ raw mỗi lần chạy.) (b) ≥ 2 điểm mù có bằng chứng (box lỏng nhẹ, nhầm lớp mà model lớn cũng nhầm, ảnh trùng biến đổi mạnh…).

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa train: ghi "chưa chạy" và lý do, rồi bỏ qua phần còn lại của mục này. Nếu đã train: dán bảng từ `submission/results/comparison.md` (model, epoch, seed, số ảnh train mỗi biến thể).

TODO(học viên): chênh mAP50 (clean − raw) so với **độ lệch chuẩn giữa các seed**; kết luận **trung thực** (nằm trong nhiễu thì nói là chưa kết luận được); ≥ 2 giả thuyết giải thích (ảnh trùng/rò rỉ test trong `raw`, nhãn máy còn lại trong `clean`, test chỉ 100 ảnh…) và cách kiểm.

## 6. Quyết định và ghi chú AI
TODO(học viên): (a) quyết định cuối; nếu Hold/Rework cần làm gì để thành Release? (b) Dùng AI viết code? Ghi 3 dòng: prompt chính, cách bạn kiểm code AI, 1 lần AI sai. Không dùng: ghi "không dùng".

*(Thưởng +2, tùy chọn)* CVAT: lọc `width < 20`, ảnh/số box khớp, đối chiếu `BOX_TINY` (xem docs/CVAT_FILTER.md), 1 câu về `can_review`.
