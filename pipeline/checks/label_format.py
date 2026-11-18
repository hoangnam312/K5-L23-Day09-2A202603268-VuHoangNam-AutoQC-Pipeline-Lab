"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    issues = []
    options = cfg.get("label_format", {})
    overflow_tol = float(options.get("overflow_tol", 0.0))
    min_box_px = float(options.get("min_box_px", 0.0))
    max_box_frac = float(options.get("max_box_frac", 1.0))
    dup_box_iou = float(options.get("dup_box_iou", 1.0))

    for path in ds.label_files():
        stem = path.stem
        previous = []
        size = ds.image_size(stem)
        for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                class_id, cx, cy, width, height = parse_yolo_line(raw)
            except (ValueError, OverflowError) as exc:
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem,
                                    f"Dòng nhãn không hợp lệ: {exc}", line_no))
                continue
            if not 0 <= class_id < len(ds.classes):
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", stem,
                                    f"ID lớp {class_id} nằm ngoài danh sách lớp.", line_no))
                continue
            if width <= 0 or height <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem,
                                    "Chiều rộng và chiều cao box phải lớn hơn 0.", line_no))
                continue

            box = (cx, cy, width, height)
            overflow = max(0.0, - (cx - width / 2), - (cy - height / 2),
                           cx + width / 2 - 1, cy + height / 2 - 1)
            if overflow > overflow_tol:
                issues.append(Issue("label_format", ERROR, "BOX_OUT_OF_IMAGE", stem,
                                    f"Box tràn khỏi ảnh {overflow:.6g} đơn vị chuẩn hóa.",
                                    line_no, {"overflow": overflow}))
            if any(old_class == class_id and iou_xywh(box, old_box) > dup_box_iou
                   for old_class, old_box in previous):
                issues.append(Issue("label_format", ERROR, "DUP_BOX", stem,
                                    "Box trùng với một box cùng lớp ở dòng trước.", line_no))
            previous.append((class_id, box))
            if size and min(width * size[0], height * size[1]) < min_box_px:
                issues.append(Issue("label_format", WARN, "BOX_TINY", stem,
                                    "Cạnh ngắn của box nhỏ hơn ngưỡng pixel.", line_no))
            if width * height > max_box_frac:
                issues.append(Issue("label_format", WARN, "BOX_HUGE", stem,
                                    "Diện tích box vượt ngưỡng cấu hình.", line_no))
    return issues
