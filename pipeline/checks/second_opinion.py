"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")
    options = cfg.get("second_opinion", {})
    conf_min = float(options.get("conf", 1.0))
    min_area = float(options.get("min_area_px", 0.0))
    fp_conf = float(options.get("fp_conf", conf_min))
    iou_missing = float(options.get("iou_missing", 0.0))
    iou_match = float(options.get("iou_match", 1.0))
    iou_loose = float(options.get("iou_loose", 1.0))
    issues = []

    for image_path in ds.image_files():
        stem = image_path.stem
        lines = ds.read_label_lines(stem)
        if stem not in preds or lines is None:
            continue
        labels = []
        for line_no, raw in lines:
            try:
                class_id, cx, cy, width, height = parse_yolo_line(raw)
            except (ValueError, OverflowError):
                continue
            if 0 <= class_id < len(ds.classes):
                labels.append((line_no, class_id, (cx, cy, width, height)))
        size = ds.image_size(stem)
        if not size:
            continue
        width_px, height_px = size
        valid_preds = []
        for pred in preds[stem]:
            try:
                class_id, confidence, cx, cy, width, height = pred
                vals = (float(confidence), float(cx), float(cy), float(width), float(height))
                if not all(v == v and abs(v) != float("inf") for v in vals):
                    continue
                valid_preds.append((int(class_id), vals[0], (vals[1], vals[2], vals[3], vals[4])))
            except (TypeError, ValueError):
                continue

        for class_id, confidence, box in valid_preds:
            area_px = box[2] * width_px * box[3] * height_px
            if confidence < conf_min or area_px < min_area:
                continue
            candidates = [(label_cls, iou_xywh(box, label_box)) for _, label_cls, label_box in labels]
            candidates = [(label_cls, overlap) for label_cls, overlap in candidates if overlap >= iou_missing]
            if not candidates:
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_MISSING", stem,
                                    "Model phát hiện vật thể chưa có box nhãn.", extra={"conf": confidence}))
                continue
            same = [overlap for label_cls, overlap in candidates if label_cls == class_id]
            if same:
                best_same = max(same)
                if best_same < iou_loose:
                    issues.append(Issue("second_opinion", WARN, "POSSIBLE_LOOSE_BOX", stem,
                                        "Box nhãn cùng lớp có thể bị lệch hoặc quá rộng.", extra={"conf": confidence, "iou": best_same}))
                continue
            other = max(overlap for label_cls, overlap in candidates if label_cls != class_id)
            if other >= iou_match:
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_WRONG_CLASS", stem,
                                    "Box khớp với nhãn có ID lớp khác.", extra={"conf": confidence, "iou": other}))

        for line_no, _, label_box in labels:
            matching = [iou_xywh(label_box, box) for _, confidence, box in valid_preds if confidence >= fp_conf]
            if all(overlap < iou_missing for overlap in matching):
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_FALSE_POS", stem,
                                    "Nhãn không khớp với dự đoán model đủ độ tin cậy.", line_no,
                                    {"iou": max(matching, default=0.0)}))
    return issues
