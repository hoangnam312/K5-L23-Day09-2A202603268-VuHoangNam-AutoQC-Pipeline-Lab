"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    remove_prefixes = ("DUP_IMAGE_",)
    remove_codes = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    removed_images = set()
    changes = []
    by_file = defaultdict(dict)

    for issue in issues:
        if issue.severity != ERROR:
            continue
        code, stem = issue.code, issue.image
        if code == "MISSING_LABEL_FILE":
            changes.append({"image": stem, "line": None, "action": "needs_label",
                            "before": "(không có file)", "after": "chuyển người gán"})
        elif code == "BAD_CLASS_NAME" and ds.yaml_path.exists():
            data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            data["names"] = list(cfg.get("classes", ds.classes))
            ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            changes.append({"image": "data.yaml", "line": None, "action": "rename_classes",
                            "before": "names cũ", "after": "names theo cấu hình"})
        elif code in remove_codes or code.startswith(remove_prefixes):
            removed_images.add(stem)
        elif code in drop_codes and issue.line is not None:
            entry = by_file.setdefault(stem, {}).setdefault(issue.line, {"action": "drop_line", "codes": set()})
            entry["action"] = "drop_line"
            entry["codes"].add(code)
        elif code == "BOX_OUT_OF_IMAGE" and issue.line is not None:
            entry = by_file.setdefault(stem, {}).setdefault(issue.line, {"action": "clip_box", "codes": set()})
            if entry["action"] != "drop_line":
                entry["action"] = "clip_box"
            entry["codes"].add(code)

    for stem in sorted(removed_images):
        image = ds.image_path(stem)
        if image and image.exists():
            image.unlink()
        label = ds.label_path(stem)
        if label.exists():
            label.unlink()
        changes.append({"image": stem, "line": None, "action": "remove_image", "before": stem, "after": "removed"})

    for stem, line_map in by_file.items():
        if stem in removed_images:
            continue
        path = ds.label_path(stem)
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8", errors="replace").splitlines()
        updated = list(original)
        for line_no, info in line_map.items():
            if not 1 <= line_no <= len(original):
                continue
            before = original[line_no - 1]
            if info["action"] == "drop_line":
                updated[line_no - 1] = None
                changes.append({"image": stem, "line": line_no, "action": "drop_line", "before": before, "after": ""})
                continue
            try:
                class_id, cx, cy, width, height = parse_yolo_line(before)
            except (ValueError, OverflowError):
                updated[line_no - 1] = None
                changes.append({"image": stem, "line": line_no, "action": "drop_line", "before": before, "after": ""})
                continue
            x1, y1 = max(0.0, cx - width / 2), max(0.0, cy - height / 2)
            x2, y2 = min(1.0, cx + width / 2), min(1.0, cy + height / 2)
            if x2 <= x1 or y2 <= y1:
                updated[line_no - 1] = None
                changes.append({"image": stem, "line": line_no, "action": "drop_line", "before": before, "after": ""})
            else:
                after = f"{class_id} {(x1+x2)/2:.8g} {(y1+y2)/2:.8g} {x2-x1:.8g} {y2-y1:.8g}"
                updated[line_no - 1] = after
                if after != before:
                    changes.append({"image": stem, "line": line_no, "action": "clip_box", "before": before, "after": after})
        path.write_text("\n".join(line for line in updated if line is not None) + ("\n" if updated else ""), encoding="utf-8")
    return changes
