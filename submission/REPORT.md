# Báo cáo Day 09 — Vũ Hoàng Nam, 2A202603268

## 1. Kết luận trước

Pipeline kết luận **Hold**, tức là cần tiếp tục kiểm tra trước khi phát hành dataset. Trong 224 ảnh đầu vào, pipeline loại 22 ảnh và xuất 202 ảnh vào `clean_v1`.

Sau khi máy sửa lỗi và bước review mô phỏng xử lý 80 ảnh, không còn ERROR. Tuy nhiên, còn 38/202 ảnh (18,81%) có WARN chưa được xử lý, vượt ngưỡng 10%. Vì vậy, dataset được xuất chỉ là bản nháp để thử nghiệm. Số liệu lấy từ `submission/reports/08_summary.json` và `submission/export/clean_v1/manifest.json`.

## 2. Lỗi và phân việc

Bảng dưới lấy từ `02_error_list_v1.csv`, gồm lỗi ở bước nhận 224 ảnh và lỗi nhãn trên 202 ảnh được giữ lại. Một ảnh có thể có nhiều lỗi, nên không cộng các dòng trong bảng để tính tổng số ảnh lỗi.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| CORRUPT_IMAGE | 2/224 ảnh | Máy | Loại ảnh hỏng, không đọc được. |
| BAD_SIZE | 3/224 ảnh | Máy | Loại ảnh có kích thước không đạt yêu cầu. |
| DUP_IMAGE_EXACT | 8/224 ảnh | Máy | Loại bản sao giống hệt ảnh đã có. |
| DUP_IMAGE_NEAR | 4/224 ảnh | Máy | Loại ảnh gần trùng theo threshold đã chọn. |
| LEAK_TEST | 5/224 ảnh | Máy | Loại ảnh trùng với kho test/validation để tránh rò rỉ dữ liệu. |
| BAD_CLASS_ID | 6 dòng trên 6 ảnh | Máy | Xóa dòng có class ID không hợp lệ. |
| BAD_FORMAT | 3 dòng trên 3 ảnh | Máy | Xóa dòng không đúng định dạng YOLO. |
| BOX_ZERO | 4 dòng trên 4 ảnh | Máy | Xóa box có chiều rộng hoặc chiều cao không hợp lệ. |
| DUP_BOX | 7 dòng trên 7 ảnh | Máy | Xóa box bị lặp trong cùng ảnh. |
| BOX_OUT_OF_IMAGE | 8 dòng trên 8 ảnh | Máy | Cắt box về trong ảnh; xóa nếu sau khi cắt không còn diện tích. |
| BAD_CLASS_NAME | 2 lỗi ở cấp dataset | Máy | Sửa tên lớp theo config. |
| MISSING_LABEL_FILE | 6/202 ảnh | Người | Cần gán nhãn; máy không được tự tạo file rỗng để thay thế. |
| POSSIBLE_FALSE_POS | 29 cảnh báo trên 27/202 ảnh | Người | Xem lại box có thể đã được gán vào nơi không có vật thể. |
| POSSIBLE_LOOSE_BOX | 28 cảnh báo trên 24/202 ảnh | Người | Xem lại box có thể bị lệch hoặc bao quá rộng. |
| POSSIBLE_MISSING | 53 cảnh báo trên 52/202 ảnh | Người | Kiểm tra vật thể có thể bị thiếu box. |
| POSSIBLE_WRONG_CLASS | 18 cảnh báo trên 18/202 ảnh | Người | Kiểm tra vật thể có thể bị gán sai lớp. |
| BOX_TINY | 6 cảnh báo trên 4/202 ảnh | Người | Đối chiếu box nhỏ với guideline. |
| BOX_HUGE | 3/202 ảnh | Người | Kiểm tra box chiếm diện tích quá lớn. |
| EMPTY_LABEL | 12/202 ảnh | Người | Xác nhận ảnh thật sự không có vật thể cần gán nhãn. |
| CLASS_RARE | 4 cảnh báo ở cấp dataset | Người | Xem lại phân bố lớp; không tự đổi nhãn để cân bằng số lượng. |

Các lỗi theo dòng trong bảng được phát hiện trên 202 ảnh được giữ lại. Ở lần QC đầu, máy ghi nhận 36 ERROR và 153 WARN. File `03_fixes.csv` có 36 bản ghi xử lý; trong đó `needs_label` chỉ là chuyển ảnh thiếu nhãn cho người, chưa có nghĩa là lỗi đã được sửa.

Ảnh `img_0020` đứng đầu `05_review_queue.csv` với điểm ưu tiên 11,524 vì vừa có dòng nhãn bị xóa, vừa có cảnh báo thiếu box và sai lớp. Điểm được tính từ trọng số của từng loại cảnh báo và confidence; các ảnh thiếu file nhãn có điểm ưu tiên 10.

Hàng đợi có 125 ảnh nhưng budget chỉ đủ 80 ảnh. Theo `06_review_log.csv`, bước review mô phỏng đã xử lý 80 ảnh; 74 ảnh trong số đó chứa 98/128 cảnh báo `POSSIBLE_*` ban đầu. Đây là số cảnh báo được đưa ra xem xét, không phải số lỗi ngữ nghĩa đã được xác nhận đúng.

## 3. Calibration (2 ngưỡng)

Trong bảng, Precision cho biết tỷ lệ cảnh báo đúng trong số cảnh báo được tạo ra; Recall cho biết tỷ lệ lỗi chèn sẵn mà công cụ phát hiện được.

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 10 | Ảnh gần trùng: 1,000 / 0,500 → 1,000 / 0,667 | Phát hiện đúng tăng từ 3/6 lên 4/6 ảnh gần trùng, không báo nhầm trên calib. Chấp nhận bỏ sót một số ảnh trùng để giảm nguy cơ loại nhầm ảnh tốt trên raw. |
| `second_opinion.conf` | 0.25 | 0.6 | Sai lớp: 0,556 / 0,833 → 0,625 / 0,833. Thiếu box: 0,500 / 1,000 → 0,571 / 1,000 | Vẫn bắt được 5/6 lỗi sai lớp và 8/8 lỗi thiếu box. Số cảnh báo nhầm tương ứng giảm từ 4 xuống 3 và từ 8 xuống 6, nhưng lỗi box lỏng chỉ bắt được 1/5 thay vì 2/5. |

Nguồn là hai sweep trong `submission/calibration/calibration.json` và `calibration_report.md`. Kết quả chạy lại với config cuối cùng nằm trong `submission/calibration/selected/`. Không tăng `conf` lên 0.7 vì khi đó bắt đầu bỏ sót lỗi thiếu box. Cảnh báo `POSSIBLE_FALSE_POS` dùng `fp_conf` cố định, nên vẫn có 5 cảnh báo đúng và 11 cảnh báo nhầm dù thay đổi `conf`.

Một điểm calibration có thể gây hiểu nhầm là **kết quả tốt trên calib chưa chắc đúng trên raw**. Ban đầu, threshold 16 không báo nhầm trên 39 ảnh calib. Nhưng khi mở xem 7 cặp ảnh raw bị báo gần trùng, có hai cặp rõ ràng khác nhau: `img_0008` là cảnh trượt tuyết còn `img_0011` là tennis; `img_0113` là người ngồi bàn còn `img_0177` là người lướt sóng. Cả hai cặp đều có phash distance bằng 16.

Vì vậy, threshold được giảm xuống 10 và pipeline được chạy lại. Cặp `img_0061/img_0186` có distance 10 vẫn là cùng cảnh ghế công viên nên được giữ trong nhóm gần trùng. Cách chọn này giảm Recall trên calib từ 5/6 ở threshold 16 xuống 4/6, nhưng tránh được hai trường hợp loại nhầm ảnh đã nhìn thấy. Chưa thể khẳng định mọi ảnh bị loại trên raw đều chính xác.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được

Để kiểm tra gate, đã chạy pipeline với budget bằng 0, tức là không có bước review ảnh:

```cmd
python -m pipeline.run --budget 0 --work work/run_nb --reports work/rep_nb --export-dir work/exp_nb
```

Kết quả là **Rework**, exit code **20**, vì vẫn còn 6/202 ảnh thiếu file nhãn. Trong `work/rep_nb/08_summary.json`, trường `exported` bằng `null`; thư mục `work/exp_nb/clean_v1` không được tạo. Thí nghiệm dùng thư mục riêng nên không ghi đè kết quả chạy chính.

Khi dùng budget 80, bước review mô phỏng xử lý các ảnh thiếu nhãn nên không còn ERROR. Tuy nhiên, pipeline vẫn Hold vì còn quá nhiều ảnh có WARN chưa xử lý. Exit code 0, 10 và 20 lần lượt tương ứng với Release, Hold và Rework, giúp CI quyết định bước tiếp theo.

Pipeline còn các điểm mù sau:

- **Bỏ sót ảnh gần trùng:** với threshold 10, công cụ chỉ bắt được 4/6 trường hợp trên calib, còn bỏ sót 2/6. Hash không nhận ra được mọi cách biến đổi ảnh.
- **Bỏ sót lỗi nhãn ngữ nghĩa:** với `conf=0.6`, công cụ bỏ sót 4/5 lỗi box lỏng và 1/6 lỗi sai lớp trên calib. Vì vậy, ảnh không bị cảnh báo vẫn có thể có nhãn sai.

Review trong lần chạy này là mô phỏng; không có nghĩa người thật đã xem và xác nhận toàn bộ dataset.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)

**Chưa chạy.** Bài hiện hoàn thiện phần core trước, chưa thực hiện phần train tùy chọn trên Colab. Do chưa có mAP và kết quả giữa các seed, chưa đủ cơ sở kết luận clean tốt hơn raw.

## 6. Quyết định và ghi chú AI

Quyết định cuối cùng là **Hold**. Để đạt Release với 202 ảnh, số ảnh còn WARN chưa xử lý phải giảm từ 38 xuống tối đa 20 ảnh, đồng thời không còn ERROR. Như vậy, cần xử lý dứt điểm cảnh báo của ít nhất 18 ảnh nữa trong một vòng làm việc được dự án cho phép; không tự tăng budget hoặc đổi rule để đạt Release.

- **Prompt chính:** nhờ AI kiểm tra 4 module, chạy CP3/CP4 và hoàn thiện CP5. Phiên làm việc hiện tại không có bản prompt viết 4 module ban đầu nên không ghi thêm nội dung chưa xác minh.
- **Cách kiểm:** chạy đủ 79 unit test, calibration trên bộ có lỗi chèn sẵn, mở xem các cặp ảnh gần trùng và thử gate với budget 0. Người học vẫn cần đọc và giải thích được code của mình.
- **Một lần AI sai:** AI chọn `phash_dist=16` dựa trên calib nhưng threshold này loại nhầm ảnh raw. Sau khi kiểm tra ảnh, đã sửa thành 10 và chạy lại để cập nhật số liệu.

Khi kiểm tra bài nộp trên Windows, validator gốc so sánh path dùng dấu `\` với danh sách dùng dấu `/`, nên báo nhầm hai lỗi ở file `calibrate.py` viết sẵn. Đã giữ nguyên file đề và dùng script tạm `work/cp5_validate.py` để chuẩn hóa path khi chạy, không bỏ qua bước kiểm tra nào. Kết quả: **79 test passed, 0 lỗi và 1 cảnh báo do chưa train**. Chạy trực tiếp validator gốc trên Windows vẫn có thể gặp hai lỗi báo nhầm này.
