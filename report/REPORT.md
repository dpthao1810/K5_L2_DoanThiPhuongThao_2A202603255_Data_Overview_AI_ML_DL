# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy? Record này nói về cả bức ảnh, vì trong report không có bbox, chỉ có nhãn lớp cho toàn ảnh. Model cũng không chỉ ra vật ở đâu.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint? Class list đi kèm taxonomy/checkpoint (ImageNet-1K), đã được cố định khhi mở ra, vì vậy Google Colab không thể tự tạo ra mà chỉ chọn trong danh sách.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp? Vì class_id chỉ là số thứ tự, chỉ có nghĩa trong một taxonomy cụ thể. Tên lớp có thể trùng. Vì vậy cần ghi đủ cả 3 trường để người khác tra lại đúng.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? Guideline phải có quy tắc chọn 1 lớp. Ví dụ, chọn vật/chủ đề nổi bật nhất (to nhất, ở giữa, rõ nhất), có lựa chọn"không rõ khác" hoặc escalation khi vẫn không quyết định được. 
6. Vì sao không được lấy `score` của model làm ground truth? Vì score là mức tin của model, model có thể sai mà vẫn tự tin cao. Còn ground truth là nhãn do người gán và QC. Vì vậy score không thể nói nhãn đúng hay sai. 

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="bus" ; `score`=0.912557 ; `bbox_xyxy`=[93.17, 187.95, 223.01, 320.91] ; `bbox_width`=129.84 ; `bbox_height`=132.96
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? Góc trên-trái ở (120.4, 85.2), góc dưới-phải ở (310.7, 290.9). Gốc tọa độ ở góc trên-trái ảnh, x tăng sang phải, y tăng xuống dưới (đơn vị pixel). Box rộng 190 px, cao 206 px.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? Ngưỡng 0.20: 17 prediction, ngưỡng 0.60: 6 prediction 
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì? Reviewer cần phải kiểm tra từng box, xóa hoặc sửa box sai, đối chiếu với guideline.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này). Quy tắc box: mỗi vật thể thuộc danh sách lớp được vé một box hình chữ nhật sát mép phần nhìn thấy của vật, không chừa khoảng trống thừa và không gán vào phần bóng đổ hoặc phần bị che. 
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao? Chiếc bát trên bàn bị che một nửa. Nếu tình huống lặp lại thường xuyên và có đáp án rõ ràng thì dùng quy tắc trong guideline: vật bị che khuất một phần thì hãy vẽ box xung quanh phần nhìn thấy được, không thấy hoặc bị che thì không kéo dài bbox qua đó, chỉ bao quanh phần nhìn thấy thôi. Nếu chưa có quy tắc nào bao trùm thì hỏi Lab coach hoặc quyết định thay đổi nhãn, không được tự bịa. 

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="traffic-001" ; `class_name`="bus" ; `score`=0.925745 ; số điểm polygon=`polygon_point_count`=120 ; 3–5 điểm đầu của `polygon_xy`=[148.0, 189.0],[147.0, 190.0],[145.0, 190.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể? Box chỉ cho vùng chữ nhật bao quanh vật thể, có cả nền ở các góc. Polygon cho hình dạng thật của vật, gồm đường cong, phần lõm, phần khuyết, nên sẽ tính được diện tích chính xác của vật hơn. 
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào? instance_id dùng để phân biệt từng vật riêng lẻ trong cùng ảnh. Nó không phải class_id vì class_id cho biết loại vật, còn instance id cho biết đó là vật nào. Còn tracking id theo dõi một vật qua nhiều khung hình video, còn instance id chỉ dùng cho ảnh tĩnh. 
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo. Vé polygon bám theo mép nhìn thấy của vật, gồm toàn bộ phần thấy được và loại bỏ vật che phía trước, đủ điểm để giữ các đoạn cong, không để lọt nền vào trong mask. 
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao? Oven bị biên mờ một đoạn. Nếu tình huống lặp lại thường xuyên và có đáp án rõ ràng thì dùng quy tắc trong guideline: vật bị che khuất một phần hoặc biên mờ thì hãy chấm đúng các điểm xung quanh phần nhìn thấy được, không thấy hoặc bị che thì không kéo dài chấm điểm qua đó, chỉ bao quanh phần nhìn thấy thôi. Nếu chưa có quy tắc nào bao trùm thì hỏi Lab coach, không được tự bịa. 

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho cả ảnh |Ảnh traffic có nhiều chủ đề như đèn, xe, biển,..., model chọn "traffic light" nhưng ảnh có thể xem là "cảnh đường phố".  | Chọn một lớp theo quy tắc chủ đề chính trong guideline |Xem nhãn có đúng quy tắc chọn chủ đề chính không, xem có ảnh mơ hồ cần escalation không  |
| Phát hiện vật thể |Một box [x1, y1, x2, y2] + lớp cho mỗi vậ | Vật bị che hoặc cắt mép, box dự đoán có thể thừa hoặc thiếu nhãn, hạ ngưỡng sẽ sinh thêm box thừa. |Vẽ box sát phần thấy, gán lớp, không bỏ sót vật trong danh sách lớp  |Xem box có sát không, có thiếu, thừa hay trùng không, lớp có đúng không  |
| Instance segmentation |Một polygon hoặc mask + lớp cho mỗi instance |Vẽ biên theo mép thấy, tách từng instance, đặt instance_id riêng  |Biên mờ, 2 vật chạm nhau, vật bị che làm polygon bị lẹm hoặc dính nền   |Biên có bám đúng không, có lẫn instance không, mask có lọt nền không  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu) Chỉ dùng ảnh mẫu của bài lab, không đưa số điện thoại hay bất kì dữ liệu cá nhân nào vào báo cáo hay input. Họ tên và MSSSV chỉ nằm trong tên repository. 
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai? Nếu gặp ảnh/dữ liệu không đúng phạm vi, tôi sẽ dừng lại ngay và chưa nộp ngay, báo cchoLab Coach hoặc người phụ trách để được hướng dẫn cách giải quyết.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
