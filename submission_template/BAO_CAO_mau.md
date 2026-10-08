# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** Susois

**Thành viên:** Hoàng Văn Nam

**Mã học viên:** 2A202602853

**Trạng thái dữ liệu:** `video_4` thiếu 31 ảnh (869/900). Kết quả video này chạy trên toàn bộ ảnh hiện có, giữ số frame gốc, nhưng chưa đáp ứng yêu cầu đủ frame của bài nộp.

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

## 1. Cấu hình đã chọn

Mỗi video: tracker bạn nộp, `conf`, `iou`, điều bạn **nhìn thấy** trên video, và một cấu hình đã thử rồi loại.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | botsort | 0.30 | 0.50 | Trong mẫu frame 11/51/101, hai người bên trái giữ hộp liên tục; người áo vàng phía phải được theo dõi khi tiến ra mép ảnh. BoT-SORT giữ thêm người phía xa so với ByteTrack. | bytetrack, conf=0.30, iou=0.50: ít hộp người phía xa hơn trong mẫu quan sát. |
| video_2 (phố đêm, tĩnh, rất đông) | ocsort | 0.30 | 0.50 | Người đứng gần xe bên phải giữ ID 4 ở frame 11/31/51, tương tự BoT-SORT. OC-SORT giữ thêm hộp người phía xa so với ByteTrack; người nhỏ và vùng đèn sáng vẫn khó. | bytetrack, conf=0.30, iou=0.50: nhiều người nhìn thấy nhưng chưa có hộp trong các frame mẫu. |
| video_3 (camera di động, ảnh nhỏ) | ocsort | 0.30 | 0.50 | Người áo sọc giữ ID 1 trong mẫu đầu và còn ID 1 ở frame 559; người áo xám tiền cảnh có dấu hiệu đổi ID giữa frame 1/51. | bytetrack, conf=0.30, iou=0.50: mất hộp người áo sọc khi camera tiến gần. |
| video_4 (trong nhà, camera di chuyển) | ocsort | 0.30 | 0.50 | Người áo đỏ giữ ID 2 đến frame 101, nhưng có dấu hiệu đổi ID ở các đoạn sau. Vùng kính bên trái khó phân biệt người thật và phản chiếu. Dữ liệu thiếu 31 ảnh. | bytetrack, conf=0.30, iou=0.50: đổi ID người áo đỏ giữa frame 51 và 101 của baseline. |
| video_5 (trên xe bus, giao lộ đông) | ocsort | 0.30 | 0.50 | Người áo đỏ bên phải giữ ID 9 đến frame 101; OC-SORT theo thêm người vùng tối bên trái. Khi xe rẽ, các frame 501/750 vẫn có người nhỏ chưa có hộp. | bytetrack, conf=0.30, iou=0.50: ID người áo đỏ thay đổi trong mẫu baseline. |

### Phạm vi dữ liệu và thử nghiệm

- Dữ liệu chạy: `C:\Su\VinLab\phase2-track4\lec7-tracking-forecasting\data_lab21`.
- Các video 1/2/3/5 có lần lượt 600/1050/837/750 ảnh, đều đọc được.
- `video_4` chỉ có **869/900 ảnh**, thiếu 31 frame. Kết quả trên ảnh còn lại phải được xem là **chưa đủ dữ liệu để nộp chính thức**. Không tạo ảnh thay thế, không bổ sung nhãn.
- So sánh ByteTrack và BoT-SORT ở `conf=0.30, iou=0.50` trên 150 frame đầu mỗi video; riêng `video_4` dùng 110 frame đầu, trước frame thiếu đầu tiên.
- Thử thêm OC-SORT cho cả 5 video. Quét `conf=0.15/0.30/0.50` khi giữ `iou=0.50`; quét `iou=0.40/0.50/0.70` khi giữ `conf=0.30`. Video 1 quét trên 150 frame; các lượt còn lại dùng 60 frame để giảm thời gian CPU. Khi so sánh lượt có độ dài khác nhau, chỉ xét phần frame chung.
- Rà thêm cả năm cấu hình ngưỡng của BoT-SORT trên cùng 60 frame đầu `video_1`, đặt seed PyTorch/OpenCV bằng 42 cho các lượt này; sau đó chạy đủ 600 frame cho ứng viên `conf=0.15` để đối chiếu bản `conf=0.30`.
- Môi trường thực thi: Conda `cv_robotics_lab21_py310`, Python 3.10, CPU; tận dụng môi trường đã có trên máy.
- Video 1 chọn BoT-SORT theo HOTA trên đoạn có nhãn. Video 2–5 chọn OC-SORT dựa trên quan sát các mục tiêu chính trong đoạn thử, cùng ưu điểm giảm phần tính mạng Re-ID trên CPU. Không có nhãn để khẳng định OC-SORT tốt hơn toàn video.
- Quan sát bằng các frame lấy từ video có vẽ ID. Đây là đánh giá định tính trên mẫu, không phải số lần đổi ID theo ground truth.
- Log và cấu hình thử ở `runs/baseline_manifest.json`, `runs/tuning_manifest.json`; ảnh đối chiếu ở `runs/quan_sat/`.

Ở frame 31 của video 2–5, ngưỡng 0.50 làm mất thêm người phía xa; riêng video 4 mất hộp người áo đỏ. Ngưỡng 0.15 tạo thêm hộp người nhỏ nhưng cũng tăng số hộp/ID cần kiểm tra, đặc biệt ở vùng che khuất và phản chiếu. Giữ 0.30 để cân bằng độ phủ với việc kiểm tra hộp khó phân biệt trên các đoạn đã xem. Hai mức IoU 0.40/0.70 cho các mục tiêu chính tương tự trong mẫu; giữ 0.50, không khẳng định đây là ngưỡng tối ưu khi bốn video không có nhãn.

## 2. Số liệu video_1

### Đối chiếu trên 150 frame đầu

TrackEval chấm cùng 150 frame đầu, với nhãn `video_1` giới hạn đúng đoạn này. Các số dưới đây theo thang phần trăm và chỉ dùng chọn cấu hình thử, không thay bảng đủ 600 frame.

| Tracker | conf | iou | HOTA | MOTA | IDF1 |
|---|---|---|---:|---:|---:|
| ByteTrack | 0.30 | 0.50 | 31.15 | 15.63 | 27.23 |
| BoT-SORT | 0.30 | 0.50 | **32.29** | **17.32** | 30.72 |
| OC-SORT | 0.30 | 0.50 | 29.79 | 16.80 | 30.24 |
| OC-SORT | 0.15 | 0.50 | 31.70 | 14.72 | **33.69** |
| OC-SORT | 0.50 | 0.50 | 26.97 | 13.81 | 24.53 |
| OC-SORT | 0.30 | 0.40 | 30.08 | 17.00 | 30.98 |
| OC-SORT | 0.30 | 0.70 | 30.52 | 16.43 | 28.66 |

BoT-SORT có HOTA và MOTA cao nhất trong các lượt này, nên được chọn để rà ngưỡng tiếp. OC-SORT ở `conf=0.15` có IDF1 cao hơn nhưng MOTA thấp hơn; không chọn chỉ dựa trên việc đầu ra có nhiều hộp hoặc nhiều ID.

### Rà ngưỡng BoT-SORT trên 60 frame đầu

Cùng đoạn 60 frame và nhãn tương ứng; không so trực tiếp số trong bảng này với bảng 150 frame.

| conf | iou | HOTA | MOTA | IDF1 |
|---|---|---:|---:|---:|
| 0.30 | 0.50 | 34.37 | 17.84 | 32.83 |
| 0.15 | 0.50 | **35.76** | 19.13 | 35.92 |
| 0.50 | 0.50 | 32.94 | 15.56 | 27.73 |
| 0.30 | 0.40 | 34.56 | 17.84 | 32.96 |
| 0.30 | 0.70 | 35.36 | **20.34** | **36.17** |

Ứng viên `conf=0.15, iou=0.50` có HOTA cao nhất trên đoạn này và được chạy lại đủ 600 frame. Log cấu hình và điểm: `runs/botsort_tuning_manifest.json`, `runs/botsort_tuning_scores.json`.

### Đối chiếu hai bản đủ 600 frame

| BoT-SORT | HOTA | MOTA | IDF1 |
|---|---:|---:|---:|
| conf=0.30, iou=0.50 — bản chọn | **29.460** | 19.811 | 29.354 |
| conf=0.15, iou=0.50 — bản loại | 29.343 | **20.731** | **29.561** |

Giữ `conf=0.30` theo HOTA quan sát trên toàn video. Hạ ngưỡng cải thiện MOTA và IDF1 nhưng HOTA giảm nhẹ; chênh lệch nhỏ và không chứng minh cấu hình nào vượt trội một cách ổn định. Kết quả ngắn 60 frame không thay thế được việc kiểm tra toàn video.

### Kết quả đủ 600 frame

Bảng do `scripts/evaluate_practice.py` chấm cho `runs/nop_bai/video_1.txt`:

```
HOTA: Susois_video1-pedestrian     HOTA      DetA      AssA      DetRe     DetPr
video_1                            29.46     18.095    48.223    18.596    78.888

CLEAR: Susois_video1-pedestrian    MOTA      MOTP      CLR_TP    CLR_FN    CLR_FP    IDSW
video_1                            19.811    81.471    4043      14538     337       25

Identity: Susois_video1-pedestrian IDF1      IDR       IDP       IDTP      IDFN      IDFP
video_1                            29.354    18.137    76.941    3370      15211     1010

Count: Susois_video1-pedestrian    Dets      GT_Dets   IDs       GT_IDs
video_1                            4380      18581     52        62
```

`video_2` đến `video_5` không có nhãn trong gói lab. Không điền số cho các video đó.

Kết quả toàn video: **HOTA 29.46 / MOTA 19.811 / IDF1 29.354**. Bỏ sót là hạn chế lớn nhất: 14.538 FN so với 337 FP và 25 lần đổi ID theo CLEAR. Kết quả này chưa thể gọi là theo dõi tốt toàn cảnh; việc chọn BoT-SORT dựa trên các cấu hình đã thử, không phải chứng minh đây là cấu hình tối ưu.

Log chấm đầy đủ: `runs/evaluation_video1.log`. Lệnh chấm đã kết thúc với mã thoát 0, nhãn dùng đủ 600 frame. Gói dữ liệu không có `eval_config.json`; cấu hình chấm trong bản dữ liệu cục bộ được bổ sung từ metadata của `video_1`, không chỉnh sửa nhãn.

## 3. Phân tích

### Video 1 — quảng trường, camera tĩnh

Trong các frame mẫu, ByteTrack theo được nhóm người gần camera nhưng bỏ qua thêm người phía xa so với BoT-SORT. OC-SORT ở cùng ngưỡng cũng giữ được nhiều hộp hơn ByteTrack, nên được đưa vào lượt thử ngưỡng. Khi hạ `conf` của OC-SORT từ 0.30 xuống 0.15, tổng hộp trong 150 frame tăng từ 841 lên 1756 và số ID xuất hiện tăng từ 23 lên 51; hai số này chỉ mô tả đầu ra, không chứng minh tracker tốt hơn. Cảnh tĩnh thuận lợi cho mô hình chuyển động, nhưng người giao nhau và che khuất vẫn khiến thông tin ngoại hình có ích. Kết luận định lượng dùng các bảng TrackEval trong mục 2, không dùng số hộp thay cho HOTA.

### Video 3 — camera di động, ảnh nhỏ

Người áo sọc giữ ID 1 với OC-SORT tại frame 11/31/51/101 và còn ID 1 ở frame 559; BoT-SORT cũng giữ được người này trong mẫu chung ban đầu. Camera tiến gần làm hộp thay đổi kích thước đáng kể, nhưng người này vẫn có hộp lớn và được detector quan sát liên tiếp. Chọn OC-SORT vì giữ được mục tiêu đã xem và giảm phần tính ngoại hình trên CPU. Tuy nhiên, người áo xám tiền cảnh có dấu hiệu đổi từ ID 2 ở frame 1 sang ID 14 ở frame 51, nên không coi kết quả là ổn định cho mọi người. Người rất nhỏ và bị che vẫn phụ thuộc vào detector cố định.

### Video 4 — trong nhà và phản chiếu

OC-SORT giữ ID 2 cho người áo đỏ đến frame 101, trong khi ByteTrack đổi ID giữa frame 51 và 101 của baseline. Chọn OC-SORT dựa trên đoạn chung đã xem và ưu điểm giảm phần tính ngoại hình trên CPU. Các frame muộn hơn có dấu hiệu đổi ID người áo đỏ; camera chuyển động và các khoảng thiếu ảnh làm việc theo dõi dài hạn khó hơn, nhưng chưa có nhãn để tách nguyên nhân. Các hộp trong vùng kính bên trái cần xem kỹ để phân biệt người thật với phản chiếu. Do thiếu 31 ảnh, kết quả video này chưa đáp ứng yêu cầu đủ frame.

### Video 5 — camera trên xe bus

Người áo đỏ bên phải giữ ID 9 với OC-SORT đến frame 101; BoT-SORT cũng giữ ổn định người này trong baseline. Khi xe tiến tới và rẽ, nền cùng vị trí người dịch chuyển đáng kể, như các mẫu frame 501 và 750. Chọn OC-SORT vì mục tiêu đầu video được giữ liên tục và tracker giảm phần tính ngoại hình trên CPU. Các người nhỏ ở xa vẫn có trường hợp chưa có hộp, nên không dùng việc giữ ID ở người áo đỏ để kết luận toàn video đã theo dõi tốt.

## 4. Nếu có thêm thời gian

Ưu tiên khôi phục đủ 31 ảnh của `video_4`, rồi chạy lại toàn bộ video này. Sau đó xem kỹ những đoạn người giao nhau hoặc đi vào vùng tối, và quét `conf` mịn hơn quanh cấu hình đã chọn, giữ nguyên detector và Re-ID theo luật lab.

## 5. Kiểm tra và tệp bàn giao

- Năm tệp MOT ở `runs/nop_bai/video_1.txt` đến `video_5.txt` đã được kiểm tra định dạng 10 cột, số frame gốc và ID không trùng trong cùng frame.
- Video xem trước ở `runs/nop_bai/` đã được kiểm tra số frame và đọc các frame mẫu đầu/giữa/cuối.
- Notebook `on_tap_metrics.ipynb` đã chạy đủ các ô mã, trả lời đúng phần ôn tập và có kết quả detector trên ảnh thực tế.
- Bộ kiểm thử: **7 passed**. Kiểm tra `git diff --check` đạt.
- Gói `runs/submission_Susois_2A202602853.zip` chứa năm tệp MOT, báo cáo và `manifest.json`; không chứa dữ liệu ảnh, nhãn hoặc trọng số mô hình. Manifest ghi rõ frame thiếu của `video_4`.

Gói hiện tại cần bổ sung dữ liệu và chạy lại `video_4` trước khi coi là bài nộp đủ frame cho cả năm video.
