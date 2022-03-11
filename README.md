# Bài tập nhóm giữa kì: Nhận dạng cửa hàng của hoá đơn

## Giới thiệu

Trong phần bài tập này, bạn sẽ cài đặt các thuật toán xử lý ảnh cơ bản
để nhận dạng loại hoá đơn, theo ảnh đầu vào.
Có tất cả 3 loại hoá đơn chính là `highlands`, `phuclong`, `starbucks`,
trong trường hợp không phải 3 loại hoá đơn trên thì trả về là `others`.
Tập ảnh ví dụ (được thu thập từ công cụ tìm kiếm google) được để trong thư mục
`sampledata` với nhãn được để trong tệp `labels.csv`
Các ảnh đầu vào bị ảnh hưởng bởi cái điều kiện ánh sáng, góc chụp và có thể
bị xoay.

Các bạn cần cài đặt thuật toán trong tệp tin `simple_ocr.py` với 2 hàm chính:
+ Hàm khởi tạo:
+ Hàm `find_label`: trả về nhãn hoá đơn của 1 ảnh đầu vào trong 4 loại `others`, `highlands`, `phuclong`, `starbucks`

Các bạn có thể sử dụng `main.py` để chấm điểm.

Các nhóm được khuyến khích sử dụng mô hình học sâu hoặc các phương pháp
xử lí ảnh thông thường để giải bài toán.
Điểm của nhóm sẽ được cho theo:
+ Độ chính xác trên tập kiểm tra (5pts)
+ Báo cáo (5pts) 

## Chấm điểm

- Mỗi nhóm có tối đa không quá 4 người
- Kết quả phần độ chính xác sẽ được kiểm tra theo tập ảnh mới, khác
tập ảnh hiện tại
- Phần báo cáo: cách tiếp cận, kết quả chạy thử, tổ chức chương trình 

## Giới hạn
Việc sử dụng thêm các thư viện ngoài là không được phép.

Thời gian khởi tạo mô hình không quá: 60s, tổng thời gian xử lí không quá: 120s.

Số ảnh được test không quá 30 ảnh, kích thước không quá 2048x2048.

Các bạn chỉ được sử dụng các thư viện đi kèm trong `requirements.txt`.

Mô hình được sử dụng không quá 50MB 
