# doi-chung-02

- Nguồn: `report/report.tex`, mục 1.1 Dữ liệu, đoạn 3
- Chức năng: giải thích một quyết định về thiết kế thí nghiệm
- Kích thước hiện tại: 2 câu, 68 tiếng, câu dài nhất 44 tiếng
- Ngày ghi: 2026-08-10
- Vai trò: mốc chống thụt lùi. Đoạn này đang đạt phần lớn yêu cầu, nên phép thử
  là nó không xấu đi sau khi sửa file văn phong.

## Đoạn hiện tại

> Toàn bộ thí nghiệm chạy trên một mẫu ngẫu nhiên \num{200000} bản ghi, chia
> thành \num{160000} điểm huấn luyện và \num{40000} điểm kiểm tra. Việc lấy mẫu
> là quyết định về chi phí tính toán chứ không phải về thống kê: với toàn bộ một
> triệu bản ghi, một lần tính gradient mất \SI{271}{\milli\second} thay vì
> \SI{39}{\milli\second}, và toàn bộ lưới tham số sẽ mất hàng chục giờ.

## Lỗi từng câu

**Câu 1.** Đạt. Mở bằng dữ kiện, động từ chủ động "chạy trên" thay cho bị động
"được thực hiện trên", ba con số đi liền nhau và cộng lại đúng bằng nhau.

**Câu 2.** Mở đầu bằng danh hóa "Việc lấy mẫu". Mục 3 gọi đúng tên dấu hiệu này:
chuỗi "việc" đứng đầu câu để lấp chỗ chủ ngữ, theo lối tiếng Anh cần một danh ngữ
làm chủ ngữ. Bản nhẹ đặt thẳng chủ thể lên đầu: "Nhóm lấy mẫu vì chi phí tính
toán chứ không vì thống kê".

**Câu 2.** Dùng khuôn tương phản "không phải X mà là Y". Mục 6 nhóm 3 giới hạn
khuôn này ở một lần mỗi mục, nên khi sinh lại cần đếm cả mục 1.1.

**Câu 2.** Dấu hai chấm dẫn vào phần định lượng cho mệnh đề đứng trước: hợp lệ
theo mục 6 nhóm 3.

**Câu 2.** "hàng chục giờ" là con số duy nhất trong đoạn không có mốc so sánh và
không kiểm chứng được từ bảng nào.

**Cả đoạn.** Thiếu điều kiện đảo chiều. Đoạn không nói mẫu 200 000 điểm đủ cho
kết luận nào và hỏng ở kết luận nào, trong khi cỡ mẫu chi phối trực tiếp phần
SGD ở mục 5.

## Bảng lập luận để sinh lại

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Lấy mẫu 200 000 bản ghi là quyết định về chi phí, không phải về thống kê | Một lần tính gradient: 271 ms trên toàn bộ dữ liệu so với 39 ms trên mẫu | Chi phí mỗi lần tính gradient tỉ lệ với $n$, và toàn bộ lưới tham số nhân số đó lên theo tổng số vòng lặp của mọi cấu hình | Kết luận đảo chiều nếu đại lượng cần đo phụ thuộc $n$, chẳng hạn phương sai gradient của SGD |
| Tỉ lệ chia 160 000 / 40 000 không ảnh hưởng tới phần tối ưu hóa | $f^*$ tính theo nghiệm đóng trên tập huấn luyện | Tập kiểm tra chỉ dùng cho RMSE, không vào hàm mục tiêu | Nếu chọn $\lambda$ bằng tập kiểm tra thay vì cross-validation thì tỉ lệ chia sẽ ảnh hưởng |
