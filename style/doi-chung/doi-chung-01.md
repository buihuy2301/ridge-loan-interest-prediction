# doi-chung-01

- Nguồn: `report/report.tex`, mục 1.1 Dữ liệu, đoạn 1
- Chức năng: mô tả thành phần bộ dữ liệu
- Kích thước hiện tại: 2 câu, 101 tiếng, câu dài nhất 73 tiếng
- Ngày ghi: 2026-08-10

## Đoạn hiện tại

> Bộ dữ liệu \emph{Used Phone Price Prediction} trên Kaggle gồm \num{1000000}
> bản ghi và 28 cột, không có giá trị thiếu và không có bản ghi trùng lặp. Trong
> đó 21 cột là định lượng (dung lượng RAM, dung lượng lưu trữ, kích thước màn
> hình, dung lượng pin, tuổi máy, giá gốc, điểm vi xử lý, điểm camera, và các cột
> nhị phân mô tả tình trạng máy), 6 cột là định tính (\texttt{brand} 7 mức,
> \texttt{model} 23 mức, \texttt{os\_type} 2 mức, \texttt{condition} 4 mức,
> \texttt{city\_tier} 3 mức, \texttt{seller\_type} 2 mức), và cột còn lại là biến
> mục tiêu \texttt{resale\_price}.

## Lỗi từng câu

**Câu 1.** Ba ý ngang hàng nối bằng dấu phẩy: quy mô bảng, không có giá trị
thiếu, không có bản ghi trùng lặp. Mục 5 yêu cầu tách từ ba ý ngang hàng trở lên.

**Câu 1.** Hai dữ kiện "không có giá trị thiếu" và "không có bản ghi trùng lặp"
nêu ra rồi bỏ đó. Hệ quả của chúng là bỏ được bước điền khuyết và bước khử trùng
lặp, nhưng hệ quả đó không có trong đoạn, nên câu dừng ở phần dữ kiện của mục 4.

**Câu 2.** Một câu 73 tiếng gồm ba vế ngang hàng, không vế nào là mệnh đề phụ.
Đây đúng dạng mà mục 5 nhắm tới: mốc là số ý ngang hàng, không phải số tiếng.

**Câu 2.** Hai ngoặc đơn liệt kê 8 tên cột định lượng và 6 cột định tính kèm số
mức. Không tên nào trong hai danh sách này được dùng lại ở phần sau của báo cáo,
trừ `original_price` và `screen_size_inches` ở mục 2.1. Nội dung thuộc về một
bảng, không thuộc văn xuôi.

**Cả đoạn.** Không câu nào mang lập luận, tức không câu nào có mệnh đề phụ chỉ
nguyên nhân, điều kiện hoặc tương phản. Mục 5 đòi ít nhất một câu như vậy mỗi
đoạn.

**Cả đoạn.** Không có cơ chế và không có hệ quả, chỉ có dữ kiện. Ba phần của mục
4 khuyết hai.

**Cả đoạn.** Không có `\ref` nào, trong khi đoạn đang mô tả thứ lẽ ra nằm ở bảng.

## Bảng lập luận để sinh lại

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Bộ dữ liệu dùng được ngay, không cần bước làm sạch nào | 1 000 000 bản ghi, 28 cột, 0 giá trị thiếu, 0 bản ghi trùng | Dữ liệu Kaggle đã qua tiền xử lý, nên phần chuẩn bị chỉ còn mã hóa cột định tính và chuẩn hóa | Nếu dùng dữ liệu rao bán thu thập trực tiếp thì hai con số 0 này không còn, và chi phí làm sạch sẽ vượt chi phí tối ưu hóa |
| Kích thước bài toán sau mã hóa là $d = 280$, phần lớn do biến tương tác chứ không do biến định tính | 21 cột định lượng, 6 cột định tính với tổng 41 mức, $d = 280$ (mục 2) | Tích từng cặp của 21 cột định lượng đã sinh 210 cột, gấp năm lần toàn bộ số mức của biến định tính | Nếu `model` có hàng nghìn mức thì phần one-hot mới là phần chi phối $d$ |
