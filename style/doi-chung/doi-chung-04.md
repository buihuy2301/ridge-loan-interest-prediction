# doi-chung-04

- Nguồn: `report/report.tex`, mục 2.3 Chọn hệ số hiệu chỉnh, đoạn 3
- Chức năng: nêu đánh đổi của một quyết định
- Kích thước hiện tại: 2 câu, 69 tiếng, câu dài nhất 36 tiếng
- Ngày ghi: 2026-08-10

## Đoạn hiện tại

> Về mặt thống kê, cái giá phải trả không đáng kể: RMSE trên tập kiểm tra là
> \num{0.20601} tại $\lambda = \num{0.01}$ so với \num{0.20579} tại điểm cực tiểu
> cross-validation, kém đi 0{,}1\%. Đổi lại, $\kappa$ giảm đúng 100 lần theo tỉ lệ
> giữa hai giá trị $\lambda$, và toàn bộ các thuật toán bậc một trở nên so sánh
> được với nhau trong thời gian chạy hợp lý.

## Lỗi từng câu

**Câu 1.** Đạt. "cái giá phải trả không đáng kể" đã tránh lối hệ từ tiếng Anh
"cái giá phải trả là nhỏ" mà mục 6 nhóm 3 cấm.

**Câu 1.** Dấu hai chấm dẫn vào phần định lượng cho mệnh đề trước: hợp lệ. Số
viết trong `\num{}` nên in ra dấu phẩy thập phân theo `\sisetup` ở
`preamble.tex`, không vi phạm quy tắc dấu thập phân.

**Câu 1.** "kém đi 0,1%" là số làm tròn xuống từ 0,107%. Đúng nhưng nên kiểm lại
mỗi lần sinh lại, vì đây là con số dễ trôi khi hai giá trị RMSE thay đổi.

**Câu 2.** Đạt. "Đổi lại" là liên từ tương phản có trong bảng ở mục 5, và đặt
đúng chỗ: nó nối phần mất với phần được.

**Câu 2.** Hai ý nối bằng "và": $\kappa$ giảm 100 lần, và các thuật toán bậc một
so sánh được. Hai ý này cùng một cơ chế nên giữ chung được, không thuộc diện tách
theo mục 5.

**Cả đoạn.** Thiếu điều kiện đảo chiều. Quy tắc một sai số chuẩn được dùng nhưng
không nói nó hỏng ở đâu, chẳng hạn khi đường cong cross-validation có cực tiểu rõ,
hoặc khi mục tiêu là RMSE nhỏ nhất chứ không phải $\kappa$ nhỏ.

**Cả đoạn.** Đây là đoạn thứ ba liên tiếp trong mục 2.3 mở đầu bằng trạng ngữ
hoặc cụm chỉ đối tượng. Mục 5 cấm hai câu liên tiếp cùng khuôn; ở cấp đoạn thì
chưa cấm, nhưng cần kiểm khi sinh lại cả mục.

## Bảng lập luận để sinh lại

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Chọn $\lambda = 0{,}01$ theo quy tắc một sai số chuẩn gần như không mất gì về dự báo | RMSE 0,20601 so với 0,20579, kém 0,1% | Đường cong cross-validation phẳng nên dịch $\lambda$ trong khoảng phẳng không đổi sai số | Nếu đường cong có cực tiểu rõ thì 0,1% này sẽ lớn hơn nhiều |
| Đổi lại, chi phí tối ưu hóa giảm hai bậc | $\kappa$ từ 26 831 xuống 268,3 | $\kappa = L/\lambda$, nên $\kappa$ tỉ lệ nghịch với $\lambda$ | Với thuật toán bậc hai, $\kappa$ không chi phối số vòng lặp nên lợi ích này biến mất |
