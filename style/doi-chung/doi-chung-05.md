# doi-chung-05

- Nguồn: `report/report.tex`, chương 6 So sánh tổng hợp, câu dẫn và đoạn đầu
- Chức năng: mở đầu một chương so sánh, rồi bình luận một bảng số
- Kích thước hiện tại: 4 câu, 76 tiếng, câu dài nhất 38 tiếng
- Ngày ghi: 2026-08-10

## Đoạn hiện tại

> Bảng~\ref{tab:summary} và hai hình~\ref{fig:all-iter}, \ref{fig:all-time} cho
> ba kết luận.
>
> \paragraph{Phương pháp Newton.} Toàn bộ lời giải mất \SI{0.076}{\second}, ít
> hơn 10 lần so với thời gian gradient descent cần chỉ để đạt $10^{-6}$, và ít
> hơn 580 lần so với thời gian nó cần để chạm giới hạn số học. Hàm mục tiêu bậc
> hai nên bước Newton chính là nghiệm đóng, và khoảng cách này không nói được gì
> về các bài toán phi tuyến.

## Lỗi từng câu

**Câu dẫn.** Câu không mang dữ kiện nào, chỉ báo trước số lượng kết luận sắp
viết. Mục 6 nhóm 1 yêu cầu bỏ loại câu này, và mục 4 phần 1 đòi câu đầu là dữ
kiện, không có câu dẫn đứng trước.

**Câu dẫn.** Phép thử của mục 6 nhóm 1: xóa câu đi thì ba `\paragraph` phía sau
vẫn tự nêu đủ ba kết luận, và thứ duy nhất mất là con số "ba", vốn đếm được bằng
mắt. Vậy câu này không mang thông tin.

**Câu dẫn.** Ba nhãn `\ref` dồn vào một câu. Bảng và hai hình vẫn cần được `\ref`
ít nhất một lần theo mục 5 của `CLAUDE.md`, nên khi bỏ câu này phải chuyển các
nhãn đó xuống các đoạn thật sự dùng tới chúng.

**Câu 1 đoạn Newton.** Đạt. Mở bằng số, kèm hai mốc so sánh chứ không để số đứng
trơ. Cả hai mốc kiểm được từ bảng `tab:summary`: 0,773 chia 0,076 bằng 10,2 và
44,32 chia 0,076 bằng 583.

**Câu 2 đoạn Newton.** Đạt. Một câu chứa cả cơ chế và giới hạn, nối bằng "nên"
rồi "và", đúng hình dạng mà mục 4 mô tả.

**Tiêu đề.** `\paragraph{Phương pháp Newton.}` dùng lối danh hóa: đạt theo mục 6
nhóm 2.

**Cả cụm.** Đoạn Newton đủ ba phần của mục 4 và không cần sửa. Lỗi nằm trọn ở câu
dẫn, nên đây là phép thử sạch: bỏ được câu dẫn mà không làm hỏng đoạn sau thì
thay đổi có tác dụng.

## Bảng lập luận để sinh lại

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Newton giải xong bài toán này trước khi các phương pháp bậc một kịp đạt $10^{-6}$ | 0,076 s so với 0,773 s và 44,32 s (bảng `tab:summary`) | Hàm mục tiêu bậc hai nên một bước Newton chính là nghiệm đóng | Kết luận không chuyển sang bài toán phi tuyến, nơi mỗi bước Newton chỉ là một xấp xỉ địa phương |
