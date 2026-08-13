# doi-chung-03

- Nguồn: `report/report.tex`, mục 2.3 Chọn hệ số hiệu chỉnh, đoạn 1
- Chức năng: bình luận một hình
- Kích thước hiện tại: 2 câu, 58 tiếng, câu dài nhất 41 tiếng
- Ngày ghi: 2026-08-10

## Đoạn hiện tại

> Đường cong cross-validation 5 fold ở hình~\ref{fig:lambda-cv} phẳng trên một
> khoảng rộng: sai số trung bình giống nhau tới bốn chữ số thập phân với mọi
> $\lambda$ từ $10^{-6}$ đến $10^{-3}$, nên cross-validation không phân biệt được
> các giá trị này. Trong khi đó, theo \eqref{eq:mu-equals-lambda}, số điều kiện
> giữa hai đầu khoảng đó chênh nhau 1000 lần.

## Lỗi từng câu

**Câu 1.** Đạt phần lớn. Mở bằng dữ kiện kèm `\ref`, định lượng ngay sau dấu hai
chấm, và khép bằng hệ quả nối bằng "nên".

**Câu 1.** "phẳng trên một khoảng rộng" là nhận định định tính đứng trước con số
chứng minh nó. Trật tự này chấp nhận được vì mệnh đề sau dấu hai chấm định lượng
đúng cụm đó, nhưng "rộng" thì tới cuối câu mới rõ là ba bậc thập phân.

**Câu 2.** Đạt. "Trong khi đó" là liên từ tương phản có trong bảng ở mục 5, và
`\eqref` dẫn công thức ngay trong câu đúng như mục 4 phần 2 yêu cầu.

**Cả đoạn.** Đoạn khép bằng dữ kiện thứ hai chứ không bằng hệ quả. Mục 4 phần 3
đòi câu cuối là kết luận hoặc giới hạn; ở đây kết luận thật sự, tức phải chọn
$\lambda$ bằng một tiêu chí khác ngoài cross-validation, bị đẩy sang đoạn sau.

**Cả đoạn.** Hai câu, hai cơ chế khác nhau: một là đường cong phẳng, hai là số
điều kiện chênh 1000 lần. Theo mục 4, mốc tách đoạn là số cơ chế, nên đây là chỗ
cần cân nhắc tách hoặc nối hai cơ chế lại bằng một câu hệ quả chung.

## Bảng lập luận để sinh lại

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Cross-validation không chọn được $\lambda$ cho bài toán này | Sai số trung bình giống nhau tới bốn chữ số thập phân với mọi $\lambda \in [10^{-6}, 10^{-3}]$ (hình `fig:lambda-cv`) | Hiệu chỉnh nhỏ hơn nhiễu của dữ liệu thì không đổi được sai số dự báo | Nếu $n$ nhỏ hơn nhiều hoặc $d$ gần $n$, đường cong sẽ có cực tiểu rõ |
| Nhưng $\lambda$ vẫn phải chọn cẩn thận, vì nó quyết định chi phí tối ưu hóa | Hai đầu khoảng chênh nhau 1000 lần về $\kappa$, theo `eq:mu-equals-lambda` | $\lambda_{\min}(\tfrac{1}{n}X^\top X) \approx 0$ nên $\mu = \lambda$ và $\kappa = L/\lambda$ | Nếu ma trận Gram không suy biến thì $\mu$ do dữ liệu quyết định, và $\lambda$ hết vai trò này |
