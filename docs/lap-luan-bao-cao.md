# Bảng lập luận báo cáo, lượt A

Theo mục 1 của `docs/van-phong-tieng-viet.md`. Mỗi dòng là một đoạn dự kiến. Chưa
viết câu văn nào ở bước này.

Duyệt và sửa bảng trước khi sang lượt B. Hai chỗ cần bạn tự đặt tay vào: **cột
Kết luận**, vì đó là chỗ hội đồng hỏi thẳng, và **cột Điều kiện đảo chiều**, vì
đó là chỗ dễ bịa nhất nếu để tôi tự điền.

Dòng nào cột Cơ chế còn trống là dòng chưa đủ nguyên liệu, phải chạy thêm thí
nghiệm chứ không viết đoạn mô tả bảng để lấp chỗ.

Ký hiệu: `[G]` số liệu đã có trong `results/`, `[13.x]` dẫn tới mục ghi chép
tương ứng trong `KE_HOACH_TRIEN_KHAI.md`, `[?]` chưa có số.

---

## Chương 1. Bài toán và ba hằng số

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Bài toán Ridge cho phép đo sai số tuyệt đối chứ không chỉ so sánh tương đối giữa các thuật toán | $f^* = 6{,}151114$, $\lVert\nabla f(w^*)\rVert = 7{,}4\cdot10^{-13}$ [G] | Hàm bậc hai lồi mạnh có nghiệm đóng, giải bằng Cholesky | Mất hiệu lực nếu đổi sang Lasso hoặc Huber, khi đó $f^*$ phải ước lượng bằng lần chạy dài nhất |
| Ba hằng số đo được, không phải giả định | $L = 9{,}1156$, $\mu = 0{,}034073$, $\kappa = 267{,}5$ [G] | $L$ và $\mu$ là hai đầu phổ của $\frac1n X^\top X$ cộng $\lambda$ | $\kappa$ đổi khi đổi $\lambda$, xem chương 10 |
| Hessian hằng số là lý do Newton kết thúc sau một vòng | $\nabla^2 f = \frac1n X^\top X + \lambda I$ không chứa $w$ | Khai triển Taylor bậc hai không còn số hạng dư | Sai ngay khi hàm mục tiêu không còn bậc hai |
| Vẽ $f - f^*$ bằng dạng toàn phương thay vì phép trừ hạ sàn số học tám bậc | tại $\lVert w-w^*\rVert \approx 10^{-12}$: phép trừ cho $-1{,}4\cdot10^{-17}$, dạng toàn phương cho $3{,}7\cdot10^{-24}$ [13.5] | $f(w)-f^* = \frac12 (w-w^*)^\top H (w-w^*)$ đúng chính xác vì gradient triệt tiêu tại $w^*$ | Chỉ đúng cho hàm bậc hai |

## Chương 2. Dữ liệu: rò rỉ và chuẩn hóa

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Giữ `sub_grade` thì bài toán hồi quy mất ý nghĩa | $R^2 = 0{,}9554$ toàn bộ, $0{,}9834$ trong riêng năm 2016; sau khi loại còn $\approx 0{,}486$ [13.9] | Lending Club ấn định lãi suất từ bảng tra theo `sub_grade` | Nếu đề bài chỉ quan tâm tối ưu hóa thì rò rỉ không làm sai phép tính nào, chỉ làm hỏng phần đối chiếu RMSE |
| `fico_range_high` tạo một hướng suy biến chính xác | trị riêng $1{,}07\cdot10^{-7}$, vector riêng có đúng hai hệ số $\pm0{,}7071$; quan hệ đúng ở 2 260 227 trên 2 260 668 dòng [13.10] | Hai cột lệch nhau đúng hằng số 4 nên trùng nhau sau chuẩn hóa | Nếu giữ lại thì $\mu = \lambda$ và $\kappa = L/\lambda$, làm chương 10 mất nội dung |
| Công của chuẩn hóa nằm ở phép chia, không ở phép trừ | $\kappa$: thô $3{,}87\cdot10^{12}$, chỉ trừ trung bình $1{,}91\cdot10^{12}$, chuẩn hóa đầy đủ $267{,}5$ [13.17] | Các cột đứng ở thang khác nhau tới sáu bậc, trừ trung bình không đụng tới chênh lệch đó | Nếu mọi cột vốn cùng thang thì chuẩn hóa không còn tác dụng |
| Quy tắc một sai số chuẩn đổi rất ít chất lượng lấy rất nhiều điều kiện | $\lambda$ 0,001778 lên 0,031623 (18 lần), sai số CV 12,0430 lên 12,0607 (0,15%) [13.8] | Đường cong CV phẳng trên nhiều bậc $\lambda$, nên cực tiểu là lựa chọn tùy tiện | Đảo chiều nếu đường cong CV có cực tiểu nhọn |

## Chương 3. Cài đặt và kiểm thử

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Tách hai trục làm tám cấu hình bắt buộc thành thứ kiểm đếm được | 8 = 4 hướng $\times$ 2 quy tắc bước, chạy hết trong phép kiểm thứ 5 [mục 8] | `Direction` và `StepRule` là hai loại đối tượng rời, `iterate` là vòng lặp duy nhất | Không áp dụng cho Adam, vì Adam không có độ dài bước vô hướng [4.2.1] |
| Điều kiện Armijo phải viết dạng tổng quát | $\nabla f(y)^\top p < 0$ với hướng Newton, còn $-\lVert\nabla f\rVert^2$ chỉ đúng khi $p = -\nabla f$ | Line search nhận hướng từ ngoài nên không được giả định hướng nào | Không đảo chiều |
| Gradient và Hessian đúng | sai số tương đối $1{,}2\cdot10^{-9}$ và $1{,}8\cdot10^{-9}$ so với sai phân trung tâm [G] | Sai phân trung tâm có bậc $\varepsilon^2$ | Không đảo chiều |
| Momentum ghép với backtracking phải tính lại theo bước thật | $\beta$ lấy từ `accept` chứ không từ $\kappa$ [13.6] | Công thức $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ chỉ đúng khi $t = 1/L$ | Với bước cố định $1/L$ thì hai cách trùng nhau |

## Chương 4. Chọn độ dài bước thế nào

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Ngưỡng $2/L$ là ngưỡng thật, quan sát được | $t = 2{,}1/L$ phân kỳ ở vòng 94; $t = 2/L$ đình trệ ở $9{,}4\cdot10^{-2}$; $t = 1{,}9/L$ hội tụ [G] | $\lvert 1 - tL\rvert > 1$ làm hướng dốc nhất nở ra mỗi vòng | Không đảo chiều |
| Bước tối ưu theo lý thuyết thua $1{,}9/L$ trên bài này | 766 vòng so với 331 vòng để đạt $10^{-6}$ [13.14] | Sai số ban đầu dồn vào hướng trị riêng lớn: $\lambda>1$ mang 80,5%, $\lambda<0{,}1$ mang 0,3%; $\lvert1-tL\rvert$ là 0,9000 so với 0,9926 | Đảo chiều khi cần độ chính xác rất cao, lúc đó hướng $\mu$ quyết định và $2/(L+\mu)$ nhỉnh hơn |
| Mô hình modal giải thích được con số, không chỉ hướng | dự đoán $9{,}80\cdot10^{-7}$ tại vòng 331 và $9{,}95\cdot10^{-7}$ tại vòng 766 [13.14] | $f-f^* = \frac12\sum_i \lambda_i c_i^2 (1-t\lambda_i)^{2k}$ | Chỉ dùng được vì biết toàn bộ phổ |
| Dò bước bằng tay tốn ba bậc để trúng vùng dùng được | $t = 0{,}1$ hội tụ 691 vòng, $t = 0{,}01$ không đạt sau 5000, $t = 1$ phân kỳ sau 5 vòng [G] | Vùng hội tụ là $(0, 2/L)$ với $2/L = 0{,}219$ | Nếu lưới dò đủ dày thì tìm được, nhưng phải trả bằng số lần chạy |
| Backtracking thắng bước cố định trên cả hai trục | 143 vòng và 3,14 giây so với 331 vòng và 4,75 giây, dù tốn 4,9 lần đánh giá hàm mỗi vòng [13.15] | Armijo so với thương Rayleigh tại chỗ chứ không với $L$ toàn cục, nên nhận bước lớn hơn $2/L$ ở hướng phẳng | Đảo chiều khi mỗi lần đánh giá $f$ đắt ngang một gradient và $\kappa$ nhỏ |
| $\rho$ quyết định chi phí, $c$ gần như không | sáu cấu hình $\rho=0{,}5$ đứng trên toàn bộ mười hai cấu hình $\rho=0{,}8$ và $0{,}9$; $c$ đổi 3000 lần chỉ làm 167 xuống 143 vòng [13.15] | $\rho$ đặt độ thô của lưới bước thử, mỗi lần thử tốn một lần tính $f$ | [?] chưa thử $\rho < 0{,}5$ |
| Phần đuôi của line search là chi phí chết | số lần đánh giá hàm mỗi vòng vọt lên khi $f-f^*$ chạm sàn phân giải [13.7], hình `step-armijo_cost` | Mức giảm thật nhỏ hơn nhiễu làm tròn nên mọi bước thử đều trượt điều kiện | Tiêu chí dừng theo đình trệ cắt được phần này |

## Chương 5. Quán tính: đổi bộ nhớ lấy số vòng lặp

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Quán tính đổi một vector bộ nhớ lấy mười lần số vòng lặp | AGD 286 vòng và 4,01 giây so với GD 3000 vòng và 42,13 giây, cùng bước $1/L$ [G] | Giá mỗi vòng như nhau vì cùng một lần tính gradient; chỉ số vòng đổi | Đảo chiều nếu $\kappa$ nhỏ, khi đó $\sqrt\kappa \approx \kappa$ |
| Hằng số 0,9 mà đề bài gợi ý gần đúng ở đây do trùng hợp | 314 vòng so với 286 vòng; $(\sqrt\kappa-1)/(\sqrt\kappa+1) = 0{,}885$ với $\kappa = 267{,}5$ [13.12] | Công thức momentum không nhạy quanh cực trị | Sai rõ khi $\kappa$ lệch xa 361; phải phát biểu theo điều kiện |
| Công thức tăng dần không hợp với hàm lồi mạnh | $\beta_k = (k-1)/(k+2)$ đình trệ ở $2{,}1\cdot10^{-11}$ sau 894 vòng [G] | Momentum tiến tới 1 nên vượt đà, sinh dao động tuần hoàn | Công thức này dành cho trường hợp không giả định biết $\mu$ |
| Restart là cơ chế bù cho việc không biết $\mu$ | với $\beta_k$ tăng dần: 143 xuống 89 vòng, đình trệ thành hội tụ; với $\beta$ theo $\mu$: trùng từng chữ số [13.16] | Điều kiện $\nabla f(y_k)^\top(w_{k+1}-w_k) > 0$ không kích hoạt lần nào khi $\beta$ đã đặt đúng | Nếu không biết $\mu$ thì restart là bắt buộc |

## Chương 6. Ngẫu nhiên hóa: đổi độ chính xác lấy giá mỗi vòng

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| SGD bước hằng không hội tụ về $w^*$, và đó là kết quả đúng | mọi $B$ dừng quanh $5\cdot10^{-4}$ sau 40 epoch [G] | Dao động trong lân cận bán kính $\mathcal O(\eta\sigma^2/\mu)$ | Bước giảm dần phá được sàn, nhưng chậm hơn trong 40 epoch đầu |
| Kích thước lô đổi giá mỗi vòng chứ không đổi mức sàn | $B=32$: 250 000 vòng và 4,16 giây; $B=2048$: 3880 vòng và 2,17 giây; sàn như nhau [G] | Bước đặt theo $1/L_B$ nên phương sai gradient trên mỗi bước bù trừ với số bước | Đảo chiều nếu dùng chung một bước cho mọi $B$ |
| Một bước an toàn cho gradient đầy đủ làm mọi kích thước lô phân kỳ | $\eta = 1/L$ phân kỳ ở cả $B = 32$, 256 và 2048 [13.11] | $L_{\max} = 200\,056$ so với $L = 9{,}12$, gấp 21 947 lần | Không có $B$ nào trong lưới cứu được; hình không có bậc chuyển tiếp |
| $L_{\max}$ bị một dòng duy nhất kéo lên | trung vị $\lVert x_i\rVert^2 = 82{,}2$, phân vị 99 là 560,9, cực đại 200 056; tọa độ lớn nhất là `addr_state=other` ở 447 độ lệch chuẩn [13.11] | Chuẩn hóa một cột chỉ dấu hiếm biến mỗi giá trị 1 thành đỉnh nhọn | Gộp hoặc bỏ mức hiếm sẽ hạ $L_{\max}$; chưa thử [?] |
| Cận lý thuyết vẫn đáng tính dù bị ngoại lai kéo lệch | $1/L_B = 0{,}001265$ so với $\eta_0 = 10^{-3}$ tốt nhất đo được; $3\cdot10^{-3}$ đã hỏng, $10^{-2}$ phân kỳ [13.11] | Cận là chặn trên an toàn, và ở đây nó chặt | Đảo chiều nếu ngoại lai nặng hơn nữa |
| Quy tắc giảm dần thắng bước hằng ba bậc, nhưng chỉ khi tốc độ giảm đặt theo $\mu$ | ở 200 epoch: staircase $8{,}5\cdot10^{-7}$, inverse với $\gamma=\mu\eta_0$ $6{,}9\cdot10^{-6}$, constant $1{,}1\cdot10^{-3}$ [G] | Bước hằng dừng ở sàn nhiễu $\mathcal O(\eta\sigma^2/\mu)$, còn bước giảm dần đưa sàn đó về 0 | Đảo chiều nếu ngân sách ngắn: ở 40 epoch bước hằng vẫn đang dẫn |
| Chọn tốc độ giảm bằng mắt cho ra kết luận ngược | $\gamma = 1/\text{spe}$ cho $1{,}46\cdot10^{-2}$, $\gamma = \mu\eta_0$ cho $6{,}9\cdot10^{-6}$; hai giá trị lệch nhau 30 lần | $\gamma$ quá lớn làm $\eta_k$ tụt xuống $3\cdot10^{-6}$ và dãy đóng băng trên sàn của bước hằng | Không đảo chiều: đây là lý do phải neo $\gamma$ vào $\mu$ chứ không vào số vòng mỗi epoch |
| Bước hằng có phương sai giữa các seed lớn hơn hẳn | constant trải từ $2{,}59\cdot10^{-4}$ tới $9{,}64\cdot10^{-4}$ (3,7 lần); ba quy tắc kia gần trùng nhau [G] | Điểm dừng của bước hằng nằm trong lân cận ngẫu nhiên, còn quy tắc giảm dần đã co về gần tất định | Không đảo chiều |

## Chương 7. Thông tin bậc hai: đổi giá mỗi vòng lấy số vòng lặp

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Newton kết thúc sau đúng một vòng, sai số bằng 0 | $f - f^* = 0$ đúng bằng 0 trong `float64` sau một vòng [G] | Bước Newton từ $w_0 = 0$ chính là nghiệm đóng | Chỉ đúng cho hàm bậc hai; với Huber thì cần nhiều vòng |
| Backtracking không có việc gì làm trên hàm bậc hai | Armijo nhận bước đầy đủ ở lần thử thứ nhất, `fevals[1] = 1` [G] | Xấp xỉ bậc hai trùng khít hàm thật nên điều kiện giảm thỏa ngay | Đây là lý do mục 4.4 đề xuất bài toán phụ Huber |
| Chi phí một bước Newton là $\mathcal O(nd^2)$, và $d^3$ không đáng kể | dựng $X^\top X$ 163,2 ms, Cholesky $116\times116$ 0,2 ms, một gradient 87,3 ms [13.20] | $d = 116$ nên $d^3$ nhỏ so với $nd^2$ với $n = 1{,}2$ triệu | Đảo chiều khi $d$ lên tới vài nghìn |
| Con số 0,16 giây trong bảng là thiếu, số thật là 0,25 giây | biến thể dựng lại Hessian đo 0,40 giây; `warm_up` trả trước phần dựng Hessian cho biến thể dùng lại [13.20] | `Direction.propose` gói gradient chung với phần đắt tiền, nên lần gọi cuối để kiểm tra điều kiện dừng kéo theo một lần dựng Hessian thừa | Giới hạn thiết kế của `iterate`, chỉ ảnh hưởng Newton |

## Chương 8. So sánh tổng hợp trên hai trục

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Thứ hạng theo hai trục giống nhau ở bài này | Newton 0,03 s, AGD 1,05 s, Armijo 3,14 s, GD 4,75 s; thứ tự theo vòng lặp cũng vậy [13.13] | Giá mỗi vòng của bốn phương pháp chênh nhau ít hơn số vòng chênh nhau | Đảo chiều nếu $d$ lớn: khi đó Newton đắt lên theo $d^2$ còn GD theo $d$ |
| Số vòng lặp không phụ thuộc $n$, thời gian tỷ lệ tuyến tính với $n$ | GD 2022 xuống 1959 vòng, AGD 286 xuống 283, khi $n$ tăng 6 lần; thời gian tăng 5,2 tới 6,5 lần [13.19] | $\kappa$ chỉ đổi từ 267,5 xuống 266,2 nhờ hệ số $\frac1{2n}$ trong hàm mục tiêu | Đảo chiều nếu bỏ hệ số $\frac1n$ |
| Quét lưới trên mẫu nhỏ là hợp lệ | ba hằng số lệch dưới 0,6% giữa hai quy mô [13.8] | Cùng phân phối, cùng cách mã hóa | Phải kiểm lại nếu lấy mẫu theo thời gian thay vì ngẫu nhiên |
| Tốc độ quan sát khớp cận lý thuyết | GD: tỷ lệ 2,73 so với 2,73 và 2,98 so với 2,98; AGD bám $\sqrt\kappa$ [13.18] | GD co theo $(\kappa-1)/(\kappa+1)$, AGD theo $1 - 1/\sqrt\kappa$ | AGD lệch nhẹ vì hằng số trước lũy thừa không phải hằng số theo $\kappa$ |
| SGD nhanh nhất trên mỗi đơn vị thời gian nhưng không tới đích | 13,17 giây cho 40 epoch ở quy mô toàn phần, dừng ở $6{,}0\cdot10^{-4}$ [G] | Sàn nhiễu, xem chương 6 | Nếu chỉ cần độ chính xác $10^{-3}$ thì SGD thắng tuyệt đối |

## Chương 9. So sánh với scikit-learn

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Hai thư viện cần hai hằng số quy đổi khác nhau | `Ridge` cần $\alpha = \lambda n = 3{,}795\cdot10^4$; `SGDRegressor` cần $\alpha = \lambda$ | `Ridge` bỏ cả $\frac1n$ lẫn $\frac12$, `SGDRegressor` giữ cả hai | Sai hằng số không sinh lỗi, chỉ lặng lẽ so hai bài toán khác nhau |
| Quy đổi kiểm chứng được bằng số, không tin suy luận trên giấy | $\lVert w_{\text{sklearn}} - w^*\rVert/\lVert w^*\rVert = 2{,}35\cdot10^{-15}$ [G] | Cắm nghiệm sklearn vào hàm $f$ của nhóm và so với $f^*$ | Không đảo chiều |
| Mã tự viết và thư viện làm cùng một việc, mất cùng thời gian | `Ridge(cholesky)` 0,273 giây; Newton của nhóm khoảng 0,25 giây phần tính thật [13.20] | Cả hai đều là nghiệm đóng qua Cholesky | Trùng khớp là dấu hiệu cả hai cài đúng, không phải bên nào thắng |
| Solver lặp của thư viện đổi độ chính xác lấy thời gian | `lsqr` 2,03 s sai số $1{,}13\cdot10^{-5}$; `sag` 45,4 s sai số $3{,}96\cdot10^{-7}$ [G] | Dừng theo ngưỡng tương đối chứ không giải chính xác | Có lợi khi $d$ lớn tới mức $\mathcal O(d^3)$ không chấp nhận được |
| Tham số mặc định của thư viện không phải lúc nào cũng dùng được | `SGDRegressor` sai số $6{,}2\cdot10^{20}$, RMSE $3{,}3\cdot10^{10}$, dừng sau 6 vòng [G] | Bước mặc định không đặt theo $L$ của bài toán | Chỉnh tham số thì chạy được; đây là phát biểu về mặc định, không phải về thuật toán |

## Chương 10. Ảnh hưởng của hệ số hiệu chỉnh

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| $\lambda$ có hai vai trò tách biệt, thống kê và tối ưu hóa | $\kappa$ từ 2633 xuống 1,9 khi $\lambda$ từ 0,001 lên 10; RMSE từ 3,4537 lên 4,4578 [13.18] | $\mu = \lambda_{\min} + \lambda$ nên $\lambda$ đặt sàn cho phổ | Nếu $\lambda_{\min}$ của dữ liệu lớn hơn hẳn $\lambda$ thì vai trò thứ hai biến mất |
| Khoảng $\lambda$ mua được điều kiện gần như miễn phí | 0,001 lên 0,1: RMSE xấu 0,44%, $\kappa$ giảm 29 lần, GD từ không hội tụ xuống 257 vòng [13.18] | Đường cong CV phẳng trên khoảng đó | Ngoài khoảng đó thì RMSE xấu nhanh: $\lambda = 10$ cho RMSE 4,4578 |
| Giá trị mà quy tắc một sai số chuẩn chọn nằm đúng trong khoảng đó | $\lambda = 0{,}031623$, $\kappa = 267{,}5$ [13.8] | Quy tắc chọn $\lambda$ lớn nhất còn nằm trong một sai số chuẩn | Không đảo chiều |

## Chương 11. Kết luận

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |
| Tính $L$ và $\mu$ rẻ hơn dò bước rất nhiều | phổ trị riêng của ma trận $116\times116$ mất dưới một giây; lưới dò 5 giá trị mất 195 giây [G] | $d \ll n$ nên phổ tính được trực tiếp | Đảo chiều khi $d$ lớn tới mức không dựng nổi $X^\top X$ |
| Với $d$ nhỏ và $n$ lớn, phương pháp bậc hai thắng | Newton 0,25 giây so với GD 158,67 giây ở quy mô toàn phần [13.19] | Chi phí Hessian $\mathcal O(nd^2)$ chỉ gấp $d$ lần một gradient | Đảo chiều khi $d$ vài nghìn, hoặc khi dữ liệu không nạp hết vào bộ nhớ |
| Ba điều rút ra về chọn tham số | $\rho$ quan trọng hơn $c$; $\eta_0$ theo $1/L_B$; $\beta$ theo bước thật | Xem chương 4, 5, 6 | |
| Ba chỗ thực nghiệm lệch khỏi lý thuyết | $1{,}9/L$ thắng $2/(L+\mu)$; backtracking thắng cả hai trục; bước hằng thắng trong 40 epoch | Cận lý thuyết là cận xấu nhất, còn thực nghiệm chạy trên một điểm khởi tạo cụ thể và một ngân sách hữu hạn | |

---

## Việc phải làm trước lượt B

1. ~~Chương 6 thiếu số~~ **đã giải quyết.** Chạy tới 400 epoch cho thấy kết
   luận ban đầu là sai và sai vì lý do của chính tôi: $\gamma = 1/\text{spe}$
   đặt tùy tiện làm mọi quy tắc giảm dần đóng băng. `group_batch_schedule` đã
   sửa để dùng $\gamma = \mu\eta_0$, giữ giá trị cũ làm điểm đối chứng, và ngân
   sách nâng từ 40 lên 200 epoch.
2. **Hai dòng đánh dấu `[?]`**: chưa thử $\rho < 0{,}5$; chưa thử gộp mức hiếm để
   hạ $L_{\max}$. Cả hai đều bỏ được, nhưng phải bỏ có chủ ý chứ không lờ đi.
3. **Cột Kết luận và cột Điều kiện đảo chiều cần bạn duyệt.** Tôi điền theo số
   liệu đo được, nhưng chỗ nào bạn thấy phát biểu mạnh hơn số liệu cho phép thì
   sửa trước khi tôi viết thành câu.
