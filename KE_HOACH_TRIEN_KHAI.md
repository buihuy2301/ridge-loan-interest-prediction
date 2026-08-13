# Kế hoạch triển khai bài tập môn Tối ưu hóa nâng cao

**Chủ đề:** Cực tiểu hóa hàm mất mát của mô hình hồi quy Ridge dự đoán lãi suất khoản vay, bằng các thuật toán tối ưu hóa bậc một và bậc hai tự cài đặt.

**Dữ liệu:** [Lending Club (2007-2018)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (Kaggle), file `accepted_2007_to_2018Q4.csv.gz`.

**Lớp:** Khoa học dữ liệu.

---

## 1. Định vị bài toán và phạm vi công việc

Đề bài yêu cầu tự cài đặt bốn thuật toán, mỗi thuật toán hai cơ chế chọn bước, rồi so sánh chúng theo hai trục là số vòng lặp và thời gian chạy. Trọng tâm nằm ở phần tối ưu hóa chứ không ở phần mô hình hóa, nên toàn bộ kế hoạch dưới đây xoay quanh một câu hỏi duy nhất: với cùng một hàm mục tiêu cố định, thuật toán nào đưa giá trị hàm về gần cực tiểu nhanh hơn, và câu trả lời có đổi không khi chuyển từ trục vòng lặp sang trục thời gian.

Nguyên tắc kéo theo cần giữ xuyên suốt: **hàm mục tiêu phải được chốt trước khi bắt đầu thí nghiệm tối ưu hóa.** Ma trận thiết kế $X \in \mathbb{R}^{n \times d}$, vector mục tiêu $y \in \mathbb{R}^n$ và hệ số hiệu chỉnh $\lambda$ được cố định ở giai đoạn chuẩn bị dữ liệu rồi không sửa nữa. Hai đường hội tụ chỉ so sánh được với nhau khi chúng cùng đi xuống trên một mặt hàm.

Bài toán Ridge phù hợp với môn học vì hàm mục tiêu lồi mạnh, khả vi vô hạn lần, và **có nghiệm đóng**. Nhờ nghiệm đóng, nhóm tính được giá trị tối ưu $f^*$ chính xác tới sai số máy, rồi vẽ $f(w_k) - f^*$ trên thang logarit. Đề bài chỉ yêu cầu vẽ $f(w_k)$, nhưng vẽ như vậy thì mọi đường đều dẹt xuống một mức nằm ngang sau vài chục vòng lặp và không phân biệt được nữa; vẽ hiệu số trên thang log thì tốc độ hội tụ tuyến tính, tốc độ tăng tốc và hiện tượng bão hòa của SGD hiện ra thành ba dạng đường khác hẳn nhau.

### 1.1. Phát biểu bài toán tối ưu hóa

Ký hiệu $n$ là số điểm dữ liệu, $d$ là số thuộc tính sau khi mã hóa. Bài toán chính:

$$
\min_{w \in \mathbb{R}^d} \quad f(w) \;=\; \frac{1}{2n} \left\| Xw - y \right\|_2^2 \;+\; \frac{\lambda}{2} \left\| w \right\|_2^2
$$

Ba ghi chú thiết kế cần nêu lại trong báo cáo:

- Hệ số chặn $b$ **không** bị phạt, theo quy ước chuẩn, vì phạt $b$ làm nghiệm phụ thuộc vào gốc tọa độ của $y$, tức là đổi đơn vị đo lãi suất thì đổi luôn nghiệm.
- Cách xử lý gọn nhất là chuẩn hóa $X$ theo cột về trung bình $0$, độ lệch chuẩn $1$, đồng thời trừ trung bình khỏi $y$. Khi đó $b^* = 0$ và bài toán rút về một biến $w$ duy nhất, nhờ vậy gradient, Hessian và hằng số Lipschitz đều có dạng sạch.
- Hệ số $\frac{1}{2n}$ thay cho $\frac{1}{2}$ giúp hằng số Lipschitz không phụ thuộc kích thước mẫu. Không có hệ số này thì $L$ tăng tuyến tính theo $n$, và độ dài bước tìm được trên mẫu 200 nghìn điểm sẽ không dùng lại được trên mẫu 1,2 triệu điểm.

Gradient và Hessian:

$$
\nabla f(w) = \frac{1}{n} X^{\top} (Xw - y) + \lambda w
$$

$$
\nabla^2 f(w) = \frac{1}{n} X^{\top} X + \lambda I \qquad \text{(hằng số, không phụ thuộc } w\text{)}
$$

Nghiệm đóng và giá trị tối ưu:

$$
w^* = \left( \frac{1}{n} X^{\top} X + \lambda I \right)^{-1} \left( \frac{1}{n} X^{\top} y \right), \qquad f^* = f(w^*)
$$

Ba hằng số chi phối tốc độ hội tụ:

$$
L = \lambda_{\max}\!\left( \tfrac{1}{n} X^{\top} X \right) + \lambda, \qquad
\mu = \lambda_{\min}\!\left( \tfrac{1}{n} X^{\top} X \right) + \lambda, \qquad
\kappa = \frac{L}{\mu}
$$

trong đó $L$ là hằng số Lipschitz của gradient, $\mu$ là hệ số lồi mạnh, $\kappa$ là số điều kiện. Nhóm phải tính và báo cáo ba số này ngay sau khi chốt dữ liệu, vì chúng là căn cứ chọn mọi độ dài bước ở mục 5.2 và là mốc để đối chiếu tốc độ quan sát được với tốc độ lý thuyết ở mục 5.5.

### 1.2. Ba chỗ kế hoạch này khác với mô tả trong đề bài

Đề bài có ba nhận định cần chỉnh trước khi triển khai, nếu không nhóm sẽ chạy thí nghiệm rồi mới phát hiện kết quả không khớp với thứ mình chờ đợi.

**Newton không cần 5 đến 20 vòng lặp, mà cần đúng một vòng.** Bảng kỳ vọng trong đề bài ghi Newton hội tụ sau 5 đến 20 vòng, con số đúng cho hàm mục tiêu tổng quát. Với hàm bậc hai thì Hessian là hằng số, xấp xỉ bậc hai trùng khít với hàm thật, nên khai triển Taylor không còn số hạng dư. Lấy $w_0 = 0$ ta có ngay

$$
w_1 = w_0 - \left( \nabla^2 f \right)^{-1} \nabla f(w_0) = \left( \tfrac{1}{n} X^{\top} X + \lambda I \right)^{-1} \left( \tfrac{1}{n} X^{\top} y \right) = w^*
$$

tức bước Newton chính là nghiệm đóng. Backtracking cũng vì thế chấp nhận ngay bước đầy đủ ở lần thử thứ nhất. Hai hệ quả: đồ thị hội tụ của Newton chỉ có hai điểm nên không so sánh được theo trục vòng lặp, và phần backtracking cho Newton mất hết nội dung. Mục 4.4 đề xuất một hàm mục tiêu phi tuyến làm bài toán phụ để cứu hai phần này.

**Newton có thể là thuật toán nhanh nhất theo thời gian thực, ngược với dự đoán của đề bài.** Đề bài dự đoán Newton chậm vì phải tính Hessian, và dự đoán đó đúng khi $d$ lớn. Với Lending Club thì $d$ ước tính khoảng 120 đến 150 sau khi mã hóa, nên một vòng Newton tốn $\mathcal{O}(nd^2 + d^3)$, trong đó số hạng $d^3$ hoàn toàn không đáng kể và số hạng $nd^2$ chỉ đắt hơn một vòng gradient descent chừng $d$ lần. Nếu GD cần vài trăm vòng để đạt cùng độ chính xác thì Newton thắng. Nhóm chưa đo nên câu trên mới là dự đoán, và nhóm biểu đồ F ở mục 5.3 xác định chỗ hòa vốn bằng số.

**Momentum $\beta = 0.9$ không phải giá trị tối ưu.** Đề bài đề nghị lấy $\beta \approx 0.9$. Với hàm lồi mạnh, giá trị tối ưu là $\beta = \dfrac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$, và $\beta = 0.9$ chỉ trùng với giá trị này khi $\kappa \approx 361$. Vì $\kappa$ tính được từ dữ liệu nên không có lý do gì để đoán, nhưng vẫn nên giữ $\beta = 0.9$ trong lưới tham số làm một điểm đối chứng, để cho thấy chênh lệch giữa hằng số chọn theo thói quen và hằng số chọn theo lý thuyết.

---

## 2. Giai đoạn 0: Chuẩn bị môi trường

Thư mục làm việc hiện mới có tài liệu và chưa có mã nguồn, nên bước đầu là dựng môi trường ảo.

```bash
cd "/Users/huybq/Documents/work stuff/Optimization_1"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib scipy jupyter pyarrow kaggle pytest
pip freeze > requirements.txt
```

Ghi lại `requirements.txt` không phải hình thức. Kết quả đo thời gian phụ thuộc vào phiên bản numpy và thư viện BLAS phía dưới, nên khi báo cáo số liệu thời gian, nhóm phải ghi kèm cấu hình máy, phiên bản numpy và tên BLAS đang dùng, lấy bằng `numpy.show_config()`.

Gói `pyarrow` có trong danh sách vì file dữ liệu nén khoảng 2,26 triệu dòng và 151 cột; đọc thẳng bằng `pandas.read_csv` tốn vài phút và vài GB bộ nhớ, còn lưu lại dạng Parquet sau lần đọc đầu thì các lần sau chỉ mất vài giây.

Khởi tạo git để theo dõi thay đổi và chia việc:

```bash
git init
```

Về phần soạn thảo, báo cáo viết bằng LaTeX và slide làm bằng Beamer, biên dịch bằng XeLaTeX vì cần font Unicode đủ dấu tiếng Việt. Các thành viên cần cài TeX Live hoặc MacTeX bản đầy đủ. Quy ước chi tiết nằm ở `docs/quy-uoc-bao-cao.md`.

---

## 3. Giai đoạn 1: Chuẩn bị dữ liệu

Mục tiêu của giai đoạn này là tạo ra một cặp $(X, y)$ cố định, đủ lớn về số điểm theo đúng yêu cầu đề bài, rồi lưu lại để mọi thí nghiệm sau dùng chung.

### 3.1. Khảo sát dữ liệu

Bộ dữ liệu gồm hai file. File `accepted_2007_to_2018Q4.csv.gz` chứa khoảng 2,26 triệu khoản vay được duyệt với 151 cột, còn `rejected_2007_to_2018Q4.csv.gz` chứa các hồ sơ bị từ chối và không có cột lãi suất. Chỉ file thứ nhất dùng được, vì biến mục tiêu `int_rate` chỉ tồn tại ở đó.

Cần ghi nhận trong notebook khảo sát:

- Số dòng, số cột thực tế, dung lượng bộ nhớ sau khi đọc.
- Tỷ lệ giá trị thiếu của từng cột. Bộ này có nhiều nhóm cột gần như rỗng hoàn toàn, gồm nhóm hồ sơ đồng vay (`sec_app_*`, `revol_bal_joint`), nhóm hoãn nợ (`hardship_*`) và nhóm tất toán (`settlement_*`), thiếu trên 90 phần trăm.
- Phân phối của `int_rate`, khoảng giá trị và mức độ lệch.
- Kiểu dữ liệu thật của từng cột. Tùy bản phát hành, `int_rate`, `revol_util` và `term` có thể là chuỗi kèm ký hiệu phần trăm hoặc chữ `months`, phải kiểm tra bằng `df.dtypes` chứ không tin mô tả trên Kaggle.

Notebook tham khảo mà đề bài dẫn ra làm bài toán phân loại khả năng vỡ nợ chứ không phải hồi quy lãi suất, nên chỉ dùng lại được phần làm sạch cột, không dùng lại được phần chọn biến mục tiêu.

### 3.2. Rò rỉ thông tin, phần phải xử lý trước tiên

Lending Club ấn định lãi suất theo bậc tín dụng `sub_grade` theo một bảng tra cố định, nên `grade` và `sub_grade` xác định gần như trọn vẹn `int_rate`. Giữ hai cột này thì hồi quy đạt $R^2$ trên 0,99 và bài toán mất hết ý nghĩa thống kê. Cột `installment` cũng phải loại, vì nó được tính ra từ `loan_amnt`, `term` và chính `int_rate` bằng công thức trả góp, tức là chứa thẳng biến mục tiêu.

Ngoài ra cần loại toàn bộ các cột ghi nhận sau khi khoản vay đã giải ngân, gồm `loan_status`, `total_pymnt`, `total_rec_int`, `recoveries`, `out_prncp`, `last_pymnt_*` và các cột cùng nhóm. Chúng là thông tin của tương lai so với thời điểm ấn định lãi suất.

Với riêng phần tối ưu hóa thì rò rỉ không làm sai một phép tính nào: hàm vẫn lồi mạnh, các thuật toán vẫn hội tụ về cùng $w^*$. Nhưng nó làm hỏng phần đối chiếu RMSE ở mục 5.4 và là câu hỏi mà hội đồng chắc chắn sẽ hỏi, nên cách xử lý là loại các cột trên và ghi rõ lý do trong báo cáo. Nếu muốn có thêm một kịch bản đối chiếu, có thể chạy riêng một biến thể giữ `sub_grade` để cho thấy $\kappa$ và $f^*$ đổi ra sao, nhưng không lấy đó làm cấu hình chính.

### 3.3. Xử lý tối thiểu

Giữ ở mức tối thiểu, đúng phạm vi nêu ở mục 7 của `CLAUDE.md`:

1. Loại các dòng thiếu `int_rate`.
2. Loại các cột thiếu quá 30 phần trăm giá trị. Ngưỡng này cắt gọn ba nhóm cột gần rỗng đã nêu ở mục 3.1.
3. Chuyển các cột chuỗi có đơn vị về số: bỏ ký hiệu `%` ở `int_rate` và `revol_util`, tách số tháng ở `term`, ánh xạ `emp_length` từ dạng `"10+ years"` về số năm.
4. Chuyển `earliest_cr_line` thành độ dài lịch sử tín dụng tính bằng tháng, lấy mốc là `issue_d`. Giữ nguyên cột ngày tháng thì không đưa vào ma trận thiết kế được.
5. Cột định lượng thiếu giá trị: điền bằng trung vị. Cột định tính thiếu giá trị: gán thành mức riêng tên `"unknown"`.
6. Loại `funded_amnt` và `funded_amnt_inv`, vì cả hai gần trùng với `loan_amnt`. Ba cột gần cộng tuyến tạo ra trị riêng sát $0$ trong ma trận Gram, làm $\kappa$ tăng vọt mà không thêm thông tin nào.

### 3.4. Xây dựng ma trận thiết kế

- Giữ khoảng 25 đến 30 cột định lượng, gồm `loan_amnt`, `annual_inc`, `dti`, `fico_range_low`, `fico_range_high`, `open_acc`, `total_acc`, `revol_bal`, `revol_util`, `delinq_2yrs`, `inq_last_6mths`, `pub_rec`, `mort_acc`, `tot_cur_bal` và các cột cùng nhóm.
- Mã hóa one-hot các cột định tính: `term`, `home_ownership`, `verification_status`, `purpose`, `addr_state`, `application_type`, `initial_list_status`. Riêng `addr_state` có 51 mức và một mình nó đóng góp phần lớn số cột.
- Loại một mức của mỗi biến định tính khi one-hot. Giữ đủ mọi mức thì tổng các cột của một biến bằng vector toàn $1$, tạo ra một hướng suy biến chính xác trong ma trận Gram. Hiệu chỉnh Ridge vẫn giữ bài toán lồi mạnh nên không có lỗi số học, nhưng $\mu$ tụt đúng về $\lambda$ mà không vì lý do nào thuộc về dữ liệu.
- **Chuẩn hóa tất cả các cột về trung bình $0$, độ lệch chuẩn $1$.** Bước này bắt buộc và có vai trò tối ưu hóa trực tiếp, vì nó ép các trị riêng của ma trận Gram về cùng một thang. Không chuẩn hóa thì `annual_inc` cỡ $10^5$ và `dti` cỡ $10$ nằm chung một ma trận, và $\kappa$ lớn tới mức GD gần như đứng yên. Mục 5.7 đo chênh lệch này.
- Với biến mục tiêu, trừ trung bình để khử hệ số chặn. Không lấy logarit, vì `int_rate` nằm gọn trong khoảng 5 đến 31 phần trăm và không lệch mạnh.

Ước tính $d$ khoảng 120 đến 150. Con số này nhỏ hơn nhiều so với $n$, nên $\frac{1}{n}X^{\top}X$ đủ hạng và $\mu$ nhiều khả năng do dữ liệu quyết định chứ không do riêng $\lambda$. Cần kiểm chứng bằng phổ trị riêng thật ngay sau khi dựng xong ma trận, vì kết luận này chi phối toàn bộ mục 5.6.

### 3.5. Quy mô mẫu và chi phí tính toán

Đề bài yêu cầu lấy khoảng 1 đến 1,5 triệu mẫu. Yêu cầu này khả thi về bộ nhớ: với $n = 1{,}2$ triệu và $d = 150$ ở kiểu `float64`, ma trận thiết kế chiếm khoảng 1,4 GB. Nhưng nó không khả thi cho phần quét tham số, vì một lần tính gradient phải đọc hết ma trận hai lượt, mất cỡ 0,1 đến 0,15 giây, nên một lần chạy GD 5000 vòng mất chừng 10 phút và riêng lưới tám độ dài bước ở mục 5.2 đã mất hơn một giờ. Cộng cả phần SGD 200 epoch thì tổng thời gian lên tới hàng chục giờ.

Kế hoạch vì thế dùng hai quy mô:

| Quy mô | $n$ | Dùng cho | Lý do |
|---|---|---|---|
| Quét tham số | 200 nghìn | Các nhóm `step-*`, `momentum-*`, `batch-*`, `newton-damping` | Gradient cỡ 0,02 giây, chạy hết lưới trong vài chục phút |
| Toàn phần | 1,2 triệu | Nhóm biểu đồ G và H, bảng 5.4 | Đáp ứng yêu cầu đề bài, và là nơi thứ hạng theo thời gian có ý nghĩa |

Cách chia này giữ được yêu cầu về quy mô mà không phải trả giá bằng lưới tham số thưa. Điều kiện để nó hợp lệ là $L$, $\mu$ và $\kappa$ ở hai quy mô phải gần nhau, vì hệ số $\frac{1}{n}$ trong hàm mục tiêu đã khử ảnh hưởng của kích thước mẫu. Nhóm cần kiểm tra và báo cáo cả hai bộ ba hằng số; nếu chúng lệch quá 10 phần trăm thì mẫu 200 nghìn chưa đại diện và phải lấy mẫu lớn hơn.

Lấy mẫu bằng `numpy.random.default_rng(seed)` với seed cố định, ghi seed vào `problem_config.json`.

### 3.6. Chia dữ liệu và chốt $\lambda$

Chia train và test theo tỷ lệ $80/20$ với seed cố định.

Chọn $\lambda$ bằng cross-validation $5$ fold trên tập train, quét trên lưới logarit từ $10^{-6}$ đến $10^{2}$, khớp với khoảng $10^{-3}$ đến $10$ mà đề bài gợi ý nhưng rộng hơn về hai phía để nhìn được cả hai đầu của đường cong.

Cần phân biệt rõ hai việc, và nên nói thẳng chỗ này khi thuyết trình:

- Chọn $\lambda$ là bài toán **học máy**, tức chọn mô hình. Làm một lần, ở giai đoạn này.
- Cực tiểu hóa $f$ với $\lambda$ đã cho là bài toán **tối ưu hóa**. Đây mới là nội dung chính của bài tập.

Nếu đường cong cross-validation phẳng trên một khoảng dài, không lấy điểm cực tiểu, vì cực tiểu của một đường phẳng là lựa chọn tùy tiện và thường rơi vào $\lambda$ nhỏ nhất trong lưới, kéo theo $\kappa$ lớn nhất có thể. Thay bằng quy tắc một sai số chuẩn: chọn $\lambda$ lớn nhất còn nằm trong một sai số chuẩn của giá trị tốt nhất. Cách này đổi một phần rất nhỏ chất lượng dự báo lấy một bài toán tối ưu hóa dễ hơn nhiều, và bản thân sự đánh đổi đó là nội dung đáng đưa vào báo cáo.

Giữ thêm hai giá trị $\lambda$ khác, một rất nhỏ và một lớn, để làm thí nghiệm ở mục 5.6.

### 3.7. Sản phẩm của giai đoạn

Lưu ra `data/processed/`:

- `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy` cho quy mô quét tham số.
- `X_train_full.npy` và các file tương ứng cho quy mô toàn phần.
- `feature_names.json`.
- `problem_config.json`: chứa $\lambda$, $n$, $d$, $L$, $\mu$, $\kappa$, $f^*$, $\|w^*\|$, seed, và bộ ba hằng số của cả hai quy mô.

Từ thời điểm này, không ai được sửa các file trên nữa.

---

## 4. Giai đoạn 2: Cài đặt các thuật toán

### 4.1. Tách hai trục: hướng đi và độ dài bước

Đề bài có dạng một ma trận bốn thuật toán nhân hai cách chọn bước, nên kế hoạch tách đúng hai trục đó thành hai loại đối tượng rời nhau, rồi viết **một vòng lặp duy nhất** ghép chúng lại. Cách quen thuộc hơn là mỗi thuật toán một hàm tự chứa vòng lặp riêng, nhưng khi đó phần đo thời gian, phần ghi log và phần kiểm tra điều kiện dừng bị chép lại bốn lần, và chỉ cần một bản chép sai là hai đường trên cùng một hình không còn so sánh được với nhau.

**Trục thứ nhất, hướng đi.** Mỗi thuật toán là một lớp có trạng thái, trả lời câu hỏi "đi từ điểm nào, theo hướng nào".

```python
class Direction(Protocol):
    def propose(self, w, k) -> Proposal:
        """Proposal(point, direction, grad, local_objective, n_data).

        point            : where the step starts from; equals w for GD, SGD and
                           Newton, but equals the extrapolated y_k for Nesterov
        direction        : p, the descent direction at `point`
        grad             : gradient at `point`, reused by the line search
        local_objective  : the function the line search may probe; the full f for
                           deterministic methods, the mini-batch f for SGD
        n_data           : rows of X touched, for the fair-comparison axis
        """

    def accept(self, w_new, t, k) -> None:
        """Update internal state once the loop has committed to step size t."""
```

**Trục thứ hai, độ dài bước.**

```python
class StepRule(Protocol):
    def __call__(self, proposal, k) -> tuple[float, int]:
        """Returns (step size, number of objective evaluations spent)."""
```

Ba hiện thực: `Fixed(t)` trả về hằng số, `Armijo(c, rho, t0, max_tries)` lùi bước cho tới khi thỏa điều kiện giảm, `Decay(kind, eta0, gamma)` cho các quy tắc giảm dần của SGD.

**Vòng lặp chung.** Một hàm `iterate(objective, direction, step_rule, ...)` gọi `propose`, gọi `step_rule`, cập nhật $w \leftarrow \text{point} + t \cdot p$, gọi `accept`, rồi ghi log. Tám cấu hình bắt buộc của đề bài vì thế là tích Descartes của hai danh sách, viết ra được thành một vòng `for` lồng nhau, nên câu hỏi "đã phủ đủ tám cấu hình chưa" trả lời được bằng cách nhìn vào code chứ không phải bằng cách đọc lại báo cáo.

Bốn điều mà cách tách này làm đúng theo thiết kế, không phải nhờ cẩn thận:

1. **Điều kiện Armijo viết ở dạng tổng quát.** Đề bài ghi điều kiện dưới dạng $f(w - t\nabla f) \le f(w) - ct\|\nabla f\|^2$, dạng chỉ đúng khi hướng đi bằng $-\nabla f$. Vì `Armijo` nhận hướng $p$ từ bên ngoài, nó phải dùng dạng tổng quát

$$
f\!\left( y + t p \right) \;\le\; f(y) + c \, t \, \nabla f(y)^{\top} p
$$

   dạng này rút về đúng công thức của đề bài khi $p = -\nabla f(y)$, nhưng vẫn đúng cho hướng Newton $p = -H^{-1}\nabla f$, khi đó $\nabla f^{\top} p = -\nabla f^{\top} H^{-1} \nabla f < 0$.

2. **Momentum của Nesterov luôn khớp với bước thật.** Hàm `accept` nhận đúng giá trị $t$ mà line search vừa chọn, nên `Nesterov` tính lại $\beta$ từ $t$ đó theo công thức ở mục 4.3. Ghép một hằng số momentum tính cho $t = 1/L$ với một bước lớn hơn do backtracking chấp nhận là lỗi làm hàm mục tiêu bùng lên, và cấu trúc này chặn nó ngay ở kiểu dữ liệu.

3. **Trường `point` tách khỏi $w$.** Nesterov tính gradient tại điểm ngoại suy $y_k$ chứ không tại $w_k$, nên vòng lặp phải biết bước xuất phát từ đâu. Gộp hai khái niệm làm một là lỗi kinh điển khi cài Nesterov, và ở đây hai khái niệm nằm ở hai trường khác nhau.

4. **Trường `local_objective` phơi bày giới hạn của Armijo cho SGD.** Với SGD, hàm mà line search dò là hàm mục tiêu của riêng lô hiện tại, nên bước được chấp nhận chỉ bảo đảm giảm trên lô đó. Đặt nó thành một trường có tên rõ ràng buộc người đọc code phải thấy sự khác biệt, thay vì để nó ẩn trong phần thân của một hàm `sgd` dài.

Ba điểm kỹ thuật phải làm đúng trong `iterate`, nếu sai thì toàn bộ biểu đồ theo trục thời gian mất giá trị. Cả ba nằm ở một chỗ duy nhất nên chỉ phải kiểm tra một lần:

1. **Không tính thời gian ghi log vào thời gian chạy.** Tính $f(w_k)$ tốn đúng một lần nhân ma trận với vector, tức ngang một vòng gradient descent, nên ghi log mỗi vòng mà không dừng đồng hồ thì mọi số đo thời gian bị thổi lên gấp đôi. Dùng `time.perf_counter()`, dừng đồng hồ trước khi tính và ghi log, chạy lại sau đó.
2. **Chạy warm-up trước khi đo.** Lần gọi numpy đầu tiên chịu chi phí khởi tạo và chi phí nạp dữ liệu vào bộ nhớ đệm. Chạy vài vòng lặp bỏ kết quả trước khi đo thật.
3. **Ghi log thưa dần với SGD.** Với 200 epoch trên mẫu 200 nghìn điểm và lô 256, số vòng lặp lên tới hơn 150 nghìn; ghi log mỗi vòng thì riêng phần log tốn hơn cả phần chạy. Đặt `record_every` theo số vòng mỗi epoch.

Kết quả trả về là một `RunRecord` thống nhất, gồm nghiệm cuối, các dãy $f$, chuẩn gradient, thời gian tích lũy, chỉ số vòng lặp, số dòng dữ liệu đã truy cập, số lần đánh giá hàm, cùng trạng thái dừng thuộc bốn giá trị `converged`, `max_iter`, `diverged`, `stalled`.

### 4.2. Danh mục thuật toán bắt buộc

Đề bài yêu cầu bốn thuật toán, mỗi thuật toán hai cơ chế chọn bước, tức tám cấu hình gốc. Theo cách tách ở mục 4.1, bảng dưới đây đọc theo hàng là chọn `Direction`, đọc theo cột là chọn `StepRule`.

| `Direction` | với `Fixed` | với `Armijo` | Trạng thái riêng của lớp |
|---|---|---|---|
| `SteepestDescent` | Quét quanh $1/L$ và $2/(L+\mu)$ | $c$, $\rho$, $t_0$ | Không có |
| `MiniBatch(B, rng)` | Kèm `Decay` cho các quy tắc giảm dần | Armijo trên lô hiện tại, xem 4.1 | Bộ sinh ngẫu nhiên, chỉ số lô |
| `Nesterov(mu)` | Ba công thức momentum, có và không restart | $\beta$ tính lại theo $t$ thật | $w_{k-1}$, bộ đếm restart |
| `NewtonStep` | $t = 1$ | Damped | Phân rã Cholesky đã lưu |

### 4.2.1. Ranh giới của cách tách

Hai chỗ mà cách tách này không sạch, nên ghi trước để không phải sửa kiến trúc giữa chừng:

- **Adam không có độ dài bước vô hướng.** Adam co giãn theo từng tọa độ, nên phần co giãn phải nằm trong `Direction` và `StepRule` chỉ còn giữ hệ số học. Cách đó chạy được nhưng làm nhòe ranh giới hai trục, nên nếu nhóm làm Adam ở mục 4.5 thì phải nói rõ trong báo cáo rằng nó không nằm gọn trong khung.
- **SGD không dùng chung điều kiện dừng với nhóm tất định.** Chuẩn gradient trên một lô là đại lượng ngẫu nhiên, nên `iterate` phải chuyển sang tiêu chí theo số epoch cho `MiniBatch`. Ranh giới này đặt bằng một cờ trong `RunRecord`, không đặt bằng cách viết một vòng lặp thứ hai.

### 4.3. Chi tiết từng thuật toán

**Gradient Descent.** Cập nhật

$$
w_{k+1} = w_k - t \, \nabla f(w_k)
$$

Lý thuyết cho biết phương pháp hội tụ khi $0 < t < 2/L$, và bước tối ưu cho hàm bậc hai lồi mạnh là $t = \dfrac{2}{L + \mu}$. Lưới tham số nên có một giá trị $t > 2/L$ để quan sát phân kỳ, vì đó là minh chứng trực quan cho vai trò của hằng số Lipschitz và là slide dễ thuyết phục nhất trong phần GD.

**Backtracking line search (Armijo).** Bắt đầu từ $t = t_0$, lặp $t \leftarrow \rho t$ cho tới khi

$$
f\!\left( w - t \nabla f(w) \right) \;\le\; f(w) - c \, t \, \left\| \nabla f(w) \right\|_2^2
$$

với $c = 10^{-4}$ và $\rho = 0.5$ như đề bài gợi ý, kèm vài giá trị khác trong lưới.

Phải đếm và báo cáo **số lần đánh giá hàm mục tiêu**, không chỉ số vòng lặp. Mỗi lần thử bước tốn một lần tính $f$, và với $t_0$ đặt quá lớn thì line search có thể thử tới hàng chục lần trước khi chấp nhận. Đây chính là lý do backtracking thắng về số vòng lặp nhưng có thể thua về thời gian chạy, tức là lý do đề bài yêu cầu hai biểu đồ riêng biệt.

Cần thêm một giới hạn số lần thử và một tiêu chí dừng theo đình trệ. Khi $f(w_k) - f^*$ chạm sàn độ phân giải số học, cỡ $10^{-16} f^*$, mọi bước thử đều không thỏa điều kiện giảm và line search chạy hết giới hạn ở mỗi vòng lặp, biến phần đuôi đồ thị thành thời gian chết.

**Mini-batch SGD.** Mỗi bước lấy một lô ngẫu nhiên $\mathcal{B}$ với $|\mathcal{B}| = B$ và cập nhật theo gradient ước lượng

$$
g_k = \frac{1}{B} \sum_{i \in \mathcal{B}} \left( x_i^{\top} w_k - y_i \right) x_i + \lambda w_k,
\qquad
w_{k+1} = w_k - \eta_k \, g_k
$$

Bốn lưu ý:

- Vẽ theo trục số vòng lặp là **không công bằng** với SGD, vì một vòng SGD rẻ hơn một vòng GD chừng $n/B$ lần. Phải vẽ thêm theo số epoch hoặc số lần truy cập dữ liệu, và theo thời gian.
- SGD với bước hằng $\eta_k \equiv \eta$ **không hội tụ về $w^*$**, mà dao động trong một lân cận bán kính cỡ $\mathcal{O}\!\left( \eta \sigma^2 / \mu \right)$, với $\sigma^2$ là phương sai của gradient ngẫu nhiên. Trên thang log, hiện tượng này hiện ra thành một đường nằm ngang, và nhóm phải giải thích nó như một kết quả đúng chứ không sửa mã để nó biến mất. Nhóm biểu đồ D dựng quanh chính hiện tượng đó.
- Độ dài bước phải đặt theo kích thước lô. Hằng số Lipschitz của mất mát một mẫu là $\|x_i\|^2 + \lambda$, lớn hơn $L$ rất nhiều vì $L$ là trung bình còn cái kia là cực đại. Một bước an toàn cho gradient đầy đủ có thể làm SGD lô nhỏ phân kỳ. Đặt $L_B = L + \dfrac{L_{\max} - L}{B}$ và lấy bước cơ sở $1/L_B$ cho từng $B$.
- **Backtracking cho SGD chỉ bảo đảm giảm trên lô.** Trường `local_objective` ở mục 4.1 mang đúng hàm mục tiêu của lô hiện tại, nên bước được chấp nhận không nói gì về hàm $f$ đầy đủ. Đề bài yêu cầu làm backtracking cho cả bốn thuật toán nên vẫn phải cài, nhưng báo cáo phải ghi rõ rằng bảo đảm lý thuyết của Armijo không còn hiệu lực. Nếu chuyển sang kiểm tra trên toàn bộ dữ liệu thì mỗi vòng lại tốn đúng chi phí một vòng full-batch, và SGD mất sạch lợi thế về giá mỗi vòng.

Các quy tắc chọn bước cần thử, đều là công thức tất định theo $k$:

$$
\eta_k = \eta_0, \qquad
\eta_k = \frac{\eta_0}{1 + \gamma k}, \qquad
\eta_k = \frac{\eta_0}{\sqrt{k+1}}, \qquad
\eta_k = \eta_0 \cdot 2^{-\lfloor k / (10 n_{\text{epoch}}) \rfloor}
$$

**Nesterov Accelerated Gradient.**

$$
\begin{aligned}
y_k &= w_k + \beta_k \left( w_k - w_{k-1} \right) \\
w_{k+1} &= y_k - t \, \nabla f(y_k)
\end{aligned}
$$

Thử ba công thức momentum: hằng số $\beta = \dfrac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$ cho tốc độ lý thuyết tốt nhất với hàm lồi mạnh; dãy tăng dần $\beta_k = \dfrac{k - 1}{k + 2}$ cho trường hợp không giả định biết $\mu$, kèm dao động tuần hoàn thấy rõ trên đồ thị; và $\beta = 0.9$ như đề bài gợi ý, làm điểm đối chứng theo mục 1.2.

Khi ghép momentum với backtracking, **phải tính lại $\beta$ theo bước thực tế** bằng

$$
\beta = \frac{1 - \sqrt{t\mu}}{1 + \sqrt{t\mu}}
$$

Công thức $\beta = (\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$ chỉ đúng khi $t = 1/L$; ghép nó với một bước lớn hơn mà line search vừa chấp nhận thì hàm mục tiêu bùng lên thay vì giảm. Phép tính lại này đặt trong `Nesterov.accept`, nơi nhận đúng $t$ mà `StepRule` vừa trả về, nên không có đường nào để hai giá trị lệch nhau.

Nên cài thêm **adaptive restart**: khi

$$
\nabla f(y_k)^{\top} \left( w_{k+1} - w_k \right) > 0
$$

thì đặt lại bộ đếm momentum về $0$. Kỹ thuật này khử dao động của công thức $\beta_k$ tăng dần và thường cho kết quả tốt nhất trong nhóm bậc một, nên so sánh có restart với không restart là một thí nghiệm gọn mà cho hình rõ ràng.

**Newton.** Giải hệ

$$
\nabla^2 f(w_k) \, p_k = -\nabla f(w_k), \qquad w_{k+1} = w_k + t \, p_k
$$

- Dùng phân rã Cholesky (`scipy.linalg.cho_factor` và `cho_solve`), không tính `np.linalg.inv`. Nghịch đảo tường minh vừa đắt hơn vừa kém ổn định số học, và đây là điểm nên nêu trong báo cáo.
- Hessian là hằng số nên phân rã được một lần rồi dùng lại. Để so sánh trung thực với trường hợp tổng quát, đo cả hai phương án: phân rã lại mỗi vòng, và phân rã một lần.
- Chi phí mỗi vòng là $\mathcal{O}(nd^2 + d^3)$ so với $\mathcal{O}(nd)$ của gradient descent. Với $d$ khoảng 150, tỷ số này là $d$ lần chứ không phải $d^2$ lần, nên xem lại dự đoán ở mục 1.2 trước khi kết luận Newton chậm.

### 4.4. Bài toán phụ để phần Newton và phần backtracking có nội dung

Như mục 1.2 đã nêu, Newton kết thúc sau một vòng trên hàm bậc hai và backtracking luôn nhận ngay bước đầy đủ. Đề xuất thêm một hàm mục tiêu phi tuyến trên cùng bộ dữ liệu, dùng mất mát Huber kết hợp hiệu chỉnh Ridge:

$$
f_{\text{huber}}(w) = \frac{1}{n} \sum_{i=1}^{n} H_{\delta}\!\left( x_i^{\top} w - y_i \right) + \frac{\lambda}{2} \left\| w \right\|_2^2,
\qquad
H_{\delta}(r) =
\begin{cases}
\dfrac{r^2}{2}, & |r| \le \delta \\[2ex]
\delta \left( |r| - \dfrac{\delta}{2} \right), & |r| > \delta
\end{cases}
$$

Hàm này lồi, khả vi bậc một liên tục, Hessian tồn tại hầu khắp nơi, nhưng không còn là hàm bậc hai, nên Newton cần nhiều vòng lặp và thể hiện được hội tụ bậc hai ở giai đoạn cuối. Mất mát Huber cũng có lý do thực tế với Lending Club, vì `annual_inc` có những giá trị khai báo tới hàng chục triệu đô la và kéo theo phần dư rất lớn ở một nhóm nhỏ hồ sơ.

Phần này tùy chọn. Nếu quỹ thời gian hạn chế thì lược bỏ, và bài vẫn đáp ứng đủ yêu cầu đề bài với riêng bài toán Ridge, đổi lại phần Newton chỉ còn một điểm dữ liệu trên đồ thị.

### 4.5. Thuật toán mở rộng (đề bài khuyến khích, chọn hai hoặc ba)

| Thuật toán | Lý do đưa vào |
|---|---|
| Heavy-ball (Polyak momentum) | So sánh trực tiếp với Nesterov, cho thấy khác biệt giữa hai dạng momentum |
| Adam | Phổ biến trong thực tế, và trên hàm lồi mạnh nó thường thua AGD, một kết quả đáng bàn |
| L-BFGS tự cài, bộ nhớ $m = 5$ hoặc $10$ | Nhóm quasi-Newton, chi phí $\mathcal{O}(md)$ mỗi vòng thay vì $\mathcal{O}(d^3)$ |
| Coordinate Descent | Có nghiệm đóng theo từng tọa độ với hàm bậc hai |
| Newton-CG | Giải hệ Newton xấp xỉ bằng CG, hữu ích khi $d$ lớn |

---

## 5. Giai đoạn 3: Thiết kế thí nghiệm

### 5.1. Cấu hình chung

- Điểm khởi tạo $w_0 = 0$ cho mọi thuật toán. Với các phương pháp ngẫu nhiên, chạy $5$ seed và báo cáo trung vị kèm dải min-max.
- Số vòng lặp tối đa đặt theo nhóm: GD và AGD lấy $5000$, SGD lấy $200$ epoch, Newton lấy $50$.
- Điều kiện dừng: $\left\| \nabla f(w_k) \right\| \le 10^{-10} \left\| \nabla f(w_0) \right\|$, hoặc chạm giới hạn vòng lặp, hoặc đình trệ theo mô tả ở mục 4.3. Tiêu chí đình trệ phải loại trừ các lần chạy đang phân kỳ, nếu không thì phép thử $t > 2/L$ bị dừng sớm và không quan sát được hiện tượng cần quan sát.
- Chỉ số theo dõi chính là $f(w_k) - f^*$ trên thang $\log_{10}$.
- Đo thời gian bằng `time.perf_counter()`, mỗi cấu hình chạy $3$ lần độc lập, lấy trung vị.

### 5.2. Lưới tham số cần quét

**GD bước cố định.**

$$
t \in \left\{ \frac{2.1}{L},\; \frac{2}{L},\; \frac{1.9}{L},\; \frac{2}{L+\mu},\; \frac{1}{L},\; \frac{0.5}{L},\; \frac{0.1}{L},\; \frac{0.01}{L} \right\}
$$

Giá trị $2.1/L$ đưa vào có chủ ý, để quan sát phân kỳ. Ngoài lưới theo $L$, chạy thêm lưới tuyệt đối $\eta \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}$ theo gợi ý của đề bài, rồi đối chiếu xem giá trị tốt nhất tìm được bằng cách dò rơi vào đâu so với $2/(L+\mu)$. Phép đối chiếu này trả lời thẳng câu hỏi vì sao nên tính $L$ thay vì dò tay.

**GD backtracking.** $c \in \{10^{-4},\, 0.1,\, 0.3\}$, $\rho \in \{0.5,\, 0.8,\, 0.9\}$, $t_0 \in \{1,\; 10/L\}$. Với mỗi cấu hình, ghi số lần đánh giá hàm trung bình mỗi vòng lặp.

**SGD.** Kích thước lô $B \in \{256,\, 512,\, 1024,\, 2048\}$ theo đề bài, thêm $B = 32$ để nhìn rõ hơn phần zigzag, kết hợp với bốn quy tắc chọn bước ở mục 4.3 và $\eta_0$ quét trên lưới logarit. Số tổ hợp ở đây lớn nhất trong cả lưới, nên phần này chia cho hai người.

**AGD.** Ba công thức momentum ở mục 4.3, có và không restart, $t \in \{1/L,\; 0.5/L\}$, cộng biến thể backtracking.

**Newton.** $t = 1$ so với damped có backtracking, phân rã lại mỗi vòng so với dùng lại, trên cả hai hàm mục tiêu nếu có làm phần Huber.

**$\lambda$.** Theo đề bài, quét $\lambda$ trên lưới logarit từ $10^{-3}$ đến $10$ cho thí nghiệm ở mục 5.6.

### 5.3. Bộ biểu đồ cần vẽ

Đề bài yêu cầu rõ hai biểu đồ bắt buộc: hàm mất mát theo số vòng lặp và hàm mất mát theo thời gian chạy. Quy ước của nhóm là mọi so sánh đều xuất hiện dưới dạng một cặp hình như vậy.

Mỗi nhóm mang một tên ngắn dùng luôn làm tên file, thay vì đánh chữ cái. Nhìn vào `results/figures/` là biết hình nào vẽ gì, và khi một nhóm bị bỏ hay tách đôi thì không phải đánh số lại từ đầu. Cột cuối cho biết hình rơi vào chương nào của báo cáo theo sườn ở mục 10.

| Tên nhóm | Nội dung so sánh | Số hình | Chương |
|---|---|---|---|
| `spectrum` | Phổ trị riêng của ma trận Gram và sàn do $\lambda$ đặt | 1 | 2 |
| `scaling-kappa` | Ba cách mã hóa cột: thô, chỉ trừ trung bình, chuẩn hóa (mục 5.7) | 2 | 2 |
| `step-fixed` | Bước cố định, quét $t$ quanh $1/L$, gồm cả trường hợp phân kỳ | 2 | 4 |
| `step-blind` | Các bước dò tay theo đề bài, không tính $L$ | 2 | 4 |
| `step-armijo` | Backtracking, các cấu hình $c$, $\rho$, $t_0$, kèm số lần đánh giá hàm | 2 | 4 |
| `momentum-formula` | Ba công thức $\beta$ của Nesterov | 2 | 5 |
| `momentum-restart` | Nesterov có và không adaptive restart | 2 | 5 |
| `batch-size` | SGD, các kích thước lô khác nhau | 3 | 6 |
| `batch-eta` | SGD, lưới logarit của $\eta_0$ tại một kích thước lô | 3 | 6 |
| `batch-schedule` | SGD, bốn quy tắc chọn bước, gồm hiện tượng bão hòa | 3 | 6 |
| `newton-damping` | Newton bước đầy đủ và damped, hai cách xử lý Hessian | 2 | 7 |
| `headline` | **So sánh tổng hợp:** mỗi thuật toán ở cấu hình tốt nhất | 2 | 8 |
| `library` | So sánh mã tự viết với scikit-learn (mục 6) | 2 | 9 |
| `lambda-kappa` | Ảnh hưởng của $\lambda$ lên tốc độ hội tụ (mục 5.6) | 2 | 10 |
| `lambda-kappa_tradeoff` | $\kappa$ và RMSE cùng vẽ theo $\lambda$ trên hai trục tung | 1 | 10 |

Hai nhóm `batch-*` có ba hình vì trục vòng lặp không so sánh công bằng được SGD với các phương pháp full-batch, nên cần thêm một hình theo số epoch.

Quy ước trình bày, để các hình đọc được nhất quán:

- Trục tung dùng `plt.semilogy` với $f(w_k) - f^*$.
- Tên file ghép từ tên nhóm và tên trục hoành: `step-fixed_iter.pdf`, `step-fixed_time.pdf`, `batch-size_epoch.pdf`.
- Mỗi thuật toán một màu cố định, bảng màu định nghĩa một lần trong `src/figures.py`.
- Legend ghi rõ tham số, ví dụ `GD (t = 1/L)`, không ghi chung chung là `GD`.
- Mỗi hình đi kèm ít nhất hai câu kết luận trong văn bản. Hình không có kết luận thì chưa tính là xong.

Nhóm `headline` là hình quan trọng nhất của bài trình bày, và là hình trả lời trực tiếp yêu cầu "so sánh các thuật toán ở cấu hình tốt nhất" của đề bài.

### 5.4. Bảng tổng hợp kết quả

| Thuật toán | Cấu hình tốt nhất | Số vòng lặp đạt $f - f^* < 10^{-6}$ | Thời gian đạt ngưỡng đó (giây) | Số lần đánh giá hàm | $f$ cuối cùng | RMSE trên test |
|---|---|---|---|---|---|---|

Cột "số vòng lặp đạt ngưỡng" hữu ích hơn cột "số vòng lặp chạy hết", vì nó trả lời thẳng câu hỏi thuật toán nào nhanh hơn. Cột "số lần đánh giá hàm" là chỗ backtracking phải trả giá, nên bỏ cột đó đi thì bảng ưu ái backtracking một cách không công bằng.

### 5.5. Đối chiếu với lý thuyết

Với hàm bậc hai lồi mạnh, lý thuyết cho hai cận:

$$
\text{GD, } t = \tfrac{2}{L+\mu}: \quad f(w_k) - f^* \;\le\; \left( \frac{\kappa - 1}{\kappa + 1} \right)^{2k} \left( f(w_0) - f^* \right)
$$

$$
\text{AGD, momentum tối ưu:} \quad f(w_k) - f^* \;\le\; \left( 1 - \frac{1}{\sqrt{\kappa}} \right)^{k} C \left( f(w_0) - f^* \right)
$$

Ước lượng hệ số co rút quan sát được bằng cách khớp đường thẳng vào phần tuyến tính của đồ thị $\log_{10}\!\left( f(w_k) - f^* \right)$ theo $k$, rồi đối chiếu với cận lý thuyết. Phần này thể hiện được rằng nhóm hiểu vì sao các đường có dạng như vậy chứ không chỉ chạy được mã, và thường được đánh giá cao hơn việc chỉ trưng biểu đồ.

### 5.6. Thí nghiệm về ảnh hưởng của $\lambda$

Chạy lại GD và AGD với ít nhất ba giá trị $\lambda$. Vì

$$
\mu = \lambda_{\min}\!\left( \tfrac{1}{n} X^{\top} X \right) + \lambda,
\qquad
\kappa = \frac{\lambda_{\max}\!\left( \tfrac{1}{n} X^{\top} X \right) + \lambda}{\lambda_{\min}\!\left( \tfrac{1}{n} X^{\top} X \right) + \lambda}
$$

tăng $\lambda$ làm tăng $\mu$ và giảm $\kappa$, do đó tăng tốc độ hội tụ. Hiệu chỉnh Ridge vì thế có hai vai trò tách biệt: vai trò thống kê là chống quá khớp, vai trò tối ưu hóa là cải thiện điều kiện của bài toán. Hai vai trò đó nối chủ đề Ridge với nội dung môn học ở đúng một chỗ, và mục này là chỗ ấy.

Báo cáo bằng bảng gồm $\lambda$, $\mu$, $\kappa$, số vòng GD cần để đạt $10^{-6}$, và RMSE trên test. Bảng này thường cho thấy $\lambda$ giúp hội tụ nhanh không trùng với $\lambda$ cho RMSE tốt nhất, và sự lệch đó là nội dung đáng bàn hơn cả hai con số riêng lẻ.

Mức độ rõ rệt của thí nghiệm phụ thuộc vào phổ trị riêng thật. Nếu $\lambda_{\min}\!\left( \tfrac{1}{n} X^{\top} X \right)$ lớn hơn hẳn các giá trị $\lambda$ trong lưới thì $\kappa$ gần như không đổi và hình sẽ nhạt; khi đó phải nới lưới $\lambda$ lên phía trên, hoặc thêm vài cột gần cộng tuyến để tạo một kịch bản đối chiếu có $\kappa$ lớn.

### 5.7. Thí nghiệm về ảnh hưởng của chuẩn hóa

Chạy GD trên $X$ chưa chuẩn hóa và $X$ đã chuẩn hóa, báo cáo $\kappa$ cho cả hai. Với Lending Club, khoảng giá trị giữa các cột chênh nhau tới bốn bậc, từ `dti` cỡ hàng chục tới `annual_inc` cỡ $10^5$ và `tot_cur_bal` cỡ $10^6$, nên chênh lệch $\kappa$ dự kiến rất lớn. Chuẩn hóa vì thế không chỉ là thói quen tiền xử lý, mà là một can thiệp trực tiếp lên số điều kiện của bài toán tối ưu hóa.

---

## 6. Giai đoạn 4: So sánh với thư viện

Đề bài yêu cầu so sánh **giá trị hàm mất mát cuối cùng** và **thời gian tính toán** với `sklearn.linear_model.Ridge`. Độ chính xác dự báo có thể báo cáo thêm.

Chỗ dễ sai nhất là **quy đổi hàm mục tiêu về cùng một dạng**. `Ridge` cực tiểu hóa

$$
\left\| Xw - y \right\|_2^2 + \alpha \left\| w \right\|_2^2
$$

tức không có hệ số $\frac{1}{n}$ và không có hệ số $\frac{1}{2}$. Nhân hàm mục tiêu của nhóm với $2n$ ta được $\left\| Xw - y \right\|_2^2 + \lambda n \left\| w \right\|_2^2$, nên

$$
\alpha_{\text{sklearn}} = \lambda \cdot n
$$

Quan hệ này phải kiểm chứng bằng số chứ không tin suy luận trên giấy: lấy $\hat{w}$ do sklearn trả về, cắm vào hàm $f$ của nhóm, rồi so với $f^*$ tính từ nghiệm đóng. Quy đổi đúng thì chênh lệch nằm ở mức sai số máy. Chênh lệch lớn hơn nghĩa là sai hằng số, và mọi so sánh sau đó đều vô nghĩa vì hai bên đang cực tiểu hóa hai hàm khác nhau.

Các đối tượng so sánh:

| Thư viện | Cấu hình | Ghi chú |
|---|---|---|
| `Ridge(solver='auto')` | Mặc định | Thường dùng Cholesky, tương đương nghiệm đóng |
| `Ridge(solver='sag')` | Mặc định | Gradient ngẫu nhiên trung bình |
| `Ridge(solver='lsqr')` | Mặc định | Phương pháp lặp dựa trên bình phương tối thiểu |
| `SGDRegressor(penalty='l2')` | Mặc định | Đối chiếu trực tiếp với SGD tự cài |
| `LinearRegression` | Mặc định | Trường hợp $\lambda = 0$, để tham chiếu |

Bảng kết quả cần có: $f$ đạt được, sai số $f - f^*$, thời gian huấn luyện, RMSE trên test.

Dự đoán để nhóm không bất ngờ khi trình bày. `Ridge` với solver mặc định sẽ đạt $f^*$ ở mức sai số máy và rất nhanh, vì nó giải trực tiếp hệ phương trình chuẩn tắc thay vì lặp. `SGDRegressor` với tham số mặc định thường **không** hội tụ tốt trên bài toán chưa tinh chỉnh và có thể cho $f$ cao hơn hẳn mã tự viết đã dò tham số. Cả hai quan sát đều hợp lệ và đáng bàn: tham số mặc định của thư viện không phải lúc nào cũng phù hợp, và với bài toán có nghiệm đóng thì phương pháp trực tiếp khó bị đánh bại.

Kết luận nên đưa ra một cách cân bằng. Mục đích của việc tự cài đặt không phải chạy nhanh hơn thư viện, mà là hiểu cơ chế hội tụ và biết cách chọn tham số. Khi $d$ lớn tới mức chi phí $\mathcal{O}(d^3)$ không chấp nhận được, hoặc khi dữ liệu không nạp hết vào bộ nhớ, các phương pháp lặp mới thể hiện lợi thế.

---

## 7. Cấu trúc mã nguồn đề xuất

```
Optimization_1/
├── README.md
├── CLAUDE.md
├── KE_HOACH_TRIEN_KHAI.md
├── requirements.txt
├── data/
│   ├── raw/                    # file tải từ Kaggle
│   └── processed/              # X_train.npy, ... , problem_config.json
├── docs/
│   ├── van-phong-tieng-viet.md
│   └── quy-uoc-bao-cao.md
├── src/
│   ├── dataset.py              # đọc, làm sạch, dựng ma trận thiết kế, chọn lambda
│   ├── objective.py            # RidgeObjective: f, grad, hess, w_star, f_star, L, mu
│   ├── objective_huber.py      # (tùy chọn) bài toán phụ ở mục 4.4
│   ├── direction.py            # TRỤC 1: SteepestDescent, MiniBatch, Nesterov, NewtonStep
│   ├── stepsize.py             # TRỤC 2: Fixed, Armijo, Decay
│   ├── iterate.py              # vòng lặp duy nhất, đồng hồ, ghi log, điều kiện dừng
│   ├── record.py               # RunRecord, đọc ghi JSON
│   ├── experiment.py           # spec khai báo -> tích Descartes -> chạy -> lưu
│   ├── reference.py            # bọc scikit-learn cho mục 6
│   └── figures.py              # hàm vẽ chuẩn, bảng màu dùng chung
├── notebooks/
│   ├── 01_data_audit.ipynb            # khảo sát, đo rò rỉ bằng số        -> chương 2
│   ├── 02_build_problem.ipynb         # X, y, lambda, L, mu, kappa, f*    -> chương 1, 2
│   ├── 03_sweep_step_size.ipynb       # step-fixed, step-armijo           -> chương 4
│   ├── 04_sweep_momentum.ipynb        # momentum-formula, -restart        -> chương 5
│   ├── 05_sweep_stochastic.ipynb      # batch-size, batch-schedule        -> chương 6
│   ├── 06_second_order.ipynb          # newton-damping                    -> chương 7
│   ├── 07_scale_and_headline.ipynb    # chạy lại ở 1,2 triệu, headline    -> chương 8
│   └── 08_reference_and_lambda.ipynb  # library, lambda-kappa             -> chương 9, 10
├── tests/
│   ├── test_objective.py       # gradient, Hessian, nghiệm đóng (mục 8)
│   ├── test_dataset.py
│   ├── test_composition.py     # mọi cặp Direction x StepRule cùng về một w*
│   └── test_report.py
├── results/
│   ├── raw/                    # kết quả chạy dạng JSON, một file mỗi nhóm
│   └── figures/                # <tên nhóm>_<trục>.pdf và .png
└── report/
    ├── preamble.tex
    ├── report.tex
    ├── slides.tex
    ├── refs.bib
    └── README.md
```

Toàn bộ logic thuật toán nằm trong `src/`, notebook chỉ gọi hàm và vẽ. Cách tổ chức này chặn được tình trạng cùng một thuật toán bị sửa ở ba notebook khác nhau rồi cho ba kết quả khác nhau.

Ba quyết định về cách chia file, nêu lý do để người sửa sau không gộp ngược lại:

- **`direction.py` và `stepsize.py` chia theo hai trục của đề bài, không chia theo bậc của phương pháp.** Xếp Newton chung với GD và tách riêng phần chọn bước làm cho ma trận bốn nhân hai hiện ra thẳng trong cây thư mục. Cách chia theo bậc, tức một file cho phương pháp bậc một và một file cho bậc hai, lại cắt đúng vào giữa trục mà đề bài yêu cầu so sánh.
- **`iterate.py` chỉ có một hàm và không nên dài thêm.** Mọi thứ đặc thù cho một thuật toán phải nằm trong lớp `Direction` tương ứng. Khi thấy mình sắp viết `if isinstance(direction, MiniBatch)` trong `iterate.py`, đó là dấu hiệu phần đó thuộc về `MiniBatch`.
- **`experiment.py` nhận spec dạng dữ liệu, không phải một hàm cho mỗi nhóm.** Mỗi nhóm biểu đồ ở mục 5.3 là một dict mô tả danh sách `Direction`, danh sách `StepRule` và các tham số chung, còn phần chạy và phần lưu JSON viết một lần. Nhờ vậy, thêm một giá trị vào lưới là sửa một dòng dữ liệu chứ không phải sửa mã.

Cơ chế chống gián đoạn đặt ở `experiment.py`: mỗi nhóm ghi ra `results/raw/<tên nhóm>.json` ngay khi chạy xong, và hàm chạy bỏ qua nhóm nào đã có file. Chạy lưới đầy đủ mất hàng giờ theo số đo ở mục 13.2, nên một lần máy ngủ giữa chừng mà không có cơ chế này thì mất toàn bộ.

---

## 8. Kiểm thử tính đúng đắn

Bốn phép kiểm tra sau là tối thiểu, phải chạy xong trước khi bắt đầu quét lưới tham số.

1. **Kiểm tra gradient bằng sai phân hữu hạn.** So sánh $\left[ \nabla f(w) \right]_i$ giải tích với sai phân trung tâm

$$
\frac{f(w + \varepsilon e_i) - f(w - \varepsilon e_i)}{2\varepsilon}, \qquad \varepsilon = 10^{-6}
$$

   tại vài tọa độ $i$ ngẫu nhiên. Sai số tương đối phải dưới $10^{-6}$.

2. **Kiểm tra Hessian** tương tự, dùng sai phân của gradient.
3. **Kiểm tra nghiệm đóng.** Xác nhận $\left\| \nabla f(w^*) \right\|$ ở mức sai số máy.
4. **Kiểm tra trên bài toán nhỏ có nghiệm biết trước.** Sinh dữ liệu giả với $n = 100$, $d = 5$, chạy mọi thuật toán, xác nhận tất cả hội tụ về cùng một $w^*$.
5. **Kiểm tra toàn bộ tích Descartes.** Cách tách hai trục ở mục 4.1 cho phép viết phép kiểm này thành một vòng lặp: với mọi cặp `Direction` nhân `StepRule` hợp lệ, chạy trên bài toán nhỏ ở phép kiểm 4 và xác nhận cả tám cấu hình đều đạt $\|w - w^*\| < 10^{-8}$. Cấu trúc theo hàm rời không viết được phép kiểm dạng này, vì mỗi hàm có chữ ký riêng.

Phép kiểm thứ năm còn trả lời được câu hỏi của người chấm rằng nhóm đã cài đủ tám cấu hình bắt buộc hay chưa, vì danh sách chạy trong test chính là danh sách nộp. Bỏ qua cả mục này thì rủi ro lớn nhất là chạy hết toàn bộ thí nghiệm rồi mới phát hiện gradient sai dấu ở một nhánh, và toàn bộ số liệu thời gian phải đo lại.

---

## 9. Lịch trình và phân công

Giả định nhóm bốn người. Nếu số thành viên khác thì gộp hoặc tách vai trò tương ứng.

| Tuần | Nội dung | Sản phẩm |
|---|---|---|
| 1 | Dựng môi trường, khảo sát Lending Club, xử lý rò rỉ, dựng $X$ và $y$, chốt $\lambda$, tính $L$, $\mu$, $\kappa$, $f^*$ | `data/processed/`, notebook 01 và 02 |
| 2 | Cài `objective.py`, `direction.py`, `stepsize.py`, `iterate.py`, chạy toàn bộ kiểm thử mục 8 | `src/` chạy được, test pass |
| 3 | Chạy lưới ở quy mô 200 nghìn, sinh `step-*`, `momentum-*`, `batch-*`, `newton-damping`, ghi kết luận từng nhóm | `results/`, notebook 03 đến 06 |
| 4 | Chạy lại cấu hình tốt nhất ở quy mô toàn phần, sinh `headline`, `library`, `lambda-kappa` | notebook 07 và 08 |
| 5 | Viết báo cáo LaTeX, làm slide Beamer, tập trình bày chéo | `report/report.pdf`, `report/slides.pdf` |

Phân công đi theo trục `Direction`, vì đó là trục mà mỗi người phải hiểu sâu một lớp và phải đọc được ba lớp còn lại. Trục `StepRule` thì cả nhóm dùng chung, nên người viết `Armijo` phải bàn giao sớm ở tuần 2.

| Vai trò | Phụ trách chính | Nhóm biểu đồ | Chương báo cáo |
|---|---|---|---|
| A | `dataset.py`, `objective.py`, `iterate.py`, thí nghiệm 5.6 và 5.7 | `scaling-kappa`, `lambda-kappa` | 1, 2, 3, 10 |
| B | `stepsize.py` cả ba lớp, `SteepestDescent` | `step-fixed`, `step-armijo` | 4 |
| C | `MiniBatch` và các quy tắc giảm dần | `batch-size`, `batch-schedule` | 6 |
| D | `Nesterov` và `NewtonStep` | `momentum-*`, `newton-damping` | 5, 7 |
| Cả nhóm | `experiment.py`, `figures.py`, `reference.py` | `headline`, `library` | 8, 9, 11 |

Vai trò B giữ `stepsize.py` vì `Armijo` là chỗ ba người còn lại đều phải gọi tới, còn `SteepestDescent` là lớp `Direction` đơn giản nhất nên ghép được vào cùng một người. Mỗi thành viên vẫn phải đọc và hiểu phần của người khác: trước buổi thuyết trình, tổ chức một buổi tập trong đó mỗi người trình bày phần **không phải** của mình. Cách này phát hiện lỗ hổng hiểu biết sớm hơn hẳn so với việc mỗi người chỉ ôn phần mình.

---

## 10. Sườn báo cáo và slide

### 10.1. Sườn báo cáo

Xương sống của phần phân tích đi theo **câu hỏi tối ưu hóa**, không theo tên thuật toán. Cách quen thuộc hơn là mỗi thuật toán một chương, nhưng khi đó bốn chương lặp lại cùng một dàn ý và người đọc phải tự ghép các mảnh để trả lời câu hỏi mà đề bài thực sự hỏi, tức thuật toán nào nhanh hơn và nhanh hơn theo nghĩa nào.

Ba chương lõi có chung một hình dạng. Mỗi chương lấy một thuật toán làm ví dụ nhưng đặt nó vào cùng một khung đánh đổi giữa **số vòng lặp** và **giá của mỗi vòng**, tức đúng hai trục mà đề bài bắt vẽ. Nhờ vậy chương 8 chỉ còn việc đặt cả bốn lên chung một cặp trục, không phải giới thiệu lại khái niệm nào.

| Chương | Tiêu đề | Nội dung chính | Nhóm biểu đồ |
|---|---|---|---|
| 1 | Bài toán và ba hằng số | Hàm mục tiêu, gradient, Hessian, nghiệm đóng, $L$, $\mu$, $\kappa$ đo được | |
| 2 | Dữ liệu: rò rỉ và chuẩn hóa | Cột phải loại và lý do, ảnh hưởng của chuẩn hóa lên $\kappa$, chọn $\lambda$ | `scaling-kappa` |
| 3 | Cài đặt và kiểm thử | Hai trục `Direction` và `StepRule`, năm phép kiểm ở mục 8 | |
| 4 | Chọn độ dài bước thế nào | Cố định so với backtracking, cắt ngang cả bốn thuật toán; giá của line search tính bằng số lần đánh giá hàm | `step-fixed`, `step-armijo` |
| 5 | Quán tính: đổi bộ nhớ lấy số vòng lặp | Nesterov, ba công thức $\beta$, restart; cùng giá mỗi vòng như GD nhưng ít vòng hơn | `momentum-formula`, `momentum-restart` |
| 6 | Ngẫu nhiên hóa: đổi độ chính xác lấy giá mỗi vòng | SGD, kích thước lô, quy tắc giảm bước, hiện tượng bão hòa | `batch-size`, `batch-schedule` |
| 7 | Thông tin bậc hai: đổi giá mỗi vòng lấy số vòng lặp | Newton một vòng, chi phí $\mathcal{O}(nd^2 + d^3)$, chỗ hòa vốn theo $d$ | `newton-damping` |
| 8 | So sánh tổng hợp trên hai trục | Cả bốn ở cấu hình tốt nhất, chỗ thứ hạng đổi chiều, đối chiếu tốc độ lý thuyết | `headline` |
| 9 | So sánh với scikit-learn | Quy đổi $\alpha = \lambda n$ kèm kiểm chứng số, bảng thời gian và $f$ | `library` |
| 10 | Ảnh hưởng của hệ số hiệu chỉnh | Hai vai trò của $\lambda$, thống kê và tối ưu hóa | `lambda-kappa` |
| 11 | Kết luận | | |

Tiêu đề chương 5, 6 và 7 cố tình dùng chung khuôn "đổi X lấy Y" để người đọc thấy ba chương đang trả lời cùng một câu hỏi trên ba phương án khác nhau. Riêng khuôn này được phép lặp; mọi khuôn câu khác trong báo cáo vẫn theo quy tắc ở mục 5 của `docs/van-phong-tieng-viet.md`.

Cái giá của sườn theo câu hỏi là người chấm khó dò nhanh xem một thuật toán cụ thể nằm ở đâu, nên bảng đối chiếu ở mục 11 phải đặt ngay sau phần mở đầu của báo cáo chứ không đẩy xuống phụ lục.

### 10.2. Outline bài trình bày

Thời lượng dự kiến 20 phút, khoảng 20 slide, phân bổ nghiêng hẳn về phần tối ưu hóa. Slide đi theo đúng sườn báo cáo ở mục 10.1.

**Phần 1. Đặt vấn đề (2 slide, 2 phút)**

1. Bài toán dự đoán lãi suất khoản vay trên Lending Club, mô tả dữ liệu ngắn gọn gồm $n$, $d$ và vài thống kê cơ bản.
2. Phát biểu bài toán tối ưu hóa: hàm mục tiêu Ridge, gradient, Hessian, nghiệm đóng.

**Phần 2. Đặc trưng của bài toán (2 slide, 3 phút)**

3. Ba hằng số $L$, $\mu$, $\kappa$ tính được, và ý nghĩa của chúng với tốc độ hội tụ dự kiến.
4. Ảnh hưởng của chuẩn hóa lên $\kappa$ (`scaling-kappa`), slide cho thấy tiền xử lý và tối ưu hóa không tách rời.

**Phần 3. Chọn độ dài bước (3 slide, 3 phút)**

5. Bước cố định: `step-fixed`, gồm cả trường hợp phân kỳ khi $t > 2/L$, và đối chiếu bước dò tay với $2/(L+\mu)$.
6. Backtracking: `step-armijo`, kèm số lần đánh giá hàm mỗi vòng lặp.
7. Kết luận về cách chọn bước, áp cho cả bốn thuật toán chứ không riêng GD.

**Phần 4. Ba cách đánh đổi (5 slide, 5 phút)**

8. Khung chung: số vòng lặp nhân giá mỗi vòng, ba phương án nằm ở ba góc khác nhau.
9. Quán tính: `momentum-formula` và `momentum-restart`, cùng giá mỗi vòng nhưng ít vòng hơn GD.
10. Ngẫu nhiên hóa, kích thước lô: `batch-size`.
11. Ngẫu nhiên hóa, quy tắc chọn bước: `batch-schedule`, giải thích hiện tượng bão hòa khi bước hằng.
12. Bậc hai: `newton-damping`, vì sao hội tụ sau một vòng trên hàm bậc hai, và chi phí mỗi vòng đo được.

**Phần 5. So sánh tổng hợp (3 slide, 4 phút)**

13. `headline` theo số vòng lặp.
14. `headline` theo thời gian chạy, nhấn vào chỗ thứ hạng đổi so với slide trước.
15. Bảng tổng hợp mục 5.4 và kết luận về cấu hình nên chọn.

**Phần 6. So sánh với thư viện (2 slide, 2 phút)**

16. Cách quy đổi hàm mục tiêu giữa mã tự viết và sklearn, kèm bằng chứng kiểm chứng bằng số.
17. Bảng và nhóm `library`, kết luận cân bằng như mục 6.

**Phần 7. Kết luận (2 slide, 2 phút)**

18. Ảnh hưởng của $\lambda$ lên cả chất lượng dự báo lẫn tốc độ hội tụ (`lambda-kappa`).
19. Những điều rút ra: cách chọn độ dài bước, khi nào nên dùng backtracking, khi nào phương pháp bậc hai đáng chi phí, chỗ khớp và chỗ lệch giữa lý thuyết và thực nghiệm.

---

## 11. Đối chiếu với yêu cầu đề bài

| Yêu cầu trong đề | Đáp ứng tại |
|---|---|
| Tiền xử lý Lending Club, lấy 1 đến 1,5 triệu mẫu | Mục 3.2 đến 3.5 |
| Viết hàm tính Loss, Gradient, Hessian | Mục 1.1, `src/objective.py` |
| Tự cài Gradient Descent full-batch | Mục 4.3, `SteepestDescent` trong `src/direction.py` |
| Tự cài Mini-batch SGD | Mục 4.3, `MiniBatch` trong `src/direction.py` |
| Tự cài Nesterov Accelerated Gradient | Mục 4.3, `Nesterov` trong `src/direction.py` |
| Tự cài Newton | Mục 4.3, `NewtonStep` trong `src/direction.py` |
| Mỗi thuật toán có cả bước cố định và backtracking | Bảng 4.2; tám cấu hình là tích Descartes, kiểm bằng phép kiểm 5 ở mục 8 |
| Thử nhiều learning rate và nhiều $\lambda$, chọn cấu hình tốt nhất | Mục 5.2 và 5.6 |
| Biểu đồ hàm mất mát theo số vòng lặp | Cột trái mọi nhóm ở bảng 5.3 |
| Biểu đồ hàm mất mát theo thời gian chạy | Cột phải mọi nhóm ở bảng 5.3 |
| So sánh các thuật toán ở cấu hình tốt nhất | Nhóm `headline`, chương 8 |
| So sánh loss cuối và thời gian với `sklearn.Ridge` | Mục 6, nhóm `library`, chương 9 |
| Phân tích hiện tượng quan sát được | Mục 5.5, và yêu cầu hai câu kết luận mỗi hình ở 5.3 |
| Áp dụng thêm thuật toán khác (khuyến khích) | Mục 4.4 và 4.5 |
| Ghi chép sai lệch so với kế hoạch | Mục 13 |

---

## 12. Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Rò rỉ từ `sub_grade` và `installment` | $R^2$ trên 0,99 ngay ở mô hình đầu tiên | Loại các cột theo mục 3.2, ghi rõ lý do trong báo cáo |
| Không đủ bộ nhớ ở quy mô toàn phần | Máy bắt đầu swap, thời gian mỗi vòng dao động mạnh | Giảm $n$ xuống 800 nghìn, hoặc giảm $d$ bằng cách gộp `addr_state` theo vùng |
| Newton chỉ có một điểm trên đồ thị | Nhóm biểu đồ F quá mỏng | Bổ sung bài toán Huber ở mục 4.4, và chuyển trọng tâm sang so sánh chi phí mỗi vòng |
| $\kappa$ nhỏ, mọi thuật toán hội tụ như nhau | Các đường trong nhóm `headline` gần trùng nhau | Thêm kịch bản $\lambda$ rất nhỏ ở mục 5.6, hoặc giữ vài cột gần cộng tuyến để có bài toán khó hơn |
| Đo thời gian dao động mạnh giữa các lần chạy | Chênh lệch trên 20 phần trăm | Đóng ứng dụng khác, chạy 3 đến 5 lần lấy trung vị, kiểm tra lại rằng đã loại thời gian ghi log |
| Quy đổi hàm mục tiêu với sklearn bị sai | $f(\hat{w}_{\text{sklearn}})$ lệch $f^*$ đáng kể | Kiểm chứng bằng số theo mục 6, không tin suy luận trên giấy |
| Mất kết quả khi chạy dài bị gián đoạn | Máy ngủ hoặc notebook bị ngắt giữa chừng | Mỗi nhóm thí nghiệm ghi ra một file JSON riêng ngay khi xong, hàm chạy bỏ qua nhóm đã có file |
| Hàm mục tiêu bị đổi giữa chừng | Các biểu đồ không so sánh được với nhau | Chốt `data/processed/` và `problem_config.json` từ tuần 1, không sửa về sau |

---

## 13. Ghi chép quá trình thực hiện

Mục này điền dần khi thực tế lệch khỏi kế hoạch, kèm lý do. Nhóm cần đọc lại mục này trước khi thuyết trình, vì câu hỏi của hội đồng thường rơi đúng vào những chỗ đã lệch.

Ba chỗ còn phải ghi vào đây khi chạy tới:

- Số cột thật sau khi mã hóa, và phổ trị riêng của $\frac{1}{n}X^{\top}X$. Hai con số này quyết định mục 5.6 nặng hay nhẹ.
- Bộ ba $L$, $\mu$, $\kappa$ ở hai quy mô mẫu, để xác nhận cách chia quy mô ở mục 3.5 hợp lệ.
- Thứ hạng thực tế giữa Newton và AGD theo trục thời gian, đối chiếu với dự đoán ở mục 1.2.

### 13.1. Môi trường và cấu hình máy đo

Máy đo là Apple M1 Pro, 8 nhân, 16 GB RAM, chạy Python 3.14.4 với numpy 2.5.2 trên nền BLAS Apple Accelerate. Bảng phiên bản đầy đủ nằm ở `README.md`, và mọi số liệu thời gian trong báo cáo phải đo trên đúng máy này.

Kế hoạch dự định cài `kaggle` để tải dữ liệu, nhưng hai file nén đã có sẵn trong `data/raw/` nên gói này bị bỏ khỏi `requirements.txt`. Cách tải bằng API vẫn ghi trong `README.md` cho thành viên nào cần dựng lại từ đầu.

### 13.2. Ba số đo xác nhận cách chia quy mô ở mục 3.5

Chạy thử trên dữ liệu ngẫu nhiên cùng kích thước với bài toán thật, $d = 150$, cho các số sau:

| Đại lượng | $n = 200$ nghìn | $n = 1{,}2$ triệu |
|---|---|---|
| Bộ nhớ của $X$ | 0,24 GB | 1,44 GB |
| Một lần tính gradient | 23 ms | 146 ms |
| GD chạy 5000 vòng | 1,9 phút | 12,1 phút |
| Lưới 8 độ dài bước của mục 5.2 | 15 phút | 1,6 giờ |

Các số này khớp với ước tính trong mục 3.5, nên cách chia hai quy mô giữ nguyên. Riêng phần SGD 200 epoch chưa đo, và đó mới là phần nặng nhất của lưới.

### 13.3. Newton rẻ hơn dự đoán, vì Hessian tính hết 0,2 giây

Tính $\frac{1}{n}X^{\top}X$ ở quy mô toàn phần mất 0,2 giây, tức bằng chi phí của 1,4 lần tính gradient chứ không phải hàng chục lần. Nguyên nhân là phép nhân ma trận với ma trận thuộc nhóm BLAS bậc ba, đọc mỗi phần tử một lần rồi dùng lại trong bộ nhớ đệm, trong khi tính gradient phải quét hết 1,44 GB hai lượt và bị chặn bởi băng thông bộ nhớ. Cộng thêm phân rã Cholesky của ma trận $150 \times 150$, một vòng Newton đầy đủ tốn chừng 0,35 giây, tương đương 2,4 vòng gradient descent.

Con số đó củng cố dự đoán ở mục 1.2: Newton hội tụ sau một vòng, nên nếu GD cần quá ba vòng để đạt cùng độ chính xác thì Newton đã thắng theo trục thời gian. Kết luận này đảo chiều khi $d$ tăng, vì chi phí Hessian tỷ lệ với $d^2$ còn chi phí gradient tỷ lệ với $d$; với $d$ khoảng vài nghìn thì thứ hạng quay về đúng như đề bài dự đoán.

### 13.4. Quan hệ $\alpha_{\text{sklearn}} = \lambda n$ đã kiểm chứng bằng số

Chạy `Ridge(alpha=lam*n, fit_intercept=False, solver='cholesky')` trên dữ liệu ngẫu nhiên rồi cắm nghiệm vào hàm $f$ của nhóm cho $f(\hat{w}) - f^* = 0$ đúng bằng $0$ trong `float64`, và $\|\hat{w} - w^*\| = 9 \cdot 10^{-15}$. Phép quy đổi ở mục 6 vì thế đúng, và nhóm chỉ cần chạy lại phép kiểm này trên dữ liệu thật để xác nhận.

Cùng lúc đó, nghiệm đóng qua `cho_factor` và `cho_solve` cho $\|\nabla f(w^*)\| = 1{,}5 \cdot 10^{-14}$, đạt mức sai số máy như phép kiểm thứ ba ở mục 8 yêu cầu.

### 13.5. Sàn số học của biểu đồ hội tụ bị hạ tám bậc

Kế hoạch dự tính vẽ $f(w_k) - f^*$ bằng cách trừ trực tiếp, và cảnh báo rằng đường cong chạm sàn ở cỡ $10^{-16} f^*$. Vì $f$ là hàm bậc hai và gradient triệt tiêu tại $w^*$, khai triển Taylor không còn số hạng dư, nên

$$
f(w) - f^* = \tfrac{1}{2} (w - w^*)^{\top} \nabla^2 f \, (w - w^*)
$$

đúng chính xác chứ không phải xấp xỉ. Hàm `RidgeObjective.suboptimality` tính theo vế phải.

Chênh lệch đo được trên bài toán thử $n = 400$, $d = 12$: tại $\|w - w^*\| \approx 10^{-8}$, phép trừ trực tiếp chỉ còn đúng ba chữ số, còn tại $10^{-12}$ nó trả về $-1{,}4 \cdot 10^{-17}$, tức nhiễu mang dấu âm, trong khi dạng toàn phương cho $3{,}7 \cdot 10^{-24}$. Trục tung của mọi hình hội tụ vì thế xuống được sâu hơn tám bậc so với dự tính ban đầu.

Tiêu chí dừng theo đình trệ ở mục 4.3 vẫn cần giữ, vì bản thân dãy $w_k$ vẫn ngừng dịch chuyển ở mức sai số máy; thứ được cải thiện là đại lượng đem vẽ, không phải chỗ thuật toán dừng.

### 13.6. Momentum của Nesterov dùng độ dài bước của vòng trước

Công thức $\beta = (1 - \sqrt{t\mu})/(1 + \sqrt{t\mu})$ cần $t$, nhưng $t$ chỉ biết sau khi line search chạy xong, còn $\beta$ lại cần có trước để dựng điểm ngoại suy $y_k$. Vòng lặp vì thế không thể dùng $t$ của chính vòng đó.

Cách xử lý trong `Nesterov`: `accept` lưu lại $t$ vừa được chấp nhận, và `propose` của vòng kế tiếp dùng giá trị đó, riêng vòng đầu lấy $1/L$. Với bước cố định thì hai giá trị trùng nhau nên không có sai khác nào; với backtracking thì $\beta$ chậm hơn một vòng so với $t$, và đó là cái giá phải trả để tránh lỗi ghép $\beta$ tính cho $1/L$ với một bước lớn hơn hẳn.

### 13.7. Bước khởi tạo của line search quyết định chi phí, không phải $c$ hay $\rho$

Chạy thử GD kèm backtracking trên bài toán $n = 20000$, $d = 40$, $\kappa = 1{,}2$, giữ nguyên $c = 10^{-4}$ và $\rho = 0{,}5$, chỉ đổi $t_0$:

| $t_0$ | Số vòng lặp | Lần đánh giá hàm mỗi vòng | Tổng lần đánh giá |
|---|---|---|---|
| $1/L$ | 11 | 1,00 | 11 |
| $10/L$ | 17 | 4,00 | 68 |
| $1$ (bằng $1{,}08/L$) | 34 | 18,21 | 619 |

Hàng cuối trông ngược đời vì $t_0 = 1$ gần $1/L$ hơn $10/L$, nhưng nó phơi ra hiện tượng đã nêu ở mục 4.3: khi $f(w_k) - f^*$ xuống dưới độ phân giải của $f$, mức giảm thật nhỏ hơn nhiễu làm tròn nên mọi bước thử đều trượt điều kiện Armijo, và line search lùi hết 18 lần ở mỗi vòng cuối. Bảng vì thế trộn hai hiệu ứng: chi phí dò bước ở giai đoạn đầu, và chi phí chết ở phần đuôi. Chương 4 của báo cáo cần tách hai phần đó ra, chẳng hạn bằng cách vẽ số lần đánh giá hàm theo vòng lặp thay vì chỉ báo cáo trung bình.

Kết luận dùng được cho việc chọn tham số: khởi tạo $t_0$ ở $1/L$ cho chi phí một lần đánh giá mỗi vòng, tức backtracking gần như miễn phí. Chi phí chỉ xuất hiện khi không biết $L$, và đúng lúc đó backtracking mới có lý do tồn tại.

### 13.8. Bài toán đã chốt: hai quy mô, $\kappa = 267{,}5$

Chạy `python -m src.dataset` cho kết quả sau, và từ đây `data/processed/` không được sửa nữa.

| Quy mô | $n$ | $d$ | $L$ | $\mu$ | $\kappa$ | $f^*$ | RMSE test |
|---|---|---|---|---|---|---|---|
| Quét tham số | 200 000 | 116 | 9,1156 | 0,034073 | 267,5 | 6,151114 | 3,4563 |
| Toàn phần | 1 200 000 | 116 | 9,0620 | 0,034036 | 266,2 | 6,142806 | 3,4646 |

Hệ số hiệu chỉnh chọn được là $\lambda = 0{,}031623$, so với $\lambda = 0{,}001778$ nếu lấy điểm cực tiểu của đường cong cross-validation. Quy tắc một sai số chuẩn vì thế đổi 18 lần về độ lớn $\lambda$ lấy chênh lệch sai số 12,0607 so với 12,0430, tức 0,15 phần trăm.

Ba hằng số ở hai quy mô lệch nhau 0,59 phần trăm với $L$, 0,11 phần trăm với $\mu$ và 0,48 phần trăm với $\kappa$, đều dưới ngưỡng 10 phần trăm mà mục 3.5 đặt ra. Cách chia hai quy mô do đó hợp lệ: cấu hình tìm được trên mẫu 200 nghìn dùng lại được cho mẫu 1,2 triệu mà không phải quét lại.

Kiểm chứng nghiệm đóng trên dữ liệu thật: $\|\nabla f(w^*)\| = 7{,}4 \cdot 10^{-13}$ ở quy mô nhỏ và $5{,}7 \cdot 10^{-12}$ ở quy mô toàn phần, đạt mức sai số máy như phép kiểm thứ ba của mục 8 yêu cầu.

### 13.9. Mức rò rỉ của `sub_grade` đo được là 0,955

Mục 3.2 dự đoán `grade` và `sub_grade` xác định gần trọn `int_rate`. Đo trên toàn bộ 2 260 668 dòng: hồi quy `int_rate` theo riêng `sub_grade` cho $R^2 = 0{,}9554$, theo riêng `grade` cho $R^2 = 0{,}9088$. Xét trong phạm vi một năm, nơi bảng tra lãi suất của Lending Club không đổi, con số lên tới $R^2 = 0{,}9834$ với dữ liệu năm 2016.

Sau khi loại hai cột đó cùng `installment` và nhóm cột ghi nhận sau giải ngân, mô hình đạt $R^2 \approx 0{,}486$ trên tập kiểm tra. Chênh lệch giữa 0,955 và 0,486 là thước đo trực tiếp của phần rò rỉ, và là con số nên đưa vào chương 2 của báo cáo thay vì chỉ nói rằng đã loại cột.

### 13.10. Một hướng suy biến do `fico_range_high` gây ra

Lần dựng ma trận đầu tiên cho $d = 117$ với trị riêng nhỏ nhất của ma trận Gram bằng $1{,}07 \cdot 10^{-7}$, kéo theo $\mu = \lambda$ đúng tới sáu chữ số và $\kappa = L/\lambda$. Vector riêng ứng với trị riêng đó có hai hệ số $\pm 0{,}7071$ đặt tại `fico_range_low` và `fico_range_high`, tức đúng hiệu hai cột.

Nguyên nhân nằm ở cách Lending Club ghi điểm tín dụng: `fico_range_high` bằng `fico_range_low` cộng 4 ở 2 260 227 trên 2 260 668 dòng, hệ số tương quan 0,99999991. Hai cột vì thế trùng nhau sau khi chuẩn hóa. Đã loại `fico_range_high` cùng nhóm với `funded_amnt`, và $d$ giảm còn 116.

Sau khi loại, trị riêng nhỏ nhất là $2{,}45 \cdot 10^{-3}$, nên $\mu = 0{,}0341$ vẫn do $\lambda = 0{,}0316$ chi phối tới 93 phần trăm nhưng không còn suy biến. Kết luận cho mục 5.6: $\kappa$ ở bài này gần như tỷ lệ nghịch với $\lambda$, nên thí nghiệm về ảnh hưởng của $\lambda$ sẽ cho hình rõ, và đó là lý do phải giữ nguyên thí nghiệm ấy thay vì lược bớt.

### 13.11. $L_{\max}$ lớn hơn $L$ hai vạn lần, nhưng cận cho SGD vẫn dùng được

Đo trên bài toán đã chốt: $L = 9{,}1156$ còn $L_{\max} = 200\,056$, tức gấp 21 947 lần. Thủ phạm là một dòng duy nhất có $\|x_i\|^2 = 200\,056$, trong khi trung vị là 82,2 và phân vị 99 là 560,9. Tọa độ lớn nhất của dòng đó nằm ở cột one-hot `addr_state=other` với giá trị 447 độ lệch chuẩn: chuẩn hóa một cột chỉ dấu có tỷ lệ rất nhỏ sẽ biến mỗi giá trị $1$ thành một đỉnh nhọn.

Hệ quả thứ nhất, phép thử dùng chung một độ dài bước cho mọi kích thước lô phân kỳ ở **tất cả** các lô, kể cả $B = 2048$, vì $L_B = 106{,}8$ vẫn lớn hơn $L$ hơn mười lần. Minh chứng mà mục 4.3 muốn nêu vì thế còn mạnh hơn dự kiến, nhưng hình sẽ không có bậc chuyển tiếp mà chỉ có một khối cùng phân kỳ.

Hệ quả thứ hai thì ngược với lo ngại ban đầu. Dù $L_B$ bị một dòng ngoại lai kéo lên, bước $1/L_B = 0{,}001265$ tại $B = 256$ lại nằm đúng vùng tốt nhất đo bằng thực nghiệm: quét $\eta_0$ trên lưới logarit cho sai số cuối 1,76·10⁻³ tại $\eta_0 = 10^{-3}$, 0,147 tại $3 \cdot 10^{-3}$, và phân kỳ từ $10^{-2}$ trở lên. Cận lý thuyết do đó vẫn đáng tính, và nhóm `batch-eta` là bằng chứng cho phát biểu ấy.

### 13.12. Thứ hạng sơ bộ trên bài toán thật

Chạy thử hai nhóm ở quy mô quét tham số, $n = 200\,000$, $\kappa = 267{,}5$:

| Cấu hình | Trạng thái | Số vòng lặp | $f - f^*$ | Thời gian |
|---|---|---|---|---|
| Newton, $t = 1$ | hội tụ | 1 | 0 | 0,07 s |
| Newton, Hessian dùng lại | hội tụ | 1 | 0 | 0,03 s |
| AGD, $\beta$ theo $t$ và $\mu$ | hội tụ | 286 | 1,8·10⁻¹⁸ | 4,14 s |
| AGD, $\beta = 0{,}9$ | hội tụ | 314 | 1,7·10⁻¹⁸ | 4,53 s |
| AGD, $\beta_k = (k-1)/(k+2)$ | đình trệ | 894 | 2,1·10⁻¹¹ | 12,75 s |
| GD, $t = 1/L$ | chạm giới hạn | 3000 | 2,9·10⁻¹⁵ | 42,84 s |

Dự đoán ở mục 1.2 đúng và biên độ lớn hơn dự kiến: Newton nhanh hơn GD khoảng 600 lần theo thời gian thực, vì nó cần một vòng còn GD chưa hội tụ sau 3000 vòng.

Một chi tiết đáng nói trong báo cáo: $\beta = 0{,}9$ mà đề bài gợi ý gần như tối ưu ở đây, chỉ chậm hơn 10 phần trăm so với công thức lý thuyết. Lý do là $\kappa = 267{,}5$ cho $(\sqrt{\kappa}-1)/(\sqrt{\kappa}+1) = 0{,}885$, tình cờ sát 0,9. Mục 1.2 nêu rằng hai giá trị trùng nhau khi $\kappa \approx 361$; con số thật lệch khỏi mốc đó nhưng công thức momentum không nhạy quanh cực trị, nên kết luận cần phát biểu thận trọng là hằng số đoán may mắn đúng cho bài này chứ không đúng nói chung.

### 13.13. Toàn bộ lưới đã chạy: 81 lần chạy, 24 phút

Chạy `python -m src.experiment --scale sweep` trên bài toán 200 nghìn điểm. Kết quả thô nằm ở `results/raw/`, mười file JSON theo nhóm cộng nhật ký chạy. Nhóm nặng nhất là `step-armijo` với 687 giây, kế đến `step-fixed` với 346 giây; các nhóm SGD chỉ mất 15 tới 48 giây vì mỗi vòng lặp rẻ hơn nhiều.

Thứ hạng ở cấu hình tốt nhất của từng phương pháp, đo bằng thời gian để đạt $f - f^* < 10^{-6}$:

| Phương pháp | Cấu hình tốt nhất | Số vòng lặp | Thời gian |
|---|---|---|---|
| Newton | Hessian dùng lại, $t = 1$ | 1 | 0,03 s |
| AGD | $\beta$ theo $t$ và $\mu$, có restart | 75 | 1,05 s |
| GD, backtracking | $c = 0{,}3$, $\rho = 0{,}5$, $t_0 = 1$ | 143 | 3,14 s |
| GD, bước cố định | $t = 1{,}9/L$ | 331 | 4,75 s |
| SGD | $B = 2048$, $\eta = 1/L_B$ | không đạt | dừng ở $4{,}9 \cdot 10^{-4}$ |

### 13.14. Bước tối ưu theo lý thuyết thua bước $1{,}9/L$, và vì sao

Kết quả trái với dự kiến ở mục 5.5: $t = 2/(L+\mu)$ cần 766 vòng để đạt $10^{-6}$, còn $t = 1{,}9/L$ chỉ cần 331. Lý thuyết không sai, nhưng nó trả lời một câu hỏi khác.

Cận $\left(\frac{\kappa-1}{\kappa+1}\right)^{2k}$ là cận **cực tiểu hóa trường hợp xấu nhất trên mọi điểm khởi tạo**, nên bước tối ưu cân bằng hai đầu phổ: với $t = 2/(L+\mu)$ thì $|1 - tL| = 0{,}9926$ và $|1 - t\mu| = 0{,}99255$, gần bằng nhau. Bước $1{,}9/L$ hy sinh một chút ở đầu nhỏ, $|1-t\mu| = 0{,}99290$, để đổi lấy $|1 - tL| = 0{,}9000$ ở đầu lớn.

Điểm khởi tạo thật không phải trường hợp xấu nhất. Phân tích sai số ban đầu tại $w_0 = 0$ theo cơ sở riêng cho thấy các hướng có $\lambda > 1$ mang 80,5 phần trăm của $f(w_0) - f^*$, trong khi các hướng có $\lambda < 0{,}1$ chỉ mang 0,3 phần trăm. Nguyên nhân nằm ở chính dạng toàn phương: $f(w) - f^* = \frac{1}{2}\sum_i \lambda_i c_i^2$ nên mỗi hướng được cân theo trị riêng của nó, và hướng ứng với $\lambda = 3{,}76$ một mình chiếm 35,9 phần trăm.

Mô hình modal dự đoán khớp tới hai chữ số: $t = 1{,}9/L$ cho sai số $9{,}80 \cdot 10^{-7}$ ở vòng 331, còn $t = 2/(L+\mu)$ cho $9{,}95 \cdot 10^{-7}$ ở vòng 766. Kết luận đảo chiều khi cần độ chính xác rất cao, vì lúc đó mọi hướng lớn đã tắt và tốc độ do hướng $\mu$ quyết định, nơi $2/(L+\mu)$ nhỉnh hơn.

### 13.15. Backtracking thắng bước cố định trên cả hai trục

Kế hoạch dự đoán ở mục 4.3 rằng backtracking thắng về số vòng lặp nhưng có thể thua về thời gian, vì phải trả thêm số lần đánh giá hàm. Đo được thì nó thắng cả hai: 143 vòng và 3,14 giây, so với 331 vòng và 4,75 giây của bước cố định tốt nhất, dù tốn 4,9 lần đánh giá hàm mỗi vòng.

Cơ chế nằm ở chỗ điều kiện Armijo không so với $L$ toàn cục mà so với thương Rayleigh tại chỗ. Bước được chấp nhận thỏa $t \le 2(1-c)\|g\|^2 / (g^{\top} H g)$; khi gradient nghiêng về các hướng phẳng thì mẫu số nhỏ và line search nhận bước lớn hơn $2/L$ rất nhiều. Bước cố định không có cách nào làm được điều đó vì nó phải an toàn cho hướng dốc nhất ở mọi vòng.

Hai tham số không tương đương nhau về mức ảnh hưởng. Toàn bộ sáu cấu hình $\rho = 0{,}5$ đứng trên toàn bộ mười hai cấu hình $\rho = 0{,}8$ và $\rho = 0{,}9$, trong khi $c$ đổi từ $10^{-4}$ lên $0{,}3$ chỉ làm số vòng nhích từ 167 xuống 143. Lý do là $\rho$ quyết định độ thô của lưới bước thử: $\rho = 0{,}9$ cần gấp bảy lần số lần thử để đi từ $t_0$ xuống cùng một bước, và mọi lần thử đó đều tốn một lần tính $f$.

### 13.16. Restart chỉ cứu được công thức momentum tăng dần

Với $\beta$ tính theo $t$ và $\mu$, cũng như với $\beta = 0{,}9$, kết quả có và không restart trùng nhau tới từng chữ số, vì điều kiện restart không kích hoạt lần nào. Với $\beta_k = (k-1)/(k+2)$ thì restart đưa trạng thái từ đình trệ ở sai số $2{,}1 \cdot 10^{-11}$ về hội tụ, và số vòng để đạt $10^{-6}$ giảm từ 143 xuống 89.

Kết luận nên phát biểu theo điều kiện chứ không nói chung: restart là cơ chế bù cho việc không biết $\mu$. Khi đã biết $\mu$ và đặt $\beta$ theo nó, momentum không bao giờ vượt đà nên restart không có việc gì để làm.

### 13.17. Chuẩn hóa hạ $\kappa$ mười bậc, và công nằm ở phép chia chứ không ở phép trừ

Mục 5.7 dự tính so hai cách, chưa chuẩn hóa và đã chuẩn hóa. Chạy thêm một biến thể trung gian chỉ trừ trung bình mà không chia độ lệch chuẩn thì tách được hai tác dụng vốn hay bị gộp làm một:

| Cách mã hóa | $L$ | $\mu$ | $\kappa$ | Vòng lặp đạt $10^{-6}$ |
|---|---|---|---|---|
| Thô | $1{,}225 \cdot 10^{11}$ | 0,031628 | $3{,}87 \cdot 10^{12}$ | không đạt sau 4000 |
| Chỉ trừ trung bình | $6{,}039 \cdot 10^{10}$ | 0,031628 | $1{,}91 \cdot 10^{12}$ | không đạt sau 4000 |
| Chuẩn hóa đầy đủ | 9,1156 | 0,034073 | 267,5 | 630 |

Trừ trung bình chỉ hạ $\kappa$ đúng hai lần, còn chia độ lệch chuẩn hạ tiếp $7 \cdot 10^{9}$ lần. Tổng cộng $1{,}45 \cdot 10^{10}$ lần. Nguyên nhân là các cột đứng ở những thang khác hẳn nhau, `tot_cur_bal` cỡ $10^6$ bên cạnh `dti` cỡ hàng chục, nên trị riêng lớn nhất của ma trận Gram bị chi phối bởi cột có đơn vị lớn nhất; trừ đi trung bình không đụng gì tới chênh lệch đó.

Ở hai biến thể chưa chia thang, $\mu$ bằng đúng $\lambda$ tới năm chữ số, tức ma trận Gram suy biến về mặt số học khi đặt cạnh $L \sim 10^{11}$. Sau 4000 vòng, GD mới hạ được sai số từ 5,53 xuống 3,74, tức chưa tới một phần ba. Đây là hình cho thấy tiền xử lý không tách rời khỏi tối ưu hóa.

### 13.18. Tốc độ quan sát được khớp cận lý thuyết tới hai chữ số

Chạy GD và AGD ở sáu giá trị $\lambda$ trên cùng bộ dữ liệu, mỗi lần dùng bước tính theo $L$ và $\mu$ của chính bài toán đó:

| $\lambda$ | $\kappa$ | GD, vòng lặp đạt $10^{-6}$ | AGD, vòng lặp | RMSE test |
|---|---|---|---|---|
| 0,001 | 2633,0 | không đạt sau 4000 | 209 | 3,4537 |
| 0,01 | 730,4 | 2091 | 118 | 3,4541 |
| 0,03162 | 267,5 | 766 | 75 | 3,4563 |
| 0,1 | 89,6 | 257 | 48 | 3,4690 |
| 1 | 10,1 | 29 | 17 | 3,7270 |
| 10 | 1,9 | 5 | 6 | 4,4578 |

Lý thuyết nói số vòng của GD tỷ lệ với $\kappa$ còn của AGD tỷ lệ với $\sqrt{\kappa}$. Đối chiếu từng cặp liền kề:

| $\kappa$ giảm | Tỷ lệ $\kappa$ | GD, tỷ lệ quan sát | AGD, tỷ lệ quan sát | $\sqrt{\kappa}$ dự đoán |
|---|---|---|---|---|
| 730 xuống 268 | 2,73 | 2,73 | 1,57 | 1,65 |
| 268 xuống 90 | 2,98 | 2,98 | 1,56 | 1,73 |
| 90 xuống 10 | 8,91 | 8,86 | 2,82 | 2,99 |

Cột GD khớp gần như hoàn hảo. Cột AGD bám sát $\sqrt{\kappa}$ nhưng nhỉnh hơn dự đoán một chút, do hằng số trước lũy thừa trong cận tăng tốc không phải hằng số theo $\kappa$.

Đánh đổi mà mục 5.6 muốn nêu hiện ra rõ: đi từ $\lambda = 0{,}001$ lên $\lambda = 0{,}1$ làm RMSE xấu đi 0,44 phần trăm, từ 3,4537 lên 3,4690, đổi lại $\kappa$ giảm 29 lần và GD từ chỗ không hội tụ nổi sau 4000 vòng xuống còn 257 vòng. Giá trị $\lambda = 0{,}03162$ mà quy tắc một sai số chuẩn chọn ở mục 3.6 nằm đúng trong khoảng đó.

### 13.19. Quy mô toàn phần xác nhận cách chia ở mục 3.5

Chạy nhóm `headline` ở $n = 1\,200\,000$ mất 13,5 phút, kết quả ghi ở `results/raw/full/` để không đè bản quy mô nhỏ.

| Cấu hình | Vòng, 200k | Vòng, 1,2M | Giây, 200k | Giây, 1,2M | Tỷ lệ thời gian |
|---|---|---|---|---|---|
| GD, $t = 1{,}9/L$ | 2022 | 1959 | 27,52 | 158,67 | 5,77 |
| GD, backtracking | 561 | 514 | 13,86 | 71,99 | 5,19 |
| AGD | 286 | 283 | 3,92 | 23,49 | 6,00 |
| SGD, $B = 2048$ | 3880 | 23400 | 2,02 | 13,17 | 6,52 |
| Newton, Hessian dùng lại | 1 | 1 | 0,03 | 0,16 | 5,74 |

Số vòng lặp gần như không đổi với các phương pháp tất định, đúng như dự đoán, vì $\kappa$ chỉ nhích từ 267,5 xuống 266,2. Thời gian tăng từ 5,2 tới 6,5 lần khi $n$ tăng 6 lần, tức tuyến tính theo $n$ đúng như mô hình chi phí $\mathcal{O}(nd)$. Riêng SGD tăng số vòng đúng 6 lần vì ngân sách đặt theo epoch, mà một epoch ở quy mô lớn có nhiều lô hơn.

Thứ hạng không đổi giữa hai quy mô, nên phần quét lưới trên mẫu nhỏ là hợp lệ.

### 13.20. Con số 0,16 giây của Newton bỏ sót phần dựng Hessian

Cần đính chính một chỗ trong bảng trên. Hàm `warm_up` gọi `compute_hessian` và `w_star` trước khi bấm đồng hồ, nên biến thể `reuse_factorization=True` nhận sẵn phân rã Cholesky mà không phải trả tiền. Con số 0,16 giây vì thế đo tình huống "Hessian đã có sẵn", không phải toàn bộ chi phí của phương pháp Newton.

Chi phí thật, đo riêng từng phần ở quy mô toàn phần:

| Thành phần | Thời gian |
|---|---|
| Dựng $\frac{1}{n}X^{\top}X$ | 163,2 ms |
| Phân rã Cholesky $116 \times 116$ | 0,2 ms |
| Một lần tính gradient | 87,3 ms |

Một bước Newton đầy đủ do đó tốn khoảng 250 ms. Biến thể dựng lại Hessian mỗi vòng đo được 0,40 giây, cao hơn 250 ms vì vòng lặp gọi `propose` một lần nữa ở vòng thứ hai chỉ để kiểm tra điều kiện dừng, và với Newton thì `propose` kéo theo cả việc dựng lại Hessian. Đây là giới hạn của cách thiết kế `iterate`: điều kiện dừng cần gradient tại điểm mới, mà `Direction.propose` gói gradient chung với phần đắt tiền. Chỉ Newton chịu ảnh hưởng, vì với các phương pháp khác thì `propose` chính là một lần tính gradient.

Hệ quả cho mục 6: so sánh với `Ridge(solver='cholesky')` ở quy mô toàn phần phải đọc là 0,273 giây của thư viện so với khoảng 0,25 giây phần tính toán thật của nhóm, tức **hai bên làm cùng một việc và mất cùng một thời gian**, chứ không phải mã tự viết nhanh hơn thư viện. Kết luận này đúng hơn và cũng đáng nói hơn: `Ridge` với solver mặc định chính là nghiệm đóng qua Cholesky, nên trùng khớp là dấu hiệu cả hai cài đúng.

### 13.21. Hai cột không cần chuyển đổi như kế hoạch dự tính

Bản phát hành hiện có trên Kaggle đã lưu `int_rate` và `revol_util` ở kiểu `float64`, không phải chuỗi kèm ký hiệu phần trăm, nên bước 3 của mục 3.3 chỉ còn phải xử lý `term` và `emp_length`. Hai cột `earliest_cr_line` và `issue_d` vẫn ở dạng chuỗi `"Aug-2003"`, đúng như kế hoạch dự tính, nên bước 4 giữ nguyên.

---

## 14. Việc cần làm ngay

1. Tải `accepted_2007_to_2018Q4.csv.gz` từ Kaggle về `data/raw/`, rồi chuyển sang Parquet ngay sau lần đọc đầu.
2. Dựng môi trường ảo theo mục 2.
3. Chạy notebook khảo sát để xác nhận số dòng, tỷ lệ thiếu từng cột, và kiểu dữ liệu thật của `int_rate`, `term`, `revol_util`.
4. Chốt danh sách cột loại bỏ theo mục 3.2, và kiểm tra lại bằng cách hồi quy `int_rate` theo riêng `sub_grade` để đo mức rò rỉ bằng số.
5. Dựng ma trận thiết kế, tính $L$, $\mu$, $\kappa$, $f^*$. Ba con số đầu là đầu vào cho toàn bộ phần chọn tham số phía sau nên cần có sớm.
