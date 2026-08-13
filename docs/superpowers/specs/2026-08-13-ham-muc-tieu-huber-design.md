# Thiết kế: hàm mục tiêu Huber, mục 4.4 của kế hoạch

Ngày 13 tháng 8 năm 2026.

## 1. Mục tiêu và phạm vi

Bài toán Ridge có nghiệm đóng, nên Newton kết thúc sau đúng một vòng và
backtracking nhận ngay bước đầy đủ ở mọi vòng. Hai chương lõi của báo cáo vì thế
mất phần lớn nội dung mà chúng lẽ ra phải có. Mục 4.4 của `KE_HOACH_TRIEN_KHAI.md`
đề xuất cách gỡ: thêm một hàm mục tiêu phi tuyến trên cùng bộ dữ liệu, dùng mất
mát Huber kèm hiệu chỉnh Ridge,

$$
f_{\text{huber}}(w) = \frac{1}{n} \sum_{i=1}^{n} H_{\delta}\!\left( x_i^{\top} w - y_i \right) + \frac{\lambda}{2} \left\| w \right\|_2^2,
\qquad
H_{\delta}(r) =
\begin{cases}
\dfrac{r^2}{2}, & |r| \le \delta \\[2ex]
\delta \left( |r| - \dfrac{\delta}{2} \right), & |r| > \delta
\end{cases}
$$

Hàm này lồi, khả vi bậc một liên tục, và Hessian tồn tại ở mọi điểm mà không dòng
nào rơi đúng vào $|r_i| = \delta$.

**Trong phạm vi.** Một lớp hàm mục tiêu mới, bốn thuật toán đã có chạy trên nó ở
quy mô 200 nghìn, ba nhóm thí nghiệm, năm hình, một chương báo cáo, hai frame
slide, và phần cập nhật tương ứng trong kế hoạch.

**Ngoài phạm vi.** Mục 4.5 về thuật toán mở rộng đã bị bỏ theo quyết định ngày 13
tháng 8. Quy mô toàn phần 1,2 triệu cũng nằm ngoài: chương này trả lời câu hỏi về
dạng hàm chứ không về quy mô mẫu, và quy mô đã có chương riêng.

## 2. Ba số đo đã có trước khi thiết kế

Ba con số dưới đây lấy từ một phép thăm dò chạy trên `X_train.npy` ở quy mô 200
nghìn, mã bỏ đi sau khi đo. Chúng quyết định hình dạng của chương, nên ghi lại ở
đây thay vì để người thực hiện phải đo lại.

| Đại lượng | Giá trị | Hệ quả cho thiết kế |
| --- | --- | --- |
| Phần dư của nghiệm Ridge, $\hat\sigma$ theo MAD | 3,02 | $\delta = 1{,}345\hat\sigma = 4{,}07$ |
| Tỷ lệ dòng nằm ngoài vùng bậc hai tại $\delta$ đó | 0,199 | Hàm thật sự không còn là hàm bậc hai |
| Số vòng Newton từ $w_0 = 0$ tới sai số máy | 8 | Đuôi bậc hai hiện rõ, $10^{-2} \to 10^{-5} \to 10^{-11}$ |

Phép thăm dò còn cho hai kết quả ngược với dự đoán của mục 4.4, và cả hai đều
phải vào báo cáo đúng như đo được.

Backtracking **không** lùi bước một lần nào với hướng Newton: cả 8 vòng đều nhận
ngay $t = 1$. Dự đoán rằng hàm phi tuyến sẽ làm line search có việc chỉ đúng với
gradient descent, nơi Armijo hạ bước xuống 0,25 và tốn trung bình 2,81 lần đánh
giá hàm mỗi vòng. Cơ chế: hướng Newton đã mang sẵn thông tin độ cong nên bước
đầy đủ gần như luôn thỏa điều kiện giảm đủ, còn hướng gradient thì không.

Sàn số học của phép trừ nằm ở khoảng $10^{-15}$, vì $f^* \approx 5{,}196$. Gap đo
bằng phép trừ thẳng tụt từ $6{,}7 \cdot 10^{-11}$ xuống đúng 0 chỉ sau một vòng,
tức mất chặng cuối của đuôi bậc hai. Mục 5 dưới đây xử lý chỗ này.

## 3. Kiến trúc

### 3.1. Phương án chọn

`iterate` hiện gọi hàm mục tiêu qua đúng một bề mặt hẹp, gồm `value_and_gradient`,
`suboptimality`, `gradient`, `compute_hessian`, và các hằng số `L`, `mu`, `kappa`,
`f_star`, `n`, `d`. Bề mặt đó không có gì riêng của hàm bậc hai, nên vòng lặp
không phải sửa một dòng nào.

Thiết kế vì thế tách một `Protocol` tên `SmoothObjective` mô tả đúng bề mặt ấy,
đặt trong `src/objective.py` cạnh `LocalObjective` đã có, rồi để `RidgeObjective`
giữ nguyên toàn bộ đường đi nghiệm đóng của nó. Lớp mới `HuberObjective` cài cùng
bề mặt, đặt trong file riêng `src/huber.py`.

Hai phương án còn lại đã cân nhắc và loại. Gộp hai hàm mục tiêu vào một lớp cơ sở
tham số hóa theo mất mát thì gọn về mặt lặp mã, nhưng phá ba thứ làm số liệu Ridge
đáng tin: dạng toàn phương chính xác cho `suboptimality`, ma trận Gram và nhân tử
Cholesky cached, và lập luận Hessian hằng của Newton. Cả ba là đặc quyền của hàm
bậc hai và sẽ phải đặc-biệt-hóa ngược lại trong lớp cơ sở. Viết `HuberObjective`
độc lập mà không khai báo `Protocol` thì chạy được nhờ duck typing, nhưng
`direction.py` vẫn khai kiểu `RidgeObjective` trong khi thực tế nhận cả hai, tức
chú thích kiểu nói sai và `pyright` sẽ không bắt được lỗi nào.

### 3.2. Thay đổi giao diện

Hessian của Huber phụ thuộc $w$, nên chữ ký phải đổi trên cả hai lớp:

```python
def compute_hessian(self, w: np.ndarray | None = None) -> np.ndarray: ...
```

`RidgeObjective` bỏ qua đối số và ghi trong docstring rằng Hessian của hàm bậc hai
là hằng, nên đó chính là chỗ `reuse_factorization=True` hợp lệ. Ba chỗ gọi phải
sửa theo: `NewtonStep.propose` trong `src/direction.py:259`, `warm_up` trong
`src/iterate.py:54`, và `tests/test_objective.py:36`.

`solve_hessian(rhs)` giữ nguyên chữ ký. Với Ridge nó dùng nhân tử Cholesky cached
của Hessian hằng. Với Huber, nhân tử được dựng ở lần gọi đầu tiên rồi giữ nguyên,
tức `reuse_factorization=True` trên Huber biến Newton thành phương pháp dây cung:
vẫn là hướng giảm nhưng mất hội tụ bậc hai. Đó là một đường cong đáng vẽ và gần
như không tốn thêm mã, nên giữ lại chứ không chặn.

`max_row_smoothness` và `batch_smoothness` chỉ dùng `.X`, `.lam`, `.L`, nên chỉ
cần nới chú thích kiểu sang `SmoothObjective`. Cận $L_{\max} = \max_i \lVert x_i \rVert^2 + \lambda$
vẫn là cận trên hợp lệ cho Huber, vì trọng số từng dòng của Hessian Huber nằm
trong $[0, 1]$.

`experiment.py` nới kiểu của `MakeDirection`, `MakeStep` và `ExperimentGroup.build`
sang `SmoothObjective`.

### 3.3. Bề mặt của `HuberObjective`

```python
class HuberObjective:
    def __init__(self, X, y, lam, delta): ...

    # values
    n_rows: int
    def value(self, w) -> float: ...
    def gradient(self, w) -> np.ndarray: ...
    def value_and_gradient(self, w) -> tuple[float, np.ndarray]: ...

    # curvature
    def compute_hessian(self, w=None) -> np.ndarray: ...
    def solve_hessian(self, rhs) -> np.ndarray: ...   # chord: factor frozen at first call

    # constants and optimum
    L: float          # lambda_max(X^T X / n) + lam, shared with Ridge
    mu: float         # lam, the bound that holds without knowing the solution
    mu_at_optimum: float
    kappa: float
    w_star: np.ndarray
    f_star: float

    def suboptimality(self, w) -> float: ...
    def batch(self, indices) -> HuberBatchObjective: ...
    def summary(self) -> dict: ...
```

Gradient và Hessian viết theo dạng trọng số dòng. Với $r = Xw - y$,

$$\nabla f(w) = \frac{1}{n} X^{\top} \psi(r) + \lambda w, \qquad \psi(r_i) = \operatorname{clip}(r_i, -\delta, \delta)$$

$$\nabla^2 f(w) = \frac{1}{n} X^{\top} \operatorname{diag}(s) X + \lambda I, \qquad s_i = \mathbb{1}\{|r_i| \le \delta\}$$

Dựng Hessian theo cách nhân trọng số vào từng dòng của $X$ sẽ cấp phát một bản sao
185 MB mỗi vòng ở quy mô 200 nghìn. Cách rẻ hơn là lấy chỉ số các dòng trong vùng
bậc hai rồi tính $X_S^{\top} X_S$, vì $s_i$ chỉ nhận giá trị 0 hoặc 1. Người thực
hiện đo cả hai rồi giữ cách nhanh hơn, và ghi con số vào mục 13 của kế hoạch.

### 3.4. Ba hằng số

Vì $0 \le H_{\delta}'' \le 1$, Hessian Huber luôn nằm dưới Hessian Ridge theo thứ
tự nửa xác định. Hai hằng số vì thế suy ra được mà không phải giải thêm bài toán
trị riêng nào:

$$L_{\text{huber}} \le \lambda_{\max}\!\left(\tfrac{1}{n}X^{\top}X\right) + \lambda = 9{,}1156, \qquad \mu_{\text{huber}} \ge \lambda = 0{,}0316$$

Cho $\kappa \le 288{,}3$, so với 267,5 của Ridge trên cùng dữ liệu.

Thuộc tính `mu` trả về cận $\lambda$ chứ không trả về trị riêng nhỏ nhất tại
$w^*$, vì `Nesterov` với quy tắc `strongly_convex` dùng `mu` để tính $\beta$ và
thuật toán không được phép biết trước nghiệm. Trị riêng nhỏ nhất tại $w^*$ vẫn
tính và lưu dưới tên `mu_at_optimum`, để báo cáo nói được cận $\lambda$ lỏng bao
nhiêu.

### 3.5. Nghiệm tham chiếu

$\delta$ tính một lần từ phần dư của nghiệm Ridge, theo $\delta = 1{,}345 \hat\sigma$
với $\hat\sigma$ là độ lệch chuẩn vững ước lượng qua MAD, cho giá trị 4,0666 ở quy
mô 200 nghìn. Giá trị ấy đông cứng lại cùng nghiệm tham chiếu và không tính lại ở
mỗi lần chạy, vì nguyên tắc ở mục 1 của kế hoạch bắt hàm mục tiêu phải chốt trước
khi bắt đầu thí nghiệm.

$w^*$ tìm bằng Newton có Armijo từ $w_0 = 0$, dừng khi $\lVert \nabla f \rVert \le 10^{-14}$.
Phép thăm dò đạt $2 \cdot 10^{-16}$ sau 8 vòng, nên ngưỡng này an toàn. Kết quả
lưu ra `data/processed/huber_reference.json` gồm `w_star`, `f_star`, `delta`,
`lam`, `n`, `d`, `grad_norm`, và số vòng đã dùng, để các lần vẽ sau khỏi giải lại.
File có sẵn thì đọc lại và kiểm tra rằng `delta`, `lam`, `n`, `d` khớp; lệch thì
giải lại chứ không dùng bừa.

## 4. Đo sai số

Huber không có nghiệm đóng nên không dùng được dạng toàn phương của mục 13.5 trong
kế hoạch. Phép trừ thẳng $f(w) - f^*$ thì chạm sàn ở $10^{-15}$ như mục 2 đã đo.
Cách giữ thêm được vài bậc là ghép cặp trước khi cộng:

$$f(w) - f^* = \frac{1}{n}\sum_{i=1}^{n}\left[ H_{\delta}(r_i) - H_{\delta}(r_i^*) \right] + \frac{\lambda}{2}(w - w^*)^{\top}(w + w^*)$$

Triệt tiêu xảy ra ở từng phần tử chứ không ở tổng cuối, và số hạng hiệu chỉnh viết
dưới dạng tích thay vì hiệu hai bình phương. Vector $r^* = Xw^* - y$ tính một lần
rồi giữ trong bộ nhớ, tốn 1,6 MB ở quy mô 200 nghìn.

Chi phí phải chấp nhận: mỗi lần gọi `suboptimality` là một lượt quét toàn bộ $X$,
tức $\mathcal{O}(nd)$, trong khi bản Ridge chỉ tốn $\mathcal{O}(d^2)$. Với SGD ghi
400 checkpoint thì riêng phần ghi tốn khoảng 9 GFLOP. Đồng hồ đã dừng trước khi
ghi nên số liệu thời gian không bị ảnh hưởng, chỉ thời gian chờ dài thêm.

Sàn còn lại sau khi ghép cặp sẽ đo và ghi vào báo cáo chứ không giấu. Nếu nó vẫn
cắt mất chặng cuối của đuôi bậc hai thì báo cáo ghi rõ đường Newton dừng ở sàn số
học chứ không dừng vì thuật toán hết khả năng.

## 5. Thí nghiệm

Hai nhóm, đặt trong `experiment.py` cạnh các nhóm đã có, tên có tiền tố `huber-`
nên không đụng file kết quả nào hiện tại. Cả hai chạy ở quy mô 200 nghìn với
$\lambda = 0{,}0316$ giữ nguyên như bài toán Ridge, để hai bài toán chỉ khác nhau
đúng ở dạng mất mát.

| Nhóm | Các lần chạy | Câu hỏi |
| --- | --- | --- |
| `huber-newton` | Newton $t=1$; Newton + Armijo; hai biến thể dùng lại Hessian | Newton cần bao nhiêu vòng, và dây cung mất gì |
| `huber-headline` | GD ($t = 1/L$), GD + Armijo, AGD, SGD ($B = 2048$), Newton | Thứ hạng bốn thuật toán có đổi so với Ridge không |

Hình về chi phí line search dựng từ bản ghi của hai nhóm trên chứ không cần nhóm
thứ ba, vì `RunRecord` đã lưu sẵn `fevals` và `cost_figure` trong `figures.py`
nhận thẳng danh sách bản ghi. Chạy lại cùng cấu hình lần nữa chỉ để đếm số lần
đánh giá hàm là tốn thời gian mà không thêm thông tin nào.

Nhóm `huber-headline` dùng đúng cấu hình tốt nhất mà nhóm `headline` đã chốt trên
Ridge, không quét lại lưới tham số. Lý do: câu hỏi ở đây là dạng hàm đổi thứ hạng
thế nào, và quét lại lưới sẽ trộn hai biến vào một hình.

Mỗi lần chạy trong `huber-headline` lặp ba lần lấy trung vị, giống nhóm `headline`,
vì hình theo trục thời gian là một nửa nội dung.

Năm hình, lưu ra `results/figures/` theo đúng quy ước hai định dạng của
`CLAUDE.md` mục 3:

- `huber-newton_iters`, `huber-newton_time`
- `huber-headline_iters`, `huber-headline_time`
- `huber-step_cost`

Bảng màu giữ nguyên bảng trong `figures.py`, vì `family_of` đã nhận diện đúng
`GD`, `SGD`, `AGD`, `Newton` từ nhãn.

Notebook mới `notebooks/09_huber.ipynb`, theo đúng khuôn của notebook 03 tới 08:
gọi hàm trong `src/`, chạy nhóm thí nghiệm, vẽ hình, không định nghĩa lại thuật
toán nào.

## 6. Báo cáo và slide

Chương mới đặt ngay sau chương 7 về thông tin bậc hai, tên "Khi hàm mục tiêu không
còn là hàm bậc hai". Các chương 8 tới 11 dịch số thành 9 tới 12; mọi tham chiếu
chéo đã dùng `\label` nên `\ref` tự cập nhật.

Trục lập luận của chương, bốn mục theo thứ tự:

1. Bài toán Huber và ba hằng số của nó, đặt cạnh ba hằng số của Ridge.
2. Newton mất đặc quyền một vòng nhưng được đuôi hội tụ bậc hai, kèm bảng gap
   từng vòng và đường dây cung để so.
3. Backtracking có việc với gradient descent mà không có việc với Newton, kèm số
   lần đánh giá hàm mỗi vòng của cả hai.
4. Thứ hạng bốn thuật toán trên hàm phi tuyến, đặt cạnh thứ hạng trên Ridge.

Hai frame slide, đặt sau frame về Newton: một frame cho đuôi bậc hai, một frame
cho chỗ backtracking lệch dự đoán.

Thêm mục Huber 1964 vào `report/refs.bib` và trích dẫn nó ở mục 1 của chương mới.
Quy tắc ở `CLAUDE.md` mục 5 cấm để mục nào trong `refs.bib` không được trích dẫn.

Toàn bộ phần văn xuôi viết theo `docs/van-phong-tieng-viet.md`, ba lượt gọi riêng
như mục 1 của file đó quy định.

## 7. Cập nhật `KE_HOACH_TRIEN_KHAI.md`

| Mục | Sửa gì |
| --- | --- |
| 4.4 | Viết lại theo cái đã làm, thay câu "phần này tùy chọn" bằng kết quả thật |
| 4.5 | Ghi rõ đã bỏ và lý do, giữ bảng ứng viên để người đọc thấy chỗ mở rộng được |
| 11 | Dòng "Áp dụng thêm thuật toán khác" trỏ về đúng mục 4.4, vì đáp ứng bằng hàm mục tiêu thứ hai chứ không bằng thuật toán thứ năm |
| 13 | Thêm mục cho chỗ backtracking không lùi bước với Newton, cho sàn số học của phép ghép cặp, và cho chi phí dựng Hessian |
| 14 | Cập nhật theo trạng thái hiện tại |

## 8. Kiểm thử

File mới `tests/test_huber.py`. Phép kiểm mạnh nhất đặt đầu tiên.

1. **$\delta$ rất lớn thì Huber trùng Ridge.** Với $\delta = 10^{6}$, `value`,
   `gradient` và `compute_hessian` phải khớp `RidgeObjective` tới sai số máy trên
   một bài toán nhỏ sinh ngẫu nhiên. Phép kiểm này bắt được gần như mọi lỗi hằng
   số và lỗi dấu.
2. **Gradient khớp sai phân hữu hạn** trên bài toán nhỏ, ở một điểm có cả dòng
   trong lẫn ngoài vùng bậc hai.
3. **Hessian khớp sai phân hữu hạn của gradient**, cùng điều kiện.
4. **Hessian nửa xác định dương** và trị riêng nhỏ nhất không nhỏ hơn $\lambda$.
5. **$\lVert \nabla f(w^*) \rVert$ ở mức sai số máy** sau khi giải nghiệm tham
   chiếu.
6. **Công thức ghép cặp khớp phép trừ thẳng** ở vùng gap lớn, nơi phép trừ còn
   chính xác, và cho giá trị nhỏ hơn ở vùng gap nhỏ.
7. **`suboptimality` không âm** tại một loạt điểm ngẫu nhiên.

Thêm vào `tests/test_objective.py` một phép kiểm rằng `RidgeObjective.compute_hessian`
nhận đối số $w$ rồi bỏ qua nó, vì Hessian của hàm bậc hai là hằng. Các test đã có
trong file đó không phải sửa: chúng đọc `objective.hessian`, và thuộc tính ấy vẫn
gọi `compute_hessian()` được nhờ đối số mới có giá trị mặc định. Chạy `pytest`
toàn bộ, và `pytest tests/test_report.py` sau khi sửa báo cáo.

## 9. Rủi ro

| Rủi ro | Dấu hiệu | Cách xử lý |
| --- | --- | --- |
| Ghép cặp không hạ được sàn đáng kể | Đường Newton vẫn tụt thẳng xuống 0 sau $10^{-11}$ | Ghi rõ sàn trong báo cáo, cắt trục ở $10^{-13}$, không đổi $\delta$ để làm đẹp hình |
| Dựng Hessian mỗi vòng quá chậm | Một vòng Newton trên 2 giây ở quy mô 200 nghìn | Dùng cách lấy chỉ số thay cho nhân trọng số, đo lại và ghi số vào mục 13 |
| Một dòng rơi đúng $\lvert r_i \rvert = \delta$ | Hessian nhảy giữa hai vòng, Newton dao động | Quy ước $s_i = 1$ tại biên, ghi quy ước vào docstring |
| SGD trên Huber tốn quá nhiều thời gian ghi checkpoint | Nhóm chạy trên 20 phút | Giảm số checkpoint qua `record_every`, không giảm số epoch |
| Chương mới làm lệch số chương trong slide | `pytest tests/test_report.py` báo lỗi `\ref` | Chạy test trước khi biên dịch, sửa theo `\label` chứ không sửa số |
