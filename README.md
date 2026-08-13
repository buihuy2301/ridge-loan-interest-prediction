# Tối ưu hóa hàm mất mát Ridge cho bài toán dự đoán lãi suất khoản vay

Bài tập môn Tối ưu hóa nâng cao, lớp Khoa học dữ liệu. Nội dung là tự cài đặt và
so sánh bốn thuật toán tối ưu hóa bậc một và bậc hai trên bài toán hồi quy tuyến
tính có hiệu chỉnh Ridge, dữ liệu Lending Club 2007-2018.

Kế hoạch chi tiết: [`KE_HOACH_TRIEN_KHAI.md`](KE_HOACH_TRIEN_KHAI.md).
Quy tắc làm việc: [`CLAUDE.md`](CLAUDE.md), kèm
[`docs/van-phong-tieng-viet.md`](docs/van-phong-tieng-viet.md) và
[`docs/quy-uoc-bao-cao.md`](docs/quy-uoc-bao-cao.md).

## Bài toán

$$
\min_{w \in \mathbb{R}^d} \quad f(w) = \frac{1}{2n} \left\| Xw - y \right\|_2^2 + \frac{\lambda}{2} \left\| w \right\|_2^2
$$

Hàm mục tiêu lồi mạnh và có nghiệm đóng, nên $f^*$ tính được chính xác tới sai số
máy và mọi biểu đồ hội tụ đều vẽ $f(w_k) - f^*$ trên thang logarit.

## Cài đặt môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lấy dữ liệu

Bộ dữ liệu không nằm trong repo vì hai file nén cộng lại 618 MB. Tải từ Kaggle:
<https://www.kaggle.com/datasets/wordsforthewise/lending-club>

Cách 1, tải thủ công: bấm Download trên trang Kaggle, giải nén, đặt hai file
`.csv.gz` vào `data/raw/`.

Cách 2, dùng API: tạo API token ở <https://www.kaggle.com/settings> mục API, tải
`kaggle.json` về, rồi

```bash
pip install kaggle
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
.venv/bin/kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip
```

Chỉ `accepted_2007_to_2018Q4.csv.gz` được dùng, vì biến mục tiêu `int_rate` không
có trong file các hồ sơ bị từ chối.

## Quy trình chạy

Bước 1, cố định bài toán. Chạy notebook `01` và `02`. Hai notebook này dựng ma
trận thiết kế, chọn $\lambda$ bằng cross-validation 5 fold, tính $L$, $\mu$,
$\kappa$, $f^*$, rồi ghi kết quả vào `data/processed/`. Từ thời điểm đó, các file
trong `data/processed/` không được sửa nữa: mọi so sánh giữa các thuật toán chỉ
có nghĩa khi chúng cùng làm việc trên một hàm mục tiêu.

Bước 2, chạy thí nghiệm. Notebook `03` đến `06` quét lưới tham số ở quy mô 200
nghìn điểm, `07` chạy lại cấu hình tốt nhất ở quy mô toàn phần và dựng hình so
sánh tổng hợp, `08` so sánh với scikit-learn và khảo sát $\lambda$. Kết quả thô
lưu vào `results/raw/` dạng JSON, mỗi nhóm thí nghiệm một file, nên lần chạy sau
bỏ qua được nhóm đã xong. Hình lưu vào `results/figures/` theo tên
`<tên nhóm>_<trục>` dưới cả hai định dạng PDF và PNG.

Bước 3, kiểm thử.

```bash
.venv/bin/python -m pytest tests/
```

## Cách tổ chức `src/`

Đề bài là ma trận bốn thuật toán nhân hai cách chọn bước, nên mã nguồn tách đúng
hai trục đó rồi ghép lại bằng một vòng lặp duy nhất, thay vì viết mỗi thuật toán
thành một hàm tự chứa vòng lặp riêng.

| File | Vai trò |
|---|---|
| `objective.py` | `RidgeObjective`: $f$, gradient, Hessian, nghiệm đóng, $L$, $\mu$ |
| `direction.py` | Trục 1: `SteepestDescent`, `MiniBatch`, `Nesterov`, `NewtonStep` |
| `stepsize.py` | Trục 2: `Fixed`, `Armijo`, `Decay` |
| `iterate.py` | Vòng lặp duy nhất: đồng hồ, ghi log, điều kiện dừng |
| `experiment.py` | Spec khai báo, tích Descartes, chạy, lưu JSON |
| `reference.py` | Bọc scikit-learn |
| `figures.py` | Hàm vẽ chuẩn, bảng màu dùng chung |

Phần đo thời gian nằm ở đúng một chỗ là `iterate.py`, nên quy tắc "dừng đồng hồ
trước khi ghi log" chỉ phải kiểm tra một lần. Mọi thứ đặc thù cho một thuật toán
thuộc về lớp `Direction` tương ứng: nếu phải viết `isinstance` trong `iterate.py`
thì phần đó đã đặt sai chỗ.

## Cấu hình máy dùng để đo thời gian

Mọi số liệu thời gian trong báo cáo đo trên cùng một máy. Số đo trên máy khác
không so sánh trực tiếp được với bảng trong báo cáo.

| Hạng mục | Giá trị |
|---|---|
| Máy | Apple M1 Pro, 8 nhân, 16 GB RAM |
| Hệ điều hành | macOS (Darwin 25.5.0) |
| Python | 3.14.4, arm64 |
| numpy | 2.5.2, BLAS Apple Accelerate |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
