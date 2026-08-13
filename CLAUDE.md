# Quy tắc làm việc trong thư mục này

Dự án: bài tập môn Tối ưu hóa nâng cao, chủ đề các thuật toán tối ưu hóa bậc một và bậc hai cho bài toán hồi quy tuyến tính có hiệu chỉnh Ridge. Kế hoạch chi tiết nằm ở `KE_HOACH_TRIEN_KHAI.md`.

## 0. Tài liệu tham chiếu

Đọc file tương ứng khi bắt đầu công việc thuộc phạm vi của nó, không cần đọc trước:

| File | Đọc khi |
| --- | --- |
| `docs/van-phong-tieng-viet.md` | Trước khi viết hoặc sửa bất kỳ đoạn văn tiếng Việt nào: báo cáo, slide, README, ô markdown trong notebook. Bắt buộc, vì mục 2 dưới đây chỉ là bản tóm tắt. |
| `docs/quy-uoc-bao-cao.md` | Trước khi sửa `report/*.tex` hoặc `report/preamble.tex`. |
| `report/README.md` | Cách biên dịch, danh sách gói LaTeX, mẫu trình bày. |
| `KE_HOACH_TRIEN_KHAI.md` | Nội dung thí nghiệm, ký hiệu toán học, outline báo cáo và slide. |

## 1. Ngôn ngữ

Có hai lớp nội dung, dùng hai ngôn ngữ khác nhau, không trộn lẫn.

**Tiếng Việt** dùng cho toàn bộ nội dung dành cho người đọc: file markdown, báo cáo, slide, phần văn bản trong notebook (ô markdown), tiêu đề và chú thích trong bài trình bày.

**Tiếng Anh** dùng cho toàn bộ mã nguồn, không có ngoại lệ: tên biến, tên hàm, tên lớp, tên module, tên file mã nguồn; comment; docstring; chuỗi log và thông báo lỗi; tên khóa trong dict và JSON; nhãn trục, tiêu đề và legend của biểu đồ.

Ngoại lệ duy nhất: tên file markdown, tên thư mục báo cáo, và nội dung file dữ liệu gốc tải từ Kaggle giữ nguyên như hiện có.

## 2. Văn phong tiếng Việt

Trước khi viết hoặc sửa bất kỳ đoạn văn tiếng Việt nào, đọc `docs/van-phong-tieng-viet.md`, trong đó có quy trình ba lượt, công thức viết đoạn và bộ mẫu. Mục này chỉ nêu hình dạng cần đạt, không đủ để viết nếu chưa đọc file kia. Bộ mẫu nằm ở `style/mau/`, ngoài git vì lý do bản quyền, và phải nạp kèm ở lượt viết nháp: mô tả trong tài liệu chặn được lỗi nhưng không dựng được giọng.

Văn đích là văn giáo trình toán ứng dụng: người viết đã làm thí nghiệm và đang thuật lại cho người đọc có nền tảng. Một đoạn phân tích kết quả gồm ba phần theo thứ tự, dữ kiện kèm số và nguồn `\ref`, cơ chế giải thích con số đó, rồi hệ quả hoặc điều kiện mà kết luận đảo chiều. Câu đầu đoạn là dữ kiện, không phải câu dẫn.

Việc viết chia thành ba lượt gọi riêng, mô tả ở mục 1 của file văn phong: lượt A dựng bảng lập luận và không viết câu văn nào, lượt B viết nháp theo mục 2 tới mục 5, lượt C rà theo mục 6 và mỗi lần chỉ rà một nhóm. Không đưa danh sách rà soát vào lượt viết nháp.

Hai mốc về nhịp, kiểm tra được bằng mắt sau khi viết xong:

- Mỗi đoạn có ít nhất một câu mang lập luận, tức câu có mệnh đề phụ chỉ nguyên nhân, điều kiện hoặc tương phản. Đoạn toàn câu đơn rời rạc đọc như bảng liệt kê, và đó là lỗi thường gặp nhất hiện nay. Không có mốc âm tiết cho câu này; chuẩn duy nhất về độ dài là bộ mẫu ở mục 2 của file văn phong.
- Hai câu liên tiếp không mở đầu bằng cùng một khuôn.

Quy tắc tách câu dài nhắm vào câu chứa từ ba ý ngang hàng trở lên nối bằng dấu phẩy, không nhắm vào câu ghép có mệnh đề phụ. Một câu dài gồm một ý chính và hai mệnh đề bổ nghĩa thì giữ nguyên.

Mọi kết luận phải kèm số liệu hoặc biểu đồ tương ứng; chưa có bằng chứng thì ghi rõ đó là dự đoán. Danh sách lỗi từ vựng và cú pháp cần rà sau khi viết nằm ở mục 6 của file văn phong.

## 3. Biểu đồ

- Vẽ bằng `matplotlib`. Được phép dùng `seaborn` cho phần tạo kiểu, hoặc thư viện Python khác nếu có lý do rõ ràng, nhưng mặc định là `matplotlib`.
- Toàn bộ chữ trên biểu đồ viết bằng tiếng Anh. Phần giải thích và kết luận về biểu đồ thì viết tiếng Việt trong văn bản kèm theo.
- Trục tung của biểu đồ hội tụ luôn là độ lớn hàm mục tiêu, dùng `semilogy` với đại lượng $f(w_k) - f^*$.
- Mỗi so sánh cần hai hình riêng: một theo số vòng lặp, một theo thời gian chạy tính bằng giây.
- Legend phải ghi rõ tham số, ví dụ `GD (t = 1/L)`, không ghi chung chung là `GD`.
- Mỗi thuật toán dùng một màu cố định xuyên suốt toàn bộ báo cáo. Định nghĩa bảng màu một lần trong `src/plotting.py` và gọi lại từ đó.
- Lưu mỗi hình ra `results/figures/` ở hai định dạng: PDF (dạng vector, để nhúng vào LaTeX) và PNG với `dpi=150` trở lên (để xem nhanh). Luôn dùng `bbox_inches='tight'`.
- Bản riêng cho slide, nếu cần bớt đường hoặc phóng to chữ so với bản trong báo cáo, lưu vào `results/figures/slides/` và giữ nguyên tên file. Lệnh `\resultgraphic` trong `preamble.tex` tìm ở thư mục này trước, không thấy mới lấy bản của báo cáo.
- Không dùng biểu đồ tương tác cho phần nộp bài, vì slide và báo cáo cần ảnh tĩnh.
- Kích thước hình đặt sao cho chữ đọc được khi chèn vào slide Beamer, thường là `figsize=(6, 4)` với cỡ chữ từ 10 trở lên. Không thu nhỏ hình trong LaTeX quá nhiều để bù cho hình vẽ quá to.

## 4. Công thức toán học

- Trong file markdown, viết công thức bằng LaTeX: `$...$` cho công thức trong dòng, `$$...$$` cho công thức tách khối.
- Không viết công thức toán trong khối code. Khối code chỉ dành cho mã nguồn thật sự chạy được, lệnh shell, hoặc cây thư mục.
- Giữ ký hiệu thống nhất với `KE_HOACH_TRIEN_KHAI.md`: $n$ số điểm dữ liệu, $d$ số thuộc tính, $X$ ma trận thiết kế, $y$ vector mục tiêu, $w$ tham số, $\lambda$ hệ số hiệu chỉnh, $t$ và $\eta$ độ dài bước, $L$ hằng số Lipschitz, $\mu$ hệ số lồi mạnh, $\kappa$ số điều kiện, $f^*$ giá trị tối ưu.

## 5. Báo cáo và slide

Chi tiết ở `docs/quy-uoc-bao-cao.md`. Các điểm bắt buộc:

- Báo cáo viết bằng LaTeX, slide làm bằng LaTeX Beamer. Không dùng Word, Google Docs, PowerPoint hay markdown cho hai sản phẩm này.
- Toàn bộ nguồn nằm trong `report/`, khai báo dùng chung đặt ở `preamble.tex`. Hình kết quả thí nghiệm giữ nguyên ở `results/figures/`, không sao chép sang `report/figures/`.
- Biên dịch bằng `latexmk -xelatex`, vì cần font Unicode đầy đủ dấu tiếng Việt.
- Mọi hình và bảng phải có `\caption`, có `\label` và được `\ref` ít nhất một lần. Mọi phát biểu lấy từ tài liệu phải có `\cite`; mục nào trong `refs.bib` không được trích dẫn thì xóa.
- Chạy `pytest tests/test_report.py` trước mỗi lần nộp.
- Không đưa sản phẩm biên dịch vào git. Thêm gói LaTeX sinh ra đuôi file mới thì cập nhật `.gitignore` ngay trong lần commit đó.

## 6. Tổ chức mã nguồn

- Logic thuật toán đặt trong `src/`. Notebook chỉ gọi hàm, chạy thí nghiệm và vẽ hình, không định nghĩa lại thuật toán.
- Mọi thuật toán tối ưu hóa dùng chung một chữ ký hàm và trả về cùng một cấu trúc lịch sử, theo mô tả ở mục 4.1 của kế hoạch.
- Khi đo thời gian chạy, dừng đồng hồ trước khi tính và ghi log, chạy lại sau đó. Thời gian ghi log không được tính vào thời gian thuật toán.
- Cố định seed cho mọi thành phần ngẫu nhiên và ghi seed vào kết quả.
- Kết quả chạy lưu ra `results/raw/` dạng JSON để vẽ lại được mà không cần chạy lại thí nghiệm.

## 7. Phạm vi

Trọng tâm của bài tập là tối ưu hóa. Không mở rộng sang feature engineering phức tạp, thử nhiều họ mô hình, hay tinh chỉnh siêu tham số ngoài phạm vi đã nêu trong kế hoạch, trừ khi được yêu cầu rõ ràng.
