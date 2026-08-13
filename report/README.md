# Biên dịch báo cáo và slide

## Lệnh

```bash
cd report
latexmk -xelatex report.tex
latexmk -xelatex slides.tex
latexmk -C                     # dọn file trung gian
```

Bắt buộc dùng XeLaTeX, không dùng pdfLaTeX, vì `fontspec` và `polyglossia` cần
engine Unicode. `latexmk` tự chạy đủ số lần và tự gọi `biber` cho phần tài liệu
tham khảo.

## Gói cần có

Tất cả đều nằm trong TeX Live bản đầy đủ, **không phải cài thêm gì**. Bản đã kiểm
chứng là TeX Live 2025 trên macOS.

| Gói | Dùng cho |
| --- | --- |
| `fontspec`, `polyglossia` | font Unicode và tiếng Việt |
| `unicode-math` | font toán khớp với font chữ |
| `amsmath`, `amssymb`, `amsthm`, `mathtools` | công thức, định lý |
| `graphicx`, `booktabs`, `siunitx`, `xcolor` | hình, bảng, số liệu, màu |
| `titlesec`, `tocloft` | tiêu đề chương và mục lục |
| `algorithm`, `algpseudocode` | mã giả |
| `biblatex` + `biber` | tài liệu tham khảo |
| `hyperref` | liên kết trong PDF |
| `beamer` | slide |

Theme slide **tự dựng trong `slides.tex`**, không dùng theme ngoài. Lý do: bản
TeX Live trên máy không có `metropolis`, và dùng nó sẽ buộc mọi thành viên phải
cài thêm trước khi biên dịch được.

## Font

| Vai trò | Font | Gọi bằng |
| --- | --- | --- |
| Chữ thường | TeX Gyre Pagella (họ Palatino) | `texgyrepagella` |
| Tiêu đề, slide | TeX Gyre Heros (họ Helvetica) | `texgyreheros` |
| Mã nguồn | TeX Gyre Cursor | `texgyrecursor` |
| Công thức | TeX Gyre Pagella Math | `texgyrepagella-math.otf` |

Phải gọi theo **tên file** như trên. Gọi theo tên hiển thị, ví dụ
`\setmainfont{TeX Gyre Pagella}`, sẽ báo không tìm thấy font.

Hai phương án dự phòng nếu máy khác thiếu font: `\setmainfont{Times New Roman}`
với font hệ thống macOS, hoặc bỏ hẳn dòng `\setmainfont` để `fontspec` dùng Latin
Modern.

## Hình

Hình kết quả nằm nguyên ở `results/figures/`, **không sao chép** sang `report/`.
`\graphicspath` trong `preamble.tex` trỏ tới đó. Thư mục `report/figures/` chỉ
dành cho hình không sinh ra từ notebook, ví dụ sơ đồ vẽ tay.

Ba lệnh chèn hình:

| Lệnh | Dùng khi |
| --- | --- |
| `\resultfig{tên}{chú thích}{nhãn}` | một hình đơn |
| `\resultpair{tên}{chú thích}{nhãn}` | cặp bắt buộc: `tên_iters` và `tên_time`, xếp dọc |
| `\resultgraphic[tỷ lệ]{tên}` | trong slide; tìm ở `results/figures/slides/` trước |

Cả ba đều chịu được hình chưa sinh ra: chỗ đó hiện một khung có tên file thay vì
làm hỏng lần biên dịch, nên viết báo cáo được song song với lúc thí nghiệm đang
chạy.

`\resultpair` xếp hai bảng con theo chiều dọc chứ không đặt cạnh nhau. Hình gốc
rộng 6 inch với cỡ chữ 10; đặt cạnh nhau ở nửa bề rộng thì chữ trong hình còn
khoảng 5 point và legend không đọc nổi.

## Kiểm tra trước khi nộp

```bash
.venv/bin/python -m pytest tests/test_report.py
```

Bài kiểm tra đối chiếu `\label` với `\ref`, bắt nhãn trùng, bắt một file hình bị
chèn hai lần dưới hai nhãn, kiểm tra file hình có thật, và kiểm tra mọi mục trong
`refs.bib` đều được trích dẫn.

## Nhận dạng thị giác

Bản này cố ý khác mẫu dùng cho môn Toán rời rạc:

| | Mẫu cũ | Bản này |
| --- | --- | --- |
| Chữ thường | TeX Gyre Termes (Times) | TeX Gyre Pagella (Palatino) |
| Tiêu đề chương | một dòng, số La Mã, `\Large` | khối hai dòng, "Chương 7" chữ sans nhỏ trên gạch ngang |
| Số chương | La Mã | Ả Rập |
| Frame title slide | thanh đặc màu xanh đậm | chữ sans, một gạch mảnh màu chàm |
| Màu chủ đạo | `darkblue` | chàm đậm `#0B5563` |
