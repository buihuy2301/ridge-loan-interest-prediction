# Văn phong tiếng Việt

Bản này thay bản cũ. Khác biệt chính: bộ mẫu chuyển từ văn Claude sinh sang văn
người viết, quy trình tách thành ba lượt gọi riêng, danh sách cấm rút ngắn còn
những mục mà mẫu không tự truyền đạt được.

## 0. Trạng thái

Mục 2 đã điền, gồm 20 đoạn của 16 nhóm tác giả, lưu ở `style/mau/`. File vì thế
đã có neo: mục 6 chặn lỗi, còn giọng thì lấy từ bộ mẫu chứ không lấy từ mô tả.

Bộ đối chứng ở mục 8 cũng đã dựng, gồm năm đoạn ở `style/doi-chung/`. Từ đây mọi
chỉnh sửa file này đều đo được, và lần sửa nào không qua được phép đo ở mục 8 thì
không nên giữ.

Hai chỗ nên bù trước: bộ mẫu chưa có đoạn nào lấy từ luận văn hay luận án, và bộ
đối chứng chỉ lấy từ ba mục nên chưa phủ các chương khảo sát tham số, vốn là nơi
đặt nhiều đoạn bình luận bảng nhất.

## 1. Ba lượt, ba lần gọi riêng

Bản cũ ghi "dùng theo hai lượt" nhưng đặt cả hướng dẫn viết lẫn danh sách cấm
trong một tài liệu, nên trên thực tế mọi thứ vào cùng một context và bản nháp
đầu tiên đã viết trong trạng thái phòng thủ. Bản này tách thật.

**Lượt A, dựng lập luận.** Không viết câu văn nào. Output là một bảng, mỗi đoạn
dự kiến một dòng, bốn cột:

| Kết luận | Số liệu chống lưng | Cơ chế | Điều kiện đảo chiều |
| --- | --- | --- | --- |

Người viết duyệt và sửa bảng này trước khi sang lượt B. Dòng nào chưa điền được
cột cơ chế thì chưa đủ nguyên liệu, và việc cần làm là đọc lại kết quả hoặc chạy
thêm thí nghiệm chứ không phải viết một đoạn mô tả bảng để lấp chỗ.

**Lượt B, viết nháp.** Đưa mục 2 tới mục 5, bảng đã chốt, và các file mẫu ở
`style/mau/` ứng với chức năng của đoạn sắp viết. Không đưa mục 6. Yêu cầu viết
đặc thông tin, chưa cần đẹp.

**Lượt C, rà.** Đưa mục 6, mỗi lần một nhóm, không rà cả ba nhóm trong một lượt.
Sửa để tránh lỗi ở nhóm này dễ tạo ra lỗi ở nhóm khác nếu làm đồng thời.

Câu mang kết luận của mỗi đoạn nên do người viết tự đặt, ở lượt A. Đó là chỗ khó
bắt chước nhất, cũng là chỗ hội đồng hỏi trực tiếp.

## 2. Bộ mẫu

Bộ mẫu nằm ở `style/mau/`, gồm 20 đoạn trích nguyên văn từ 16 bài báo tiếng Việt
có bình duyệt, mỗi đoạn một file kèm tác giả, tên tài liệu, năm, nơi đăng và
đường dẫn ở đầu file. Không đoạn nào do mô hình sinh ra và không đoạn nào bị sửa
chữ; thao tác duy nhất là gỡ phần ngắt dòng của bản PDF hai cột, nên nhịp câu và
trật tự mệnh đề giữ đúng như tác giả viết.

| Chức năng | Số đoạn | Tên file |
| --- | --- | --- |
| Bình luận một bảng số | 5 | `bang-so-*.md` |
| Giải thích một chênh lệch bất ngờ giữa hai phương pháp | 5 | `chenh-lech-*.md` |
| Nêu giới hạn hoặc điều kiện đảo chiều của kết luận | 4 | `gioi-han-*.md` |
| Giải thích một quyết định về thiết kế thí nghiệm | 3 | `thiet-ke-*.md` |
| Chuyển tiếp giữa hai mục | 3 | `chuyen-tiep-*.md` |

Hai mươi đoạn ứng với 16 nhóm tác giả và không chức năng nào lấy quá một đoạn từ
cùng một nhóm, nên bộ mẫu không truyền tật riêng của một người viết. Ba tạp chí
góp mặt là Khoa học Đại học Mở TP.HCM, Khoa học và Công nghệ Đại học Đà Nẵng, và
TNU Journal of Science and Technology, với các bài từ năm 2011 tới năm 2026.

Bộ mẫu hiện chưa có đoạn nào lấy từ luận văn hay luận án, tức thể loại sát bài
này nhất. Bổ sung được thì nên bổ sung, theo thứ tự nguồn ngay dưới đây.

### Nguồn

Luận văn và luận án, ưu tiên hàng đầu vì thể loại trùng với bài của bạn:

- `repository.vnu.edu.vn`, thư viện số tài nguyên nội sinh của ĐHQG Hà Nội, có
  bộ sưu tập luận văn và luận án toàn văn.
- `ir.vnulib.edu.vn`, kho của ĐHQG TP.HCM, bộ sưu tập luận văn thạc sĩ và luận
  án tiến sĩ của Trường Đại học Khoa học Tự nhiên nằm ở `handle/VNUHCM/7914`.
- Thư viện trường bạn đang học. Luận văn cùng khoa là mẫu sát nhất, vì nó đã qua
  đúng hội đồng sẽ chấm bài bạn.

Tạp chí tiếng Việt có bình duyệt, truy cập mở:

- `jst.tnu.edu.vn`, chuyên san Khoa học Tự nhiên - Kỹ thuật - Công nghệ và chuyên
  san Công nghệ Thông tin và Truyền thông.
- `jst-ud.vn`, Tạp chí Khoa học và Công nghệ Đại học Đà Nẵng.
- `journalofscience.ou.edu.vn`, chuyên san Kỹ thuật và Công nghệ.
- `jmst.info`, mục Công nghệ thông tin và Cơ sở toán học cho tin học.

Giáo trình giải tích số, tối ưu hóa, xác suất thống kê của tác giả trong nước.

Nguồn nên tránh: sách dịch, vì bản dịch mang sẵn cấu trúc tiếng Anh, đúng thứ mà
mục 3 đang tìm cách chặn. Các trang bán và chia sẻ luận văn thương mại cũng nên
tránh: chất lượng không qua bình duyệt, và nhiều bản là văn thuê viết.

### Quy trình

Quy trình dưới đây đã chạy một lần cho 20 đoạn hiện có; dùng lại nguyên vậy khi
thêm đoạn mới.

1. Tải bản PDF từ các nguồn trên, lưu vào `style/mau/nguon/`.
2. Trích đoạn thủ công, mỗi đoạn một file, đặt tên theo chức năng và số thứ tự,
   ví dụ `bang-so-01.md`. Ghi tác giả, tên tài liệu, năm, số trang ở đầu file.
3. Lọc theo mục "Lọc trước khi đưa vào" bên dưới.
4. Thư mục `style/mau/` thêm vào `.gitignore`. Bộ mẫu là tài liệu tham chiếu
   riêng, không phải một phần của báo cáo, và phần lớn nguồn cấp phép theo
   CC BY-NC-ND nên không phát hành lại được.
5. Ở lượt B, nạp các file mẫu kèm prompt. Không dán chúng vào chính file này.

Mỗi chức năng lấy từ ít nhất ba tác giả khác nhau. Một tác giả thì bộ mẫu sẽ
truyền cả tật riêng của người đó.

Đoạn trích phải nguyên văn. Bản PDF hai cột khi chuyển sang văn bản thường dính
số trang, tên chạy đầu trang và chú thích hình vào giữa đoạn, nên bước cần làm là
gỡ những mảnh đó rồi nối lại các dòng bị ngắt, không phải viết lại câu cho gọn.
Sửa chữ dù chỉ một câu thì đoạn đó thôi là mẫu, vì thứ cần học chính là lựa chọn
của người viết.

### Lọc trước khi đưa vào

Văn khoa học tiếng Việt có tật phổ biến là câu dẫn dài trước khi vào số liệu, bị
động chồng bị động, và khuôn "chúng tôi nhận thấy rằng". Loại các đoạn mắc những
lỗi này thay vì lấy nguyên cả cụm.

Để mẫu đứng trần. Nhiều nhất một dòng nói đoạn đó đang làm việc gì. Chú giải kỹ
về độ dài câu hay vị trí mệnh đề sẽ biến mẫu trở lại thành quy tắc, và mô hình sẽ
bám con số thay vì bám giọng.

## 3. Trật tự thông tin

Phần lớn cảm giác "dịch máy" trong bản nháp không nằm ở từ vựng mà nằm ở chỗ
thông tin được xếp theo trật tự chủ ngữ và vị ngữ của tiếng Anh rồi mới thay bằng
từ tiếng Việt. Tiếng Việt tổ chức câu theo đề và thuyết: phần đề nêu cái đang bàn
tới, không bắt buộc phải là chủ ngữ ngữ pháp, và phần thuyết nói về nó.

Các cặp dưới đây minh họa một lỗi cụ thể. Chúng không phải mẫu văn để bắt chước;
mẫu nằm ở mục 2.

**Danh hóa để lấp chỗ chủ ngữ.**

> Nặng: Việc chuẩn hóa các cột dữ liệu đã dẫn đến sự cải thiện đáng kể của số
> điều kiện.
>
> Nhẹ: Chuẩn hóa cột xong, số điều kiện giảm từ 17,14 xuống 2,68.

Tiếng Anh cần một danh ngữ làm chủ ngữ nên phải danh hóa động từ. Tiếng Việt đặt
thẳng hành động lên đầu làm đề. Dấu hiệu nhận ra: chuỗi "việc", "sự", "quá
trình", "tính chất của" đứng đầu câu.

**Chủ ngữ giả trỏ về cả mệnh đề trước.**

> Nặng: Ma trận Gram gần suy biến. Điều này cho thấy rằng hệ số hiệu chỉnh quyết
> định số điều kiện.
>
> Nhẹ: Ma trận Gram gần suy biến, nên số điều kiện do hệ số hiệu chỉnh quyết
> định chứ không do dữ liệu.

**Mệnh đề quan hệ lồng sâu.**

> Nặng: Phương pháp mà nhóm dùng để ước lượng hằng số Lipschitz, vốn dựa trên trị
> riêng lớn nhất của ma trận Gram, cho kết quả ổn định.
>
> Nhẹ: Nhóm ước lượng hằng số Lipschitz qua trị riêng lớn nhất của ma trận Gram,
> và kết quả ổn định qua các lần chạy.

Tiếng Việt nối mệnh đề theo chuỗi tuyến tính tốt hơn là lồng vào giữa câu. Khi
một mệnh đề chen giữa chủ ngữ và vị ngữ, tách nó ra thành vế nối bằng liên từ.

## 4. Công thức đoạn phân tích kết quả

Một đoạn bình luận bảng hoặc hình gồm ba phần, theo thứ tự lập luận chứ không
theo thứ tự đọc bảng:

1. **Dữ kiện.** Số đọc từ bảng, kèm nguồn bằng `\ref` và kèm mốc so sánh. Là câu
   đầu tiên, không có câu dẫn đứng trước.
2. **Cơ chế.** Vì sao số ra như vậy, nối vào dữ kiện bằng "vì", "do", "khi". Nếu
   cần công thức thì dẫn bằng `\eqref` ngay trong câu.
3. **Hệ quả hoặc giới hạn.** Kết luận, hoặc điều kiện mà kết luận không còn đúng.

Ba phần không cần ba câu, và phần được phép lược là cơ chế khi nó đã trình bày ở
mục trước.

Mốc để tách đoạn là số cơ chế, không phải số kết luận. Hai kết luận cùng một cơ
chế viết chung được; hai kết luận cần hai cơ chế thì nằm ở hai đoạn, ngay cả khi
cùng đọc từ một bảng.

## 5. Nhịp câu

Bệnh của bản nháp gần đây là câu đều: chuỗi câu trần thuật ngắn, mỗi câu một
mệnh đề chính, không câu nào nối vào câu nào. Đoạn như vậy đọc như bảng liệt kê.

Mỗi đoạn cần ít nhất một câu mang lập luận, tức câu có mệnh đề phụ chỉ nguyên
nhân, điều kiện hoặc tương phản. Các câu còn lại đặt dữ kiện quanh nó. Bản cũ quy
định mốc âm tiết cho câu này; bỏ mốc đó, vì nó dẫn tới việc kéo dài câu cho đủ
chỉ tiêu. Mẫu ở mục 2 là chuẩn duy nhất về độ dài.

Hai câu liên tiếp không mở đầu bằng cùng một khuôn. Các khuôn hay lặp: chủ ngữ là
tên bảng hoặc hình, "Với...", "Khi...", "Số liệu...".

Liên từ theo quan hệ:

| Quan hệ | Từ nối |
| --- | --- |
| Nguyên nhân | vì, do, bởi |
| Hệ quả | nên, do đó, nhờ vậy, vì thế |
| Điều kiện | nếu, khi, miễn là, trừ khi |
| Tương phản | nhưng, trong khi, ngược lại, đổi lại |
| Diễn giải | tức là, nói cách khác |

Tách câu chứa từ ba ý ngang hàng trở lên nối bằng dấu phẩy. Mốc là số ý, không
phải số tiếng: một câu dài gồm một ý chính và hai mệnh đề bổ nghĩa thì giữ.

## 6. Rà soát

Dùng ở lượt C, mỗi lần một nhóm.

**Nhóm 1, giàn giáo.**

- Bỏ câu chỉ bình luận rằng nội dung sắp viết là quan trọng. Phép thử: xóa câu đó
  rồi xem người đọc có mất thông tin nào không.
- Giữ câu chỉ đường mang thông tin về vị trí: "Mục 8 khai thác tính chất này".
- Bỏ câu kết đoạn dạng cách ngôn. Nếu cần câu tổng kết thì gắn nó với số cụ thể.
- Bỏ khung "Lý do là", "Nguyên nhân là". Nối thẳng bằng "vì", "do", "nên".

**Nhóm 2, câu và chủ ngữ.** Xem mục 3 trước khi rà nhóm này.

- Không mở đầu câu bằng "Đây là", "Điều này", "Điều đó" trỏ về cả mệnh đề trước.
- Chủ ngữ của việc cài đặt, chạy thí nghiệm, chọn tham số là "nhóm" hoặc lược bỏ.
  Khi câu nói về nội dung của chính văn bản thì lấy văn bản làm chủ ngữ: "Báo cáo
  gồm tám mục".
- Hạn chế bị động "được" khi đã có chủ thể rõ. Giữ khi nó mang nghĩa khả năng:
  "tính được", "đo được".
- Không dùng bị động có tác nhân "bị chi phối bởi X". Đảo thành "X chi phối".
- Tiêu đề mục dùng lối danh hóa hoặc động từ chủ động, không viết thành mệnh đề.

**Nhóm 3, từ vựng và khuôn lặp.**

- Không dùng em dash. Thay bằng phẩy, hai chấm, ngoặc đơn, hoặc tách câu.
- Không dùng từ cường điệu, câu cảm thán, emoji trong phần nội dung.
- Không trộn văn nói: "thắng rõ", "đụng tới", "chỉ đơn giản là", "tỏ ra vượt
  trội".
- Không chen tiếng Anh khi đã có từ tương đương. Thuật ngữ giữ nguyên tiếng Anh
  phải dùng nhất quán cả bài: `line search`, `backtracking`, `epoch`.
- Dấu hai chấm chỉ dùng khi dẫn vào liệt kê, định nghĩa, hoặc phần định lượng cho
  mệnh đề trước. Đếm lại nếu quá một phần mười số câu kết thúc bằng dấu hai chấm.
- Công thức "không phải X mà là Y" dùng nhiều nhất một lần mỗi mục.
- Mỗi khái niệm một tên duy nhất trong toàn bài, chú thích tiếng Anh trong ngoặc
  ở lần xuất hiện đầu.
- Dấu thập phân trong văn xuôi là dấu phẩy; số trong công thức toán giữ dấu chấm.
- Mọi kết luận rút ra từ thí nghiệm phải kèm số liệu. Chưa có bằng chứng thì ghi
  rõ đó là dự đoán.

## 7. Ranh giới với mã nguồn

Quy tắc ngôn ngữ ở mục 1 của `CLAUDE.md` cấm tiếng Việt trong mã nguồn, kể cả
comment và docstring.

```python
def backtracking_line_search(problem, w, grad, alpha=0.3, beta=0.8, t0=1.0):
    """Armijo backtracking. Returns (step_size, n_function_evals)."""
    t = t0
    n_evals = 0
    # Shrink the step until the sufficient decrease condition holds.
    while problem.f(w - t * grad) > problem.f(w) - alpha * t * grad @ grad:
        t *= beta
        n_evals += 1
    return t, n_evals
```

## 8. Bộ đối chứng

Không có cách đo thì mọi chỉnh sửa file này đều là đoán, và file sẽ dài ra mà
không ai biết mục nào đang có tác dụng.

Bộ đối chứng nằm ở `style/doi-chung/`, gồm năm đoạn lấy từ bản nháp hiện tại:

| File | Lấy từ | Chức năng | Vai trò |
| --- | --- | --- | --- |
| `doi-chung-01.md` | mục 1.1, đoạn 1 | mô tả thành phần bộ dữ liệu | lỗi nặng nhất, đoạn liệt kê thuần |
| `doi-chung-02.md` | mục 1.1, đoạn 3 | quyết định thiết kế thí nghiệm | mốc chống thụt lùi |
| `doi-chung-03.md` | mục 2.3, đoạn 1 | bình luận một hình | thiếu câu hệ quả cuối đoạn |
| `doi-chung-04.md` | mục 2.3, đoạn 3 | nêu đánh đổi của một quyết định | thiếu điều kiện đảo chiều |
| `doi-chung-05.md` | chương 6, câu dẫn và đoạn đầu | mở chương rồi bình luận bảng | phép thử sạch, lỗi nằm trọn ở câu dẫn |

Hai vai trò trong cột cuối phục vụ hai phép đo khác nhau. Đoạn có lỗi đo xem thay
đổi có sửa được lỗi không, còn đoạn làm mốc chống thụt lùi đo xem thay đổi có phá
thứ đang chạy tốt không. Thiếu vế thứ hai thì mọi chỉnh sửa đều có vẻ hiệu quả,
vì chỉ nhìn vào chỗ mình vừa nhắm tới.

Mỗi file ghi ba phần: đoạn nguyên trạng, lỗi cụ thể ở từng câu kèm mục quy tắc
tương ứng, và bảng lập luận bốn cột của lượt A. Bảng đó là phần quan trọng nhất,
vì nó cố định đầu vào: sinh lại từ cùng một bảng thì chênh lệch giữa hai bản chỉ
còn do file này thay đổi, không do người viết nghĩ ra kết luận khác.

Cách chạy: sau khi sửa file này, chạy lại lượt B trên đúng năm bảng lập luận đó,
mỗi đoạn một lần, rồi đối chiếu với phần lỗi đã ghi. Không sửa bộ đối chứng cho
khớp với kết quả mới, vì như vậy là dịch cái thước.

Ba dấu hiệu cho thấy một thay đổi có tác dụng: lỗi đã ghi biến mất, không xuất
hiện lỗi mới ở nhóm khác, và đoạn không dài thêm mà vẫn giữ đủ số liệu. Mỗi file
ghi sẵn số câu, số tiếng và độ dài câu dài nhất để so dấu hiệu thứ ba.
