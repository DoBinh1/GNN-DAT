# GNN Panels — Mạng nơ-ron đồ thị cho dự đoán ứng suất trong stiffened panel

> Triển khai lại nghiên cứu Cai & Jelovica (2024) bằng Python + PyTorch + PyTorch Geometric.

Tài liệu này giúp bạn (1) hiểu mỗi thành phần code làm gì, (2) chạy được pipeline end-to-end, và (3) tự sửa/mở rộng để thay bằng dữ liệu FEM thật. Tài liệu được viết theo phong cách giảng dạy — không chỉ liệt kê file, mà giải thích triết lý đằng sau từng quyết định thiết kế.

---

## 1. Cấu trúc project và vai trò từng phần

Project được tổ chức theo nguyên tắc **tách bạch trách nhiệm** — mỗi module phụ trách một việc duy nhất, không lẫn vào nhau. Đây là cấu trúc thực tế của các project ML production, không phải prototype trong notebook.

```
gnn_panels/
├── config.py              ← Tập trung mọi siêu tham số
├── data/                  ← Sinh data + xây graph
│   ├── synthetic_panel.py     - Sinh thiết kế ngẫu nhiên + stress giả
│   ├── graph_builder.py       - Chuyển panel → PyG Data
│   └── dataset.py             - Đóng gói thành Dataset class
├── models/                ← Kiến trúc mạng
│   └── graphsage.py           - StiffenedPanelGNN
├── utils/                 ← Chuẩn hóa + metrics
│   ├── normalize.py
│   └── metrics.py
├── train.py               ← Vòng lặp huấn luyện
├── predict.py             ← Inference + visualize
├── run_demo.py            ← Demo end-to-end (1 lệnh)
└── requirements.txt
```

Triết lý sau cấu trúc này: nếu bạn sau này muốn thay dữ liệu giả lập bằng dữ liệu ABAQUS thật, **chỉ cần sửa file `data/synthetic_panel.py`** — toàn bộ phần model, training, evaluation không đổi. Đây là sức mạnh của thiết kế module hóa: thay đổi cục bộ không kéo theo thay đổi toàn cục.

File `config.py` giữ vai trò "bảng điều khiển". Mọi siêu tham số quan trọng (số layer, learning rate, batch size, miền lấy mẫu) đều ở đây, để khi muốn thí nghiệm bạn chỉ sửa một file. Nếu siêu tham số nằm rải rác trong code, bạn sẽ phải lục từng file để đổi — đây là một trong những lỗi phổ biến nhất khi viết code research mà không có kế hoạch.

Thư mục `data/` chứa toàn bộ logic về dữ liệu. Nó được chia làm ba file để mỗi file chỉ làm một việc: `synthetic_panel.py` sinh ra thiết kế và stress giả, `graph_builder.py` chuyển thiết kế thành đồ thị, `dataset.py` đóng gói cho PyG. Khi sau này bạn có dữ liệu thật, bạn chỉ thay `synthetic_panel.py` mà giữ nguyên hai file kia.

Thư mục `models/` chứa riêng kiến trúc mạng. Cấu trúc này cho phép bạn dễ dàng thêm các model khác (vd GAT, GIN, attention-based GNN) mà không đụng vào phần data hay training.

Thư mục `utils/` chứa các tiện ích cho phép tái sử dụng giữa training và inference: `normalize.py` xử lý chuẩn hóa feature, `metrics.py` chứa các hàm đo độ chính xác.

Hai file gốc `train.py` và `predict.py` đóng vai trò "entry point" — bạn chạy chúng để thực thi pipeline. `run_demo.py` là wrapper tiện lợi gộp cả hai để chạy nhanh.

---

## 2. Graph Representation — phần quan trọng nhất

Đây là nơi thể hiện đóng góp chính của paper, và cũng là nơi cần hiểu sâu nhất nếu bạn muốn mở rộng nghiên cứu. Tôi giải thích trước về mặt khái niệm rồi mới đi vào code.

### 2.1. Triết lý: cấu trúc rời rạc thì dùng đồ thị rời rạc

Một stiffened panel không phải là một khối liên tục đồng nhất. Nó được sản xuất từ các tấm và gân riêng biệt, sau đó hàn lại với nhau. Cấu trúc rời rạc tự nhiên này gợi ý ngay một biểu diễn đồ thị: mỗi thành phần (plate-span, web, flange) là một đỉnh, mỗi mối hàn là một cạnh. Khi biểu diễn như vậy, ta tôn trọng đúng bản chất vật lý của kết cấu, không phải ép nó vào một "khuôn ảnh" hay "vector cố định".

Trong code, panel có $n$ stiffener sẽ tạo ra $n+1$ plate-span (các tấm con kẹp giữa các gân và mép), $n$ web (phần đứng của gân), và $n$ flange (cánh ngang). Tổng số đỉnh là $3n + 1$, một con số nhỏ hơn rất nhiều so với hàng vạn phần tử mesh trong cách biểu diễn truyền thống.

### 2.2. Node feature — 8 con số mã hóa một thành phần

Mỗi đỉnh mang theo một vector 8 chiều, được thiết kế để mã hóa đầy đủ thông tin cần thiết cho việc dự đoán ứng suất:

| Index | Feature | Ý nghĩa vật lý |
|-------|---------|----------------|
| 0 | width | bề rộng của đơn vị (mm) |
| 1 | length | chiều dài (mm) |
| 2 | thickness | độ dày (mm) |
| 3 | bc_edge_1 | điều kiện biên cạnh 1 |
| 4 | bc_edge_2 | điều kiện biên cạnh 2 |
| 5 | bc_edge_3 | điều kiện biên cạnh 3 |
| 6 | bc_edge_4 | điều kiện biên cạnh 4 |
| 7 | pressure | áp suất tải (MPa) |

Một câu hỏi tự nhiên: vì sao không cần feature về **vị trí** của đơn vị trong panel (vd "tôi là plate-span thứ 3 từ trái"). Câu trả lời nằm ở đặc thù của GNN: thông tin vị trí đã được mã hóa NGẦM thông qua tô-pô của đồ thị. Một plate-span ở mép có ít hàng xóm hơn một plate ở giữa, và mạng có thể tự suy ra vị trí qua message passing. Đây là một bài học quan trọng — đừng nhồi vào feature những gì cấu trúc đồ thị đã tự cung cấp.

### 2.3. Edge — mối hàn vật lý

Cạnh trong đồ thị tương ứng đúng với mối hàn vật lý giữa hai thành phần. Plate-span thứ $i$ nằm giữa stiffener $(i-1)$ và stiffener $i$, do đó nó có cạnh nối với web $(i-1)$ (nếu tồn tại) và web $i$ (nếu tồn tại). Mỗi web có một cạnh nối với flange của riêng nó. Toàn bộ logic này nằm trong hàm `_build_edges()` của `graph_builder.py`.

Ý nghĩa vật lý của cạnh là **đường truyền ứng suất**: khi plate biến dạng, ứng suất truyền sang web thông qua đúng mối hàn này, và GNN sẽ "lan truyền thông tin" qua đúng các cạnh đó trong message passing. Sự tương ứng một-một giữa cạnh đồ thị và đường truyền ứng suất cơ học là lý do tại sao GNN học hiệu quả với bài toán này.

### 2.4. Đoạn code chính

Khi bạn gọi `build_graph_from_design(design)`, hàm thực hiện ba bước. Đầu tiên, nó duyệt qua tất cả các đơn vị cấu trúc (plate-span, web, flange) và tạo ra vector đặc trưng tương ứng cho mỗi đơn vị. Tiếp theo, nó duyệt qua các cặp đơn vị có quan hệ vật lý và thêm cạnh giữa chúng. Cuối cùng, nó đóng gói tất cả thành một đối tượng `torch_geometric.data.Data` — chuẩn của PyG.

Điểm tinh tế ở bước cuối: vì đồ thị là vô hướng, mỗi cạnh phải xuất hiện ở cả hai chiều trong `edge_index`. Code thực hiện việc này bằng `torch.cat([edges, edges.flip(0)])`, nhân đôi danh sách cạnh.

---

## 3. Model GraphSAGE — học thông qua message passing

Mô hình được định nghĩa trong `models/graphsage.py` dưới dạng class `StiffenedPanelGNN`. Kiến trúc gồm ba thành phần chính, xếp theo thứ tự xử lý dữ liệu.

### 3.1. Thành phần 1: stack các lớp SAGEConv

Mỗi lớp `SAGEConv` thực hiện một vòng "message passing" theo công thức:

$$
\mathbf{h}_v^{l} = \sigma\!\left(\mathbf{W}^{l}\!\left(\mathbf{h}_v^{l-1} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{l-1}\right)\right)
$$

Hãy hiểu công thức này theo trực giác: mỗi đỉnh "nghe ngóng" từ các đỉnh kề nó, cộng tất cả thông tin nghe được vào hiểu biết của bản thân, rồi đi qua một biến đổi tuyến tính có thể học được, cuối cùng qua hàm phi tuyến $\tanh$ để tạo ra hiểu biết mới.

Sau lớp đầu tiên, mỗi đỉnh đã biết về bản thân và các hàng xóm trực tiếp. Sau lớp thứ hai, nó biết cả hàng xóm-của-hàng-xóm. Sau $L$ lớp, vùng tiếp nhận của mỗi đỉnh mở rộng đến khoảng cách $L$ trên đồ thị. Vì panel của ta chỉ có khoảng 16-30 đỉnh với đường kính graph nhỏ, 6-8 lớp đã đủ để mỗi đỉnh "nhìn thấy" toàn panel.

### 3.2. Thành phần 2: Batch Normalization sau mỗi SAGEConv

Trong mạng GNN sâu, có một hiện tượng nguy hiểm gọi là **over-smoothing**: sau quá nhiều vòng message passing, các vector đặc trưng của các đỉnh khác nhau trở nên giống nhau, làm mất khả năng phân biệt chúng. Batch Normalization giúp chống lại hiện tượng này bằng cách chuẩn hóa các vector về mean 0 variance 1 trong mỗi mini-batch, qua đó duy trì sự đa dạng giữa các đỉnh.

Đồng thời, BatchNorm cũng giúp gradient lan truyền ổn định trong mạng sâu — tránh được vấn đề gradient vanishing/exploding mà mạng nhiều lớp thường gặp.

### 3.3. Thành phần 3: linear head ra 200 chiều

Sau khi qua tất cả các lớp GraphSAGE, mỗi đỉnh có một vector trạng thái 64 chiều "đậm đặc" thông tin. Lớp `head = nn.Linear(hidden, out_dim)` chiếu vector này ra 200 chiều — chính là vector ứng suất tại 200 điểm sampling trên lưới $10 \times 20$ của đơn vị tương ứng.

Khi visualize, ta reshape vector này thành lưới 2D để vẽ contour stress.

### 3.4. Forward pass — dữ liệu thay đổi qua từng layer

Khi bạn gọi `model(x, edge_index)` với `x` shape `(N, 8)` và `edge_index` shape `(2, E)`, dòng dữ liệu như sau. Tại layer đầu, `SAGEConv(8 → 64)` biến mỗi đỉnh từ vector 8-chiều thành vector 64-chiều, đồng thời tổng hợp thông tin từ hàng xóm. Sau đó BatchNorm và `tanh` chuẩn hóa và phi tuyến hóa. Quá trình này lặp lại 6-8 lần (tùy `n_layers`), với các SAGEConv tiếp theo có shape `64 → 64`. Cuối cùng, `Linear(64 → 200)` chiếu ra 200 giá trị stress. Đầu ra cuối cùng có shape `(N, 200)`.

---

## 4. Dataset — sinh dữ liệu giả lập có ý nghĩa vật lý

Vì không phải ai cũng có ABAQUS, project này sinh dữ liệu nhân tạo bằng cách kết hợp các quy luật cơ học đơn giản với phân phối ngẫu nhiên. Dữ liệu này KHÔNG phải là ground truth thật, nhưng nó tuân thủ các quy luật vật lý cơ bản đủ để mô hình học được cái gì đó có ý nghĩa.

### 4.1. Sinh thiết kế ngẫu nhiên

Hàm `generate_panel_design(seed)` trong `data/synthetic_panel.py` lấy mẫu các tham số thiết kế từ phân phối đều trong miền cho phép (giống Bảng 2 của paper). Mỗi tham số được bốc độc lập: độ dày tấm trong [10, 20] mm, số gân trong [2, 8], áp suất trong [0.05, 0.10] MPa, v.v. Việc dùng `seed` làm tham số đầu vào đảm bảo tính tái lập — chạy hai lần với cùng seed sẽ ra cùng thiết kế.

### 4.2. Sinh stress giả có ý nghĩa vật lý

Đây là phần tinh tế. Hàm `generate_pseudo_stress_field()` không trả ngẫu nhiên một mảng số, mà tuân thủ các quy luật:

Quy luật thứ nhất là **biên độ ứng suất** tỷ lệ thuận với áp suất và bình phương kích thước, tỷ lệ nghịch với bình phương độ dày — gần đúng theo lý thuyết uốn tấm $\sigma \sim p L^2 / t^2$. Đây là quy luật cơ học cơ bản: tấm dày hơn chịu ứng suất nhỏ hơn dưới cùng tải, tấm dài hơn chịu ứng suất lớn hơn.

Quy luật thứ hai là **pattern phân bố**: nếu điều kiện biên là fixed (ngàm), ứng suất cực đại xuất hiện ở mép; nếu là simply supported (gối tựa đơn giản), ứng suất cực đại ở giữa. Code mô phỏng hai pattern này bằng tổng tuyến tính của hai hàm có hình dạng tương ứng, rồi nội suy giữa chúng theo "BC trung bình" của các cạnh.

Quy luật thứ ba là **chênh lệch ứng suất giữa các loại unit**: web thường chịu stress trung bình, flange thường thấp hơn cả hai. Code áp dụng các hệ số nhân khác nhau (0.6 cho web, 0.4 cho flange) so với plate.

Cuối cùng, một lượng nhiễu nhỏ (~5%) được thêm vào để mô phỏng "biến thiên cục bộ" như mesh thật.

Ý nghĩa của cách làm này: model GNN có thể HỌC được — không phải nhớ thuộc lòng — các quy luật cơ học. Nó học được rằng tấm dày hơn cho stress nhỏ hơn, tấm có BC fixed có pattern khác BC simply, v.v. Khi sau này ta thay bằng dữ liệu ABAQUS thật, các quy luật này chỉ trở nên CHÍNH XÁC hơn chứ không thay đổi về bản chất, nên model đã được train trên data giả vẫn có thể fine-tune nhanh.

### 4.3. Đóng gói thành PyG Dataset

Class `StiffenedPanelDataset` trong `data/dataset.py` kế thừa từ `torch_geometric.data.Dataset`. Mỗi item trong dataset là một `Data` object đại diện cho một panel. Dataset hỗ trợ chia train/val/test theo seed — quan trọng là chia ở cấp panel (không phải cấp node) để tránh rò rỉ dữ liệu.

---

## 5. Training Pipeline — model học cái gì

File `train.py` orchestrate toàn bộ vòng lặp huấn luyện. Tôi sẽ giải thích chi tiết hơn về một vài quyết định thiết kế.

### 5.1. Vì sao MSE là hàm mất mát phù hợp

MSE phạt sai số lớn nhiều hơn sai số nhỏ — bình phương sai số 10 lớn gấp 100 lần bình phương sai số 1. Trong thiết kế kết cấu, ứng suất CỰC ĐẠI là đại lượng quyết định độ bền (vì failure xảy ra tại điểm có ứng suất cao nhất), do đó việc dự đoán đúng các giá trị stress lớn quan trọng hơn dự đoán đúng các giá trị nhỏ. MSE tự nhiên hướng mạng tới ưu tiên này.

Một lựa chọn thay thế là MAE (Mean Absolute Error), nhưng MAE phạt đều mọi sai số bất kể độ lớn — không phù hợp với mục tiêu kỹ thuật. Đây là một ví dụ điển hình cho thấy việc chọn loss function nên xuất phát từ ý nghĩa kỹ thuật, không phải thói quen.

### 5.2. Mô hình đang học cái gì

Một cách hình tượng, mô hình đang học một **ánh xạ**: từ "không gian thiết kế" (hình học + BC + tải trọng) sang "không gian trường ứng suất" (200 giá trị stress trên mỗi đơn vị). Cụ thể hơn, nó học các quy tắc như:
- Khi tấm dày, stress tại tấm thấp.
- Khi áp suất cao, stress ở mọi nơi cao.
- Khi cạnh là fixed, pattern stress dồn về mép.
- Khi tấm có nhiều gân (= nhiều láng giềng trong graph), độ cứng cao hơn nên stress thấp hơn.

GNN học được quy tắc cuối cùng nhờ aggregator `sum`: tổng các vector trạng thái của hàng xóm thay đổi theo số hàng xóm, do đó mạng phân biệt được "1 stiffener" vs "5 stiffeners" — điều mà aggregator `mean` không phân biệt được.

### 5.3. Early stopping và checkpoint

Code theo dõi `val_loss` mỗi epoch. Khi `val_loss` cải thiện, nó lưu checkpoint vào `outputs/checkpoints/best.pt`. Khi `val_loss` không cải thiện trong 20 epoch liên tiếp (cấu hình mặc định), training dừng để tránh overfitting.

Đây là một mẫu thực hành chuẩn: thay vì train cố định một số epoch và hy vọng, ta để mô hình tự dừng khi nó đã hội tụ. Checkpoint cuối cùng dùng cho test luôn là checkpoint có val loss thấp nhất, không phải checkpoint cuối cùng.

---

## 6. Cài đặt và chạy thử end-to-end

### 6.1. Cài đặt môi trường

Tạo môi trường Python 3.9+, sau đó cài đặt các thư viện. Lưu ý: torch_geometric có thể yêu cầu cài đặt thêm các phụ thuộc native (torch_scatter, torch_sparse) tùy phiên bản PyTorch và CUDA. Với PyTorch 2.1+ và `torch_geometric` 2.4+, các phụ thuộc thường được cài tự động.

```bash
pip install torch
pip install torch_geometric
pip install numpy matplotlib
```

Hoặc dùng file requirements:

```bash
cd "d:/[Lab] HUST/dat/gnn_panels"
pip install -r requirements.txt
```

### 6.2. Chạy demo end-to-end

Cách đơn giản nhất — demo chạy với cấu hình nhỏ, vài phút:

```bash
cd "d:/[Lab] HUST/dat"
python -m gnn_panels.run_demo
```

Hoặc chạy từng bước riêng:

```bash
python -m gnn_panels.train       # huấn luyện
python -m gnn_panels.predict     # inference với panel test
```

### 6.3. Kết quả mong đợi

Trong quá trình train, bạn sẽ thấy log dạng:

```
Epoch   1 | train MSE=0.0234 | val MSE=0.0198 | val acc=72.34%  ★
Epoch   5 | train MSE=0.0102 | val MSE=0.0089 | val acc=85.21%  ★
...
Epoch  50 | train MSE=0.0021 | val MSE=0.0027 | val acc=93.45%
[RESULT] Test max-stress accuracy = 92.10%
```

Sau khi predict, bạn sẽ thấy bảng so sánh:

```
Node idx  Type        Pred max (MPa)   True max (MPa)      Acc
------------------------------------------------------------------
0         plate              215.34           218.40    98.6%
1         plate              225.12           220.80    98.0%
...
```

Vì là dữ liệu giả lập, "True" cũng là số giả, nên kết quả thể hiện việc mô hình đã học được các quy luật vật lý mà chúng ta cài đặt vào hàm sinh stress giả. Khi bạn thay bằng dữ liệu ABAQUS thật, "True" sẽ là số thật, nhưng pipeline không thay đổi.

---

## 7. Pipeline tổng thể — từ input đến output

Để hình dung toàn bộ hệ thống như một dòng chảy, hãy theo dõi đường đi của dữ liệu khi bạn gọi `python -m gnn_panels.run_demo`.

Đầu tiên, hệ thống sinh ra hàng trăm thiết kế panel ngẫu nhiên. Mỗi thiết kế là một dict Python chứa các tham số vật lý (số gân, kích thước, áp suất, BC). Việc sinh ngẫu nhiên dùng numpy với seed cố định, đảm bảo lặp lại được giữa các lần chạy.

Thứ hai, mỗi thiết kế được chuyển thành một đối tượng đồ thị PyG. Các đơn vị cấu trúc trở thành đỉnh, các mối hàn trở thành cạnh, và 8 thông số được mã hóa thành vector đặc trưng. Đồng thời, các stress giả lập được tính toán cho từng đơn vị và làm phẳng thành vector 200 chiều, đóng vai trò nhãn (label) y.

Thứ ba, một normalizer được fit trên tập train — tính min/max của mọi feature và label — rồi áp dụng cho cả ba tập (train/val/test). Bước này quan trọng vì các feature có thang đo rất khác nhau (chiều dài cỡ 1000 mm, áp suất cỡ 0.1 MPa), không chuẩn hóa thì gradient sẽ không đều.

Thứ tư, các đồ thị được đóng gói vào DataLoader của PyG. Khi loader tạo một batch, nó GỘP nhiều đồ thị nhỏ thành một đồ thị lớn (bằng cách nối edge_index và x), nhưng vẫn giữ thuộc tính `batch.batch` để biết node nào thuộc panel nào. Đây là cơ chế chuẩn của PyG cho phép xử lý batch của các graph có size khác nhau.

Thứ năm, model GraphSAGE forward pass: nhận `(x, edge_index)` của batch, chạy qua các SAGEConv + BatchNorm + tanh, cuối cùng qua linear head để ra `(num_nodes, 200)`. Loss MSE so sánh với label. Optimizer Adam cập nhật trọng số theo gradient.

Thứ sáu, sau mỗi epoch, evaluation trên val set tính ra MSE và max stress accuracy. Nếu val loss cải thiện, lưu checkpoint. Nếu không cải thiện trong nhiều epoch, dừng sớm.

Thứ bảy, sau khi train xong, predict.py load best checkpoint, sinh một panel test mới (với seed nằm ngoài tập train), chạy qua model, đảo ngược chuẩn hóa để có giá trị MPa thật, và in bảng so sánh predicted vs true. Nếu matplotlib có sẵn, vẽ contour stress và lưu ảnh.

Toàn bộ pipeline mất vài phút trên CPU, vài chục giây trên GPU thông thường.

---

## 8. Hướng dẫn mở rộng

Phần này hướng dẫn bạn sửa đổi cụ thể từng phần để mở rộng pipeline.

### 8.1. Thay dữ liệu giả lập bằng dữ liệu FEM thật

Đây là việc lớn nhất khi chuyển từ prototype sang nghiên cứu thật. Bạn cần làm hai việc.

Việc thứ nhất là sinh raw FEM data. Bạn cần một bộ script ABAQUS để chạy parametric study. Có thể tham khảo tài liệu `GNN_Stiffened_Panels_Mentor_Guide.md` mục 5.1 trong cùng thư mục — nó có pseudo-code cho `sample_designs.py`, `build_model.py`, `extract_stress.py`. Đầu ra mong đợi của bước này là, cho mỗi thiết kế, một file `.npz` chứa stress field cho từng đơn vị cấu trúc dưới dạng mảng 2D.

Việc thứ hai là sửa code trong `data/synthetic_panel.py`. Cụ thể, thay hàm `generate_pseudo_stress_field()` bằng hàm load stress từ file `.npz` của ABAQUS. Bạn cần thêm bước nội suy stress field từ mesh ABAQUS về lưới chuẩn $10 \times 20$, vì các đơn vị có hình dạng khác nhau. Có thể dùng `scipy.interpolate.griddata` cho việc này:

```python
from scipy.interpolate import griddata

def load_real_stress(unit_type, design, unit_idx, abaqus_data):
    raw = abaqus_data[f"{unit_type}_{unit_idx}"]   # nodes + stress values
    nodes = raw["coords"][:, 1:3]    # 2D coords
    values = raw["mises"]
    # nội suy về 10x20
    rows, cols = cfg.STRESS_GRID
    grid_x, grid_y = np.meshgrid(...)
    grid = griddata(nodes, values, (grid_x, grid_y), method='cubic')
    return grid.flatten()
```

Phần model, training, evaluation **không cần đổi gì**. Đây là sức mạnh của thiết kế module hóa.

### 8.2. Đổi kiến trúc mô hình

Nếu bạn muốn thử Graph Attention Network (GAT) thay vì GraphSAGE, sửa `models/graphsage.py` (hoặc tạo file mới `models/gat.py`):

```python
from torch_geometric.nn import GATConv

class StiffenedPanelGAT(nn.Module):
    def __init__(self, in_dim=8, hidden=64, n_heads=4, n_layers=8, out_dim=200):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden // n_heads, heads=n_heads))
        self.bns.append(nn.BatchNorm1d(hidden))
        for _ in range(n_layers - 1):
            self.layers.append(GATConv(hidden, hidden // n_heads, heads=n_heads))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x, edge_index):
        for conv, bn in zip(self.layers, self.bns):
            x = torch.tanh(bn(conv(x, edge_index)))
        return self.head(x)
```

Sau đó trong `train.py`, đổi `from .models import StiffenedPanelGNN` thành `from .models.gat import StiffenedPanelGAT`. Trong `config.py`, có thể thêm tham số `n_heads`.

### 8.3. Thêm feature mới cho node

Giả sử bạn muốn thêm "vật liệu" làm feature (nếu dataset có nhiều loại thép). Hai bước:

Bước thứ nhất, sửa `data/graph_builder.py`: trong các hàm `_make_plate_feature`, `_make_web_feature`, `_make_flange_feature`, thêm giá trị Young's modulus và/hoặc yield stress vào cuối list. Vector feature giờ dài 9 hoặc 10 chiều.

Bước thứ hai, sửa `config.py`: đổi `MODEL["in_dim"]` từ 8 lên 9 hoặc 10. Toàn bộ phần model sẽ tự thích ứng nhờ `SAGEConv(in_dim, hidden)` đọc giá trị này.

### 8.4. Thêm edge feature

Nếu sau này bạn muốn phân biệt các loại mối hàn (full weld, partial weld, bolted), bạn cần edge feature. Việc này phức tạp hơn vì SAGEConv không hỗ trợ edge feature tự nhiên — bạn phải đổi sang `NNConv` hoặc `GeneralConv` của PyG. Sửa trong `models/graphsage.py`:

```python
from torch_geometric.nn import NNConv

class StiffenedPanelGNN_EdgeFeat(nn.Module):
    def __init__(self, in_dim=8, edge_dim=2, hidden=64, n_layers=8, out_dim=200):
        super().__init__()
        # edge MLP để biến edge feature thành ma trận trọng số
        edge_mlp = nn.Sequential(nn.Linear(edge_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, in_dim * hidden))
        self.layers = nn.ModuleList([NNConv(in_dim, hidden, edge_mlp, aggr='sum')])
        # ... các layer tiếp tương tự
```

Đồng thời trong `graph_builder.py`, thêm `edge_attr` khi tạo `Data`:

```python
edge_attr = torch.tensor([weld_type_one_hot for _ in edges], dtype=torch.float)
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
```

### 8.5. Tăng độ phân giải output

Nếu bạn muốn lưới sampling stress dày hơn (vd $20 \times 40 = 800$ chiều thay vì 200), chỉ cần sửa hai chỗ trong `config.py`:

```python
STRESS_GRID = (20, 40)
MODEL["out_dim"] = 800
```

Phần model tự thích ứng. Lưu ý rằng GPU memory và thời gian train sẽ tăng tỷ lệ với `out_dim`, và cần nhiều dữ liệu hơn để model học được pattern chi tiết hơn.

### 8.6. Thay aggregator

Trong `config.py`, sửa `MODEL["aggr"]` từ `"sum"` sang `"mean"` hoặc `"max"`. Đây là cách tốt để hiểu vì sao paper chọn `sum` — bạn sẽ thấy `mean` cho accuracy thấp hơn ở các panel có số stiffener thay đổi nhiều.

### 8.7. Thêm physics-informed loss

Một hướng nghiên cứu mới: thêm loss term đảm bảo cân bằng lực (force balance). Trong `train.py`, thêm:

```python
def physics_loss(pred, batch):
    # Tổng lực trên mỗi panel phải gần bằng tải áp dụng
    # (cần biết area của mỗi unit để tính)
    ...
    return torch.mean(force_imbalance ** 2)

# Trong vòng lặp:
loss = loss_fn(pred, batch.y) + 0.1 * physics_loss(pred, batch)
```

Đây là hướng "physics-informed neural network" — kết hợp loss data-driven với loss vật lý.

---

## 9. Tham khảo nhanh các file

| File | Mục đích | Khi nào sửa |
|------|---------|-------------|
| `config.py` | Siêu tham số tập trung | Khi muốn đổi kích thước model, learning rate, batch size |
| `data/synthetic_panel.py` | Sinh dữ liệu giả lập | Khi muốn thay bằng dữ liệu ABAQUS thật |
| `data/graph_builder.py` | Xây graph từ panel | Khi đổi cách định nghĩa node/edge, thêm loại unit |
| `data/dataset.py` | PyG Dataset wrapper | Hiếm khi cần sửa |
| `models/graphsage.py` | Kiến trúc mạng | Khi muốn thử kiến trúc khác (GAT, GIN, NNConv) |
| `utils/normalize.py` | Chuẩn hóa feature | Khi muốn dùng z-score thay vì min-max |
| `utils/metrics.py` | Đo độ chính xác | Khi muốn thêm metric mới |
| `train.py` | Vòng lặp huấn luyện | Khi muốn đổi optimizer, scheduler, thêm physics loss |
| `predict.py` | Inference + visualize | Khi muốn predict trên dữ liệu mới |

---

## 10. Một số lỗi thường gặp và cách xử lý

Nếu khi import `torch_geometric` báo lỗi thiếu `torch_scatter` hay `torch_sparse`, bạn cần cài đặt phù hợp với phiên bản PyTorch và CUDA của mình. Truy cập trang chính thức của PyTorch Geometric để tìm lệnh cài đặt cho đúng combination phiên bản.

Nếu loss trở thành NaN sau vài epoch, nguyên nhân thường là learning rate quá cao hoặc gradient bị explosion. Trong code có sẵn `clip_grad_norm_` để chống, nhưng nếu vẫn NaN, hãy giảm `learning_rate` trong `config.py` xuống còn 0.001 hoặc 0.0005.

Nếu val accuracy không cải thiện qua 50 epoch, có thể model quá nhỏ so với độ phức tạp dữ liệu. Tăng `MODEL["n_layers"]` lên 12-16, hoặc tăng `MODEL["hidden"]` lên 128.

Nếu bạn dùng dataset rất nhỏ (vd 100 mẫu) và val accuracy cao mà test accuracy thấp, đó là overfitting. Tăng `weight_decay` trong config, giảm số layer, hoặc tăng dataset.

---

## 11. Tham khảo

- Paper gốc: Cai, Y., & Jelovica, J. (2024). *Efficient graph representation in graph neural networks for stress predictions in stiffened panels*. Thin-Walled Structures 203, 112157.
- Tài liệu mentor guide đi kèm: `../GNN_Stiffened_Panels_Mentor_Guide.md` — chứa thêm phân tích sâu, pseudo-code ABAQUS, pitfalls khi reproduce.
- Báo cáo khoa học đi kèm: `../Bao_Cao_Khoa_Hoc_GNN_Stiffened_Panels.md` — văn phong học thuật, dùng cho seminar/luận văn.
- PyTorch Geometric docs: https://pytorch-geometric.readthedocs.io/
- GraphSAGE paper: Hamilton et al. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.
