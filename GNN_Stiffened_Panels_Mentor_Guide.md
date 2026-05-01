# Hiểu sâu paper: Efficient Graph Representation in GNN for Stress Predictions in Stiffened Panels

> **Tác giả paper:** Yuecheng Cai, Jasmin Jelovica
> **Tạp chí:** Thin-Walled Structures 203 (2024) 112157
> **Mục tiêu tài liệu này:** giúp bạn hiểu *bản chất* của paper đến mức có thể tự tái hiện (reproduce) được nghiên cứu mà không cần nhìn lại paper.
> **Phong cách:** trực giác trước → phân tích sâu → công thức → thảo luận trade-off → kiểm tra hiểu.

---

## Cách đọc tài liệu này (logic flow)

Tài liệu được viết theo **một dòng chảy lập luận liên tục**, mỗi phần giải đáp một câu hỏi mà phần trước đặt ra:

```
"Bài toán này khó ở đâu?"  (Phần 1)
                ↓
"Vì sao FEM không đủ?"
                ↓
"Vậy dùng ML — nhưng loại nào?"  (Phần 2)
                ↓
"Vì sao MLP/CNN sai? GNN có gì đặc biệt?"
                ↓
"Đã chọn GNN — biểu diễn kết cấu thành graph thế nào?"  (Phần 3)
                ↓
"Có graph rồi — mạng học từ nó như thế nào?"  (Phần 4)
                ↓
"Đã có cơ chế học — quy trình toàn cục là gì?"  (Phần 5)
                ↓
"Hiểu rồi — làm sao tự reproduce?"  (Phần 6)
                ↓
"Tự kiểm tra hiểu sâu"  (Phần 7)
                ↓
"Đánh giá tới hạn của paper"  (Phần 8)
```

Đừng đọc lướt — mỗi phần đều "móc xích" với phần trước. Nếu bạn vướng phần nào, quay lại phần trước nó để tìm gốc rễ.

---

## Mục lục

1. [Hiểu bài toán — và vì sao nó khó về mặt bản chất](#1-hiểu-bài-toán--và-vì-sao-nó-khó-về-mặt-bản-chất)
2. [Tại sao GNN là lựa chọn tất yếu, không phải tùy chọn](#2-tại-sao-gnn-là-lựa-chọn-tất-yếu-không-phải-tùy-chọn)
3. [Cách paper "nhìn thế giới" — Graph Representation và bài học triết lý](#3-cách-paper-nhìn-thế-giới--graph-representation-và-bài-học-triết-lý)
4. [Cơ chế Message Passing — vật lý gặp toán học](#4-cơ-chế-message-passing--vật-lý-gặp-toán-học)
5. [Pipeline đầy đủ — và những điểm dễ sai khi reproduce](#5-pipeline-đầy-đủ--và-những-điểm-dễ-sai-khi-reproduce)
6. [Hướng dẫn tự reproduce — kèm phân tích pitfalls](#6-hướng-dẫn-tự-reproduce--kèm-phân-tích-pitfalls)
7. [Kiểm tra hiểu sâu — 7 câu hỏi (5 cơ bản + 2 nâng cao)](#7-kiểm-tra-hiểu-sâu--7-câu-hỏi-5-cơ-bản--2-nâng-cao)
8. [Đánh giá tới hạn paper — góc nhìn của reviewer](#8-đánh-giá-tới-hạn-paper--góc-nhìn-của-reviewer)
9. [Tổng kết và hướng nghiên cứu tiếp](#9-tổng-kết-và-hướng-nghiên-cứu-tiếp)

---

## 1. Hiểu bài toán — và vì sao nó khó về mặt bản chất

### 1.1. Hình dung trước đã

Hãy tưởng tượng một **tấm thép phẳng có gân gia cố** — kiểu như đáy tàu thủy, vỏ máy bay, hoặc sàn cầu thép. Đó là một **plate** (tấm phẳng) được "hàn dính" thêm các **stiffeners** (thanh gia cường) chạy dọc, đôi khi chạy ngang.

Gân làm nhiệm vụ **chống cong vênh** — giống xương sườn của con người: tấm da mỏng nhưng có khung xương đỡ thì cứng vững. Nhờ stiffener, panel có thể vừa **nhẹ** (vì plate mỏng) vừa **bền** (vì gân chống vênh). Đây là lý do stiffened panel là *building block* của hầu hết kết cấu vỏ mỏng (thin-walled structures) trong tàu, máy bay, cầu, bồn chứa.

Khi tải trọng (áp suất nước, sóng, hàng hóa) ép lên panel, **mỗi điểm trên kết cấu sẽ chịu một mức ứng suất khác nhau**. Có chỗ căng, có chỗ chùng, có "điểm nóng" cực kỳ nguy hiểm — thường là chỗ nối giữa plate và stiffener.

### 1.2. Input → Output của bài toán

| Hạng mục | Nội dung |
|---|---|
| **Input** | Hình học panel (số stiffener, kích thước plate/web/flange, độ cong), điều kiện biên (BC) trên các cạnh, tải trọng (uniform / patch) |
| **Output** | Phân bố **ứng suất von Mises** trên *toàn bộ* kết cấu — không phải 1 con số, mà là một trường ứng suất 3D |

> **Điểm cần khắc sâu:** Đây là bài toán **field prediction** (dự đoán *trường* — function trên cả miền không gian), không phải **scalar regression** (dự đoán 1 con số). Đây là điểm khiến bài toán khó hơn nhiều so với các paper trước (vd dự đoán *buckling load* — chỉ 1 con số). Field prediction đòi hỏi mạng phải có khả năng **biểu diễn không gian**, và đây chính là chỗ GNN với output 200-chiều/node tỏa sáng.

### 1.3. Tại sao bài toán này quan trọng — và quan trọng gấp ngàn lần khi đặt trong vòng tối ưu hóa

Một thiết kế đơn lẻ chỉ cần 1 lần phân tích ứng suất. Nhưng trong thực tế kỹ thuật, bạn không thiết kế 1 panel — bạn **tối ưu hóa**:

- **Sizing optimization**: tìm kích thước (độ dày, chiều cao gân...) tối ưu cho 1 cấu hình.
- **Topology optimization**: tìm cấu hình (số gân, vị trí gân) tối ưu.

Cả hai đều cần **vòng lặp**: thuật toán di truyền (GA), Particle Swarm, Bayesian Optimization... mỗi thuật toán đều đánh giá *hàng ngàn đến hàng chục ngàn* thiết kế.

> **Đây là chỗ "khủng hoảng tính toán" thực sự xảy ra.** Mỗi simulation FEM mất *vài phút đến vài giờ*. Nhân với 10.000 thiết kế = vài tháng đến vài năm CPU time. Đó là lý do tại sao **Reduced-Order Model (ROM)** là một field nghiên cứu lớn — không phải vì FEM kém, mà vì **vòng lặp tối ưu hóa khuếch đại chi phí FEM lên hàng vạn lần**.

Hiểu điều này quan trọng vì: paper KHÔNG đối thủ với FEM (FEM vẫn được dùng để generate training data!). Paper đối thủ với *các ROM khác* — Kriging, RBF, MLP, CNN. Đó là cuộc cạnh tranh thực sự.

### 1.4. Lịch sử ngắn của ROM — để hiểu paper định vị ở đâu

| Thời kỳ | Phương pháp đại diện | Hạn chế chính |
|---|---|---|
| 1990–2000 | **Kriging, RBF, MARS, RSM** (statistical surrogates) | Yêu cầu số biến nhỏ; kém với hình học phi tuyến |
| 2000–2015 | **MLP** (shallow neural networks) | Input fixed-size; không biểu diễn được kết cấu thay đổi |
| 2015–2020 | **CNN, GAN** (image-based ROMs) | Kết cấu phải ép thành ảnh; mất tô-pô |
| 2020–nay | **GNN** (graph-based ROMs) | Mới — paper này là một trong những đầu tiên cho 3D structures |

> **Vị trí của paper:** Paper định vị mình là *bước nhảy thế hệ* — chuyển từ "ép kết cấu thành ảnh" sang "tôn trọng cấu trúc đồ thị tự nhiên của kết cấu".

### 1.5. Nếu KHÔNG dùng AI thì người ta làm gì? — FEM

**FEM (Finite Element Method)** — Phương pháp phần tử hữu hạn.

> **Trực giác:** Bạn không thể giải phương trình vi phân của vật thể liên tục, nên bạn **chia nhỏ** vật thể thành hàng vạn ô vuông tí hon (gọi là "phần tử"), rồi giải xấp xỉ phương trình trong từng ô, rồi "ráp lại" để có lời giải toàn cục.

Giống như đo diện tích một cái hồ kỳ dị: bạn không có công thức trực tiếp, nhưng nếu chia hồ thành 10.000 ô vuông nhỏ rồi cộng lại, bạn sẽ ra kết quả khá chính xác.

Trong paper này, FEM được chạy bằng **ABAQUS** (phần mềm thương mại), với phần tử shell `S4R`, mesh dày: 15 phần tử giữa các stiffener, 15 phần tử cho chiều cao stiffener, 8 cho flange.

#### FEM mạnh ở đâu?

- **Chính xác toán học:** đảm bảo hội tụ về lời giải đúng.
- **Linh hoạt:** xử lý được hình học phức tạp, vật liệu phi tuyến, đủ loại tải trọng.
- **Là "ground truth"** trong cộng đồng nghiên cứu kết cấu.

#### FEM yếu ở đâu — và vì sao 3 điểm đau này KHÔNG thể giải quyết bằng cách "cải tiến FEM"?

| Điểm đau | Bản chất sâu hơn |
|---|---|
| **Cực chậm khi mesh dày** | Đây là *bản chất* của FEM — hàng vạn DOFs phải giải đồng thời (sparse linear system). Có thể tối ưu solver, nhưng không thể *thoát* khỏi $\mathcal{O}(N^{1.5}-N^3)$ |
| **Mỗi thiết kế = 1 simulation độc lập** | FEM không có "memory" giữa các thiết kế. Đây là vấn đề *triết lý* của phương pháp — FEM giải bài toán *cụ thể*, không *học* từ kinh nghiệm |
| **Re-meshing khi đổi hình học** | Mesh phụ thuộc hình học. Đây là vấn đề *cấu trúc dữ liệu* — không thể "re-use" mesh cũ nếu hình thay đổi |

> **Bài học:** 3 điểm đau này KHÔNG thể giải bằng cách làm FEM "thông minh hơn". Phải có **paradigm khác** — đó là ROM. Và trong các ROM, ML/DL có lợi thế riêng vì *có khả năng học từ dữ liệu*, không bị giới hạn bởi giả định toán học cứng nhắc.

### 1.6. Bridge → Phần 2

Đến đây bạn đã hiểu:

1. Bài toán là *field prediction* (dự đoán trường ứng suất).
2. Bối cảnh thực sự là *vòng tối ưu hóa* — đòi hỏi ROM nhanh.
3. FEM tốt nhưng có 3 điểm đau bản chất không sửa được.
4. ML/DL là hướng đi tự nhiên — nhưng trong họ ML có nhiều loại mạng.

→ Câu hỏi tiếp theo: **Trong họ ML, vì sao GNN — chứ không phải MLP, CNN, hay GAN — là lựa chọn đúng?**

---

## 2. Tại sao GNN là lựa chọn tất yếu, không phải tùy chọn

Đây là phần "thấm" nhất, hãy đọc chậm. Câu trả lời "vì dữ liệu là graph" KHÔNG đủ — phải hiểu vì sao MLP và CNN **thất bại**, từ đó GNN mới trở nên **tất yếu**.

### 2.1. Khái niệm chìa khóa: Inductive Bias

Trước khi phân tích từng loại mạng, hãy nắm khái niệm này:

> **Inductive bias** = "thiên kiến cấu trúc" mà mỗi loại mạng *giả định trước* về dữ liệu. Mỗi loại mạng được thiết kế để khai thác một loại cấu trúc cụ thể.

| Mạng | Inductive bias | Phù hợp với |
|---|---|---|
| MLP | Không có — coi mọi feature như độc lập | Tabular data với feature cố định |
| CNN | *Translation invariance* — pattern xuất hiện ở vị trí khác nhau là giống nhau | Ảnh, lưới đều |
| RNN | *Sequential order* — thứ tự quan trọng | Văn bản, chuỗi thời gian |
| **GNN** | ***Permutation invariance + Local connectivity*** — không quan tâm thứ tự node, chỉ quan tâm "ai nối với ai" | Graph, network |

**Ý nghĩa cốt lõi:** Khi inductive bias của mạng *khớp* với bản chất của dữ liệu, mạng học hiệu quả với ít dữ liệu hơn. Khi không khớp, mạng phải *học cả cấu trúc lẫn nội dung* — tốn dữ liệu kinh khủng và dễ overfit.

→ Bài toán stiffened panel: dữ liệu *vốn có* tính chất "permutation invariance" (không quan trọng đánh số stiffener nào trước) và "local connectivity" (mỗi component nối với các component lân cận). GNN khớp 100%. MLP/CNN không khớp.

### 2.2. Nếu dùng MLP — và vì sao thất bại

MLP yêu cầu **input có kích thước cố định** — ví dụ luôn là một vector dài 50 phần tử.

#### Vấn đề lớp 1: Variable-size input

- Tấm có **2 stiffener** → vector mô tả dài 20.
- Tấm có **8 stiffener** → vector dài 80.
- **Hai input khác kích thước → MLP không nuốt nổi.**

Cách "vá" thường thấy và hệ quả:

| Cách vá | Hệ quả tệ |
|---|---|
| Pad zero | MLP học sai — zero không có nghĩa "không có stiffener", nó là một con số thật trong không gian feature, làm mạng học quan hệ giả |
| Cố định trước số stiffener tối đa | Lãng phí, không generalize, không xử lý được thiết kế ngoài giới hạn |
| Train riêng cho từng cấu hình | Phải có hàng chục mạng cho hàng chục cấu hình → không khả thi |

#### Vấn đề lớp 2: Không có permutation invariance

Đây là điểm tinh tế hơn. Giả sử bạn cố định 5 stiffener và sắp xếp feature theo thứ tự `[stiff1, stiff2, stiff3, stiff4, stiff5]`. MLP học được mapping. Nhưng nếu bạn *đổi thứ tự* các stiffener (vd hoán đổi stiff2 và stiff4), MLP sẽ predict ra **kết quả khác**, dù đó là *cùng một panel vật lý*!

→ Để khắc phục, bạn phải **augment data** bằng tất cả các permutation → $5! = 120$ lần dữ liệu. Tốn kinh khủng, và càng vô lý khi N stiffener lớn.

#### Vấn đề lớp 3: Không học được tính lặp lại

MLP thấy "8 thông số đầu vector" và "8 thông số cuối" như hai thực thể không liên quan, dù chúng có thể là 2 stiffener giống hệt nhau. Mạng không biết rằng *cùng một vật lý* (stiffener cao 300mm dày 10mm) cho ra *cùng một stress pattern* — phải học lại nhiều lần.

> **Điểm chết của MLP:** Nó không có *bất kỳ* inductive bias nào phù hợp với kết cấu rời rạc-có-cấu-trúc. Mọi thông tin cấu trúc đều phải mạng "học từ đầu" → cần nhiều dữ liệu, dễ overfit, không generalize.

### 2.3. Nếu dùng CNN — và vì sao thất bại tinh tế hơn

CNN có inductive bias *translation invariance* — "một cạnh ngang ở bên trái cũng giống cạnh ngang ở bên phải". Đây là lý do CNN giỏi với ảnh.

Nhiều paper trước (StressNet 2021, TopologyGAN 2021…) đã làm: chuyển kết cấu thành ảnh 2D, rồi CNN dự đoán "ảnh ứng suất". Tại sao **không** áp dụng cho stiffened panel?

#### Vấn đề 1: Kết cấu 3D không thể "chiếu xuống ảnh" mà không mất thông tin

Stiffened panel có plate phẳng nằm ngang, stiffener đứng dọc. Nếu chiếu xuống thành ảnh top-view → mất chi tiết stiffener. Nếu xếp side-by-side (plate + stiffener bên cạnh nhau) → mất quan hệ kết nối vật lý giữa chúng.

#### Vấn đề 2: "Hàng xóm" của CNN sai về mặt vật lý

CNN dùng kernel 3×3 hoặc 5×5 — coi 8 pixel xung quanh là "hàng xóm". Nhưng tại điểm hàn giữa plate và stiffener:

- *Hàng xóm theo CNN*: 8 pixel cùng nằm trên ảnh.
- *Hàng xóm thật về mặt vật lý*: plate ở dưới + stiffener ở trên (đi qua mối hàn).

→ CNN học sai mối quan hệ vật lý.

#### Vấn đề 3: Variable-size lại tái xuất

Nếu số stiffener thay đổi → ảnh size thay đổi → phải resize hoặc pad → phá inductive bias *translation invariance* (vì pixel "đen" do pad không nên có ý nghĩa giống pixel kết cấu).

#### Vấn đề 4: Translation invariance KHÔNG đúng cho kết cấu

Một stiffener ở giữa panel chịu tải khác hẳn một stiffener ở mép (do BC khác). CNN coi chúng giống nhau → predict sai. *Translation invariance là LỢI THẾ với ảnh, nhưng là HẠN CHẾ với kết cấu.*

> **Điểm chết của CNN:** Inductive bias của nó (translation invariance trên lưới đều) **trái ngược** với bản chất kết cấu (BC tại mép quan trọng, kết nối tô-pô quan trọng). Càng ép vào, càng phản tác dụng.

### 2.4. GNN — và vì sao đây là điểm đến tự nhiên

GNN có hai siêu năng lực, mỗi cái giải đúng một điểm đau:

| Năng lực | Giải đúng vấn đề |
|---|---|
| **Permutation invariance** (qua aggregator như sum/mean) | Hoán đổi thứ tự node → kết quả không đổi. Không cần data augmentation |
| **Local connectivity** (qua message passing) | Chỉ truyền thông tin giữa node thực sự kết nối. "Hàng xóm" theo đúng nghĩa vật lý |

Thêm một điểm tinh tế:

- **Variable-size input native**: 2 stiffener hay 8 stiffener đều là một graph hợp lệ, cùng một mạng xử lý được tất cả.

> **Cảm giác "tất yếu":** Stiffened panel **vốn dĩ đã là một graph** rồi — các tấm con + các gân là *node*, các mối hàn là *edge*. Dùng GNN không phải là "ép" — mà là "tôn trọng cấu trúc tự nhiên". Mọi cách khác đều phải bóp méo dữ liệu để vừa khuôn.

### 2.5. Trong họ GNN — vì sao chọn GraphSAGE?

GNN không phải một mạng duy nhất, mà là *một họ*. Paper chọn **GraphSAGE** thay vì GCN, GAT, GIN, ChebNet... Vì sao?

| GNN variant | Đặc điểm | Vì sao paper KHÔNG chọn |
|---|---|---|
| **Spectral CNN, ChebNet** | Convolution dùng Laplacian spectrum | Memory cực lớn, không scale với graph có size thay đổi |
| **GCN (Kipf 2017)** | Convolution spatial đơn giản | Aggregator là *normalized mean* — mất thông tin số hàng xóm |
| **GAT** | Có attention giữa các edge | Nặng hơn, training khó hơn; chưa cần với graph nhỏ |
| **GIN** | Mạnh nhất về biểu diễn (Weisfeiler-Lehman) | Phức tạp, dễ overfit nếu data không lớn |
| **GraphSAGE** ✅ | Spatial, sampling-friendly, aggregator linh hoạt | **Cân bằng độ mạnh và tốc độ; aggregator có thể là sum** |

Lý do chọn GraphSAGE cụ thể trong paper:

1. **Spatial method** — không cần Laplacian eigendecomposition → memory thấp.
2. **Hỗ trợ aggregator là `sum`** — quan trọng để bảo toàn thông tin số hàng xóm (xem lý do trong Phần 4).
3. **Sampling-friendly** — có thể scale lên nếu sau này cần graph lớn (mặc dù paper này không dùng sampling, nhưng chọn vẫn tương lai-thân-thiện).
4. **Đơn giản, ổn định** — dễ debug, dễ tune.

### 2.6. Bridge → Phần 3

Bạn đã hiểu:

1. Mỗi loại mạng có inductive bias riêng — phải khớp với bản chất dữ liệu.
2. MLP/CNN không khớp với kết cấu — fail vì lý do *bản chất*, không phải *cài đặt*.
3. GNN khớp tự nhiên — và GraphSAGE là biến thể cân bằng nhất.

→ Câu hỏi tiếp theo: **Đã chọn GNN — nhưng cụ thể, một stiffened panel được biến thành graph như thế nào? Đây là chỗ paper sáng tạo nhất.**

---

## 3. Cách paper "nhìn thế giới" — Graph Representation và bài học triết lý

**Đây là phần đắt giá nhất của paper.** Đây cũng là chỗ paper **sáng tạo nhất**. Đọc thật chậm.

### 3.1. Định nghĩa graph (siêu nhanh, chỉ ôn lại)

Một graph $G = (V, E, A)$:

- $V$ = tập **node** (đỉnh)
- $E$ = tập **edge** (cạnh)
- $A \in \mathbb{R}^{N \times N}$ = ma trận kề; $A_{ij} = 1$ nếu có edge giữa node $i, j$
- Mỗi node có vector đặc trưng $x_i \in \mathbb{R}^D$
- Trong paper này, graph là **vô hướng** (undirected), nghĩa là $A_{ij} = A_{ji}$

### 3.2. Hai cách nhìn thế giới — đây là **trận đấu** của paper

Vấn đề cốt lõi: *Mỗi node trong graph đại diện cho cái gì về mặt vật lý?* Có hai cách trả lời, và đây là chỗ paper *thắng* paper trước.

#### Cách 1: FE-vertex graph (cách kinh điển, dùng trong CFD)

Mỗi *phần tử hữu hạn* (mỗi ô vuông trong mesh ABAQUS) → 1 node trong graph.
Hai phần tử kề nhau trong mesh → 1 edge.

**Triết lý**: "Giữ nguyên FEM, chỉ đổi solver từ linear system → GNN."

**Hệ quả:**

- Nếu mesh có 50.000 phần tử → graph có 50.000 node.
- Mesh càng dày → graph càng to → GPU càng "ngốn".
- 23.4 GB GPU memory cho batch size 64. 6.94 giây/epoch.

Cách này được dùng nhiều trong CFD GNN (vd MeshGraphNet của DeepMind 2020) vì CFD thực sự *cần* mỗi điểm mesh — flow field thay đổi liên tục theo từng pixel.

#### Cách 2: Structural unit-vertex graph (đề xuất của paper)

Mỗi *thành phần kết cấu* → 1 node. Trong stiffened panel, có **3 loại thành phần**:

1. **Plate-span** — tấm con giữa 2 stiffener (hoặc giữa stiffener và mép ngoài)
2. **Web** — phần đứng của stiffener
3. **Flange** — phần ngang của stiffener (nếu có — gân chữ T)

**Triết lý**: "Kết cấu rời rạc *vốn* đã có cấu trúc tự nhiên — hãy tôn trọng nó."

Một stiffened panel với 5 stiffener T có:

- 6 plate-spans
- 5 webs
- 5 flanges

→ **Chỉ 16 node tất cả.** (So với 50.000 trong cách kinh điển!)

Edge nối 2 node nếu 2 thành phần có **mối hàn vật lý** với nhau.

### 3.3. Phân tích sâu: Tại sao "structural unit" là level abstraction đúng?

Đây là chỗ cần suy nghĩ kỹ. Tại sao không phải các level khác?

| Level | Số node | Có hợp lý không? | Tại sao |
|---|---|---|---|
| 1 element / 1 node | 10.000+ | ❌ Quá chi tiết — thừa thãi | Mesh không phản ánh cấu trúc rời rạc |
| **1 unit (plate-span/web/flange) / 1 node** | ~16 | ✅ **Vừa đủ** | Khớp với *các thành phần được sản xuất riêng và hàn lại* |
| 1 stiffener (cả web + flange) / 1 node | ~10 | ⚠️ Mất chi tiết | Web và flange chịu stress khác nhau, không nên gộp |
| Cả panel / 1 node | 1 | ❌ Quá thô | Không có graph để học |

> **Bài học triết lý:** Level abstraction đúng là level mà tại đó *vật lý không bị mất, nhưng dư thừa thông tin được loại bỏ*. Trong stiffened panel, mỗi structural unit là một "domain" có:
> - Hình học đồng nhất (rectangle/curved rectangle)
> - BC đồng nhất trên các cạnh
> - Stress field tương đối "smooth" trong unit, nhảy bậc tại biên (chỗ hàn)
>
> → Nén toàn bộ unit thành 1 node là *hợp lý vật lý*. Mất chi tiết ở mức element nhưng *không mất bản chất*.

Đây cũng là cách classical mechanics đã làm hàng trăm năm — *domain decomposition*. Paper chỉ là đem ý tưởng này vào GNN.

### 3.4. Node feature — đại diện cho cái gì về mặt vật lý?

Mỗi node giữ **8 số** (input):

| Feature | Ý nghĩa vật lý |
|---|---|
| width | bề rộng của thành phần |
| length | chiều dài |
| thickness | độ dày |
| BC cạnh 1 | điều kiện biên (free / simply supported / fixed) |
| BC cạnh 2 | nt |
| BC cạnh 3 | nt |
| BC cạnh 4 | nt |
| pressure | áp suất tác dụng lên thành phần |

(Khi có patch loading — tải tập trung — thêm tọa độ vị trí lực.)

#### Phân tích: Tại sao đủ 8 feature là đủ?

Câu hỏi sâu: *Vì sao không cần thêm feature về vật liệu (E, ν)? Vị trí trong panel? Chiều cong?*

- **Vật liệu**: trong từng case study, vật liệu cố định → không cần feature. Nếu vật liệu thay đổi giữa các sample, phải thêm.
- **Vị trí trong panel**: KHÔNG cần! Vì "vị trí" được mã hóa qua **edge** (graph topology). Một plate-span ở mép sẽ có ít hàng xóm hơn plate ở giữa → mạng tự suy được vị trí qua connectivity.
- **Chiều cong**: trong case 6 (panel cong), curvature được encode vào geometry feature mở rộng.

> **Đây là điểm tinh tế của graph representation:** Nhiều thông tin về *vị trí tương đối* được "miễn phí" qua topology — không cần nhồi vào node feature.

#### Tại sao BC mã hóa thành 4 số (1 cho mỗi cạnh) thay vì 1 số?

Một plate-span có 4 cạnh, mỗi cạnh có thể có BC khác nhau. Nếu chỉ 1 số → mất chi tiết. Phải 4 số.

#### Encoding scheme cho BC?

Paper không nói rõ, nhưng thông thường: `0 = free, 1 = simply supported, 2 = fixed`. Hoặc one-hot: `[0,0,0]`, `[0,1,0]`, `[0,0,1]`. Khi reproduce, bạn nên thử cả hai và xem cái nào ổn định hơn.

### 3.5. Edge — mang ý nghĩa gì? Và vì sao paper không gán feature cho edge?

Edge **không có feature** trong paper này (chỉ là tô-pô). Nó chỉ nói: "node này nối với node kia". Nhưng ý nghĩa vật lý cực mạnh:

> **Edge = mối hàn = đường truyền ứng suất giữa hai thành phần.**

Khi plate biến dạng, ứng suất truyền sang stiffener qua đường hàn — và trong graph, thông tin sẽ được "truyền" qua edge tương ứng. Đây là điểm khớp giữa **vật lý** và **toán học**.

#### Thảo luận: Có nên gán feature cho edge không?

Một mở rộng tiềm năng: gán cho mỗi edge feature như "độ dài mối hàn", "loại mối hàn", "stiffness của connection". Paper KHÔNG làm điều này — và đây có thể là một điểm yếu.

**Lập luận của paper (ngầm):** Trong stiffened panel hàn cứng, mọi edge (mối hàn) đều giống nhau về tính chất → không cần phân biệt.

**Phản biện:** Trong panel có *bolted connection* hoặc *partial weld*, edge feature sẽ quan trọng. Đây là một hướng mở rộng cho future work.

### 3.6. Vì sao cách mới tiết kiệm tài nguyên — phân tích định lượng

Độ phức tạp tính toán của GNN tỉ lệ tuyến tính với **N** (số node):

$$
\mathcal{O}\left(N \times \sum_{i=1}^{L} F_{i-1} F_i\right) \;+\; \mathcal{O}\left(N \times k \times \sum_{i=1}^{L} F_i\right) \;\propto\; \mathcal{O}(N)
$$

Trong đó:

- $N$ = số node
- $L$ = số layer
- $F_i$ = embedding size tại layer $i$
- $k$ = số hàng xóm trung bình
- Term thứ nhất là **feature transformation**: $W^l \cdot h$
- Term thứ hai là **aggregation**: $\sum_{u \in \mathcal{N}(v)} h_u$

#### Phân tích sâu hơn

Vì $\sum F_{i-1} F_i$ và $k \sum F_i$ là *hằng số* (không phụ thuộc N) → cả time và memory đều **scale tuyến tính với N**.

→ Giảm N từ 10.000 → 16 = giảm chi phí *600 lần lý thuyết*. Trong thực tế, paper báo cáo *~27 lần nhanh hơn* — vì GraphSAGE aggregation không hoàn toàn thuần $\mathcal{O}(N)$ (có overhead khác).

Cụ thể trong paper:

| Cách | Số node N | GPU memory | Time/epoch | Ghi chú |
|---|---|---|---|---|
| FE-vertex | ~10.000+ | 23.4 GB | 6.94 s | Phụ thuộc mesh |
| Unit-vertex | ~16–30 | 0.5 GB | 0.26 s | Phụ thuộc số thành phần |

→ Nhanh hơn **~27 lần**. Tiết kiệm **~98% memory**.

> **Trực giác sâu hơn:** Tại sao có thể "bỏ" hàng vạn node mà vẫn đủ thông tin? Vì hình học của stiffened panel là **lặp lại** và **có cấu trúc** — bạn không cần kể từng pixel để mô tả nó, bạn chỉ cần nói "có 5 stiffener, mỗi cái cao bao nhiêu, dày bao nhiêu". Cách kinh điển *quên đi cấu trúc* và mô tả thừa thãi.
>
> Đây là một bài học sâu hơn cho deep learning: **abstraction đúng level luôn thắng brute-force**.

### 3.7. Output — node "biết" gì sau khi mạng predict?

Với mỗi node (ví dụ một plate-span), output là một **vector 200 chiều** = lưới ứng suất 10×20 đã được flatten.

Tức là: mạng dự đoán **trường ứng suất 2D** trên mỗi thành phần, tại 200 điểm sampling. Sau đó bạn ráp các unit lại → contour stress 3D toàn panel.

#### Điểm tinh tế

Một node trong graph KHÔNG predict 1 con số ứng suất, mà predict **cả một map ứng suất** trải trên thành phần đó. Đây là cách paper "nén" thông tin từ FEM (vạn điểm) thành chỉ 200 điểm/thành phần — vẫn đủ chi tiết để vẽ contour stress.

#### Phân tích trade-off

Phụ lục B của paper chứng minh: 10×20 là choice cân bằng giữa độ chi tiết và chi phí tính toán.

| Grid size | Pros | Cons |
|---|---|---|
| 5×10 (50 dim) | Memory thấp, train nhanh | Mất chi tiết stress concentration |
| **10×20 (200 dim)** ✅ | Cân bằng | — |
| 30×60 (1800 dim) | Chi tiết tối đa | Tốn memory, có thể overfit, cải thiện không đáng kể |

> **Quy tắc nhỏ:** Output dimension nên được chọn gần *Nyquist* của bài toán — đủ để bắt được pattern không gian quan trọng nhất, không hơn. Với stiffened panel, stress thường có 1-2 cực đại trên mỗi unit — 10×20 đủ để localize chính xác.

#### Bonus: Tại sao output flatten thành vector thay vì giữ 2D?

Vì GNN node embedding *bản chất* là vector. Để giữ 2D, phải dùng GNN convolutional (như MeshGraphNet — phức tạp hơn nhiều). Paper chọn cách đơn giản: flatten + reshape lại sau predict.

### 3.8. Bridge → Phần 4

Bạn đã hiểu:

1. Triết lý "structural unit = node" là sự tôn trọng cấu trúc tự nhiên của panel.
2. Node feature (8 số) đủ vì topology graph mã hóa thông tin vị trí.
3. Edge không có feature — đại diện đường truyền ứng suất.
4. Output 200-dim/node là field prediction cấp unit.

→ Câu hỏi tiếp theo: **Đã có graph + features — mạng học từ chúng như thế nào? Quá trình gì xảy ra trong 32 layer GraphSAGE?**

---

## 4. Cơ chế Message Passing — vật lý gặp toán học

Bây giờ vào **trái tim** của GNN.

### 4.1. Trực giác: "tin đồn lan trong làng"

Tưởng tượng mỗi node là **một người dân trong làng**. Mỗi người ban đầu chỉ biết về *bản thân mình* (8 thông số: tôi rộng bao nhiêu, dày bao nhiêu...).

**Layer 1 (vòng tin đồn 1):**

- Mỗi người gặp **hàng xóm trực tiếp** (các thành phần liền kề), hỏi: "Cậu thế nào?"
- Sau đó tổng hợp thông tin từ hàng xóm + thông tin bản thân → **cập nhật hiểu biết** của mình.
- Sau layer 1: mỗi node biết về *bản thân + hàng xóm cấp 1*.

**Layer 2 (vòng tin đồn 2):**

- Lặp lại. Nhưng vì hàng xóm đã biết về hàng xóm-của-hàng-xóm rồi, nên thông tin đó cũng được truyền về.
- Sau layer 2: mỗi node biết về *bản thân + hàng xóm cấp 1 + hàng xóm cấp 2*.

**Layer L:** node biết được "vùng ảnh hưởng" rộng bằng L bước trên graph.

> **Tương tự với vật lý:** đây *chính xác* là cách ứng suất lan truyền trong kết cấu. Tải trọng tác dụng tại 1 chỗ → biến dạng lan ra hàng xóm → rồi lan tiếp. Sau "đủ vòng lan", trạng thái cân bằng thiết lập trên toàn cấu trúc. **Message passing là mô phỏng số hóa của quá trình lan truyền cơ học.**

### 4.2. Khái niệm "Receptive Field" — vay mượn từ CNN

Tương tự CNN có *receptive field* (vùng ảnh mà 1 neuron "thấy"), GNN cũng có:

> **Receptive field của node v sau L layer** = tập tất cả node có khoảng cách $\leq L$ trên graph từ v.

Sau 32 layer, mỗi node đã "thấy" toàn bộ panel (vì panel ~16-30 node, đường kính graph thường ≤ 5-10).

#### Vì sao chọn 32 layer? (Và đây là một bí ẩn có tính thảo luận)

Paper dùng 32 layer mặc dù panel chỉ ~16-30 node. Theo lý thuyết, sau ~10 layer mỗi node đã "thấy" toàn bộ graph. Vì sao 32?

Có 2 giả thuyết:

1. **Tăng depth ≠ tăng receptive field thuần túy.** Mỗi layer thêm còn làm phép biến đổi phi tuyến → tăng *capacity* biểu diễn của mạng. 32 layer là "deep enough" để học mapping phức tạp giữa geometry và stress field.

2. **Chống over-smoothing tự nhiên qua kiến trúc.** Trong GNN sâu, tin đồn lan quá nhiều khiến mọi node converge về 1 giá trị (over-smoothing). Paper dùng *batch normalization* và *residual* (qua công thức $h_v + \sum h_u$ trong GraphSAGE) để chống điều này.

#### Cảnh báo: Over-smoothing là vấn đề thật

Đây là một trong những thách thức lớn của deep GNN. Nếu không cẩn thận, sau ~10-20 layer thì mọi node embedding trở nên giống nhau → mất khả năng phân biệt → predict kém.

Paper *không* thảo luận sâu về over-smoothing (đây là một điểm yếu của paper về mặt phân tích). Khi reproduce, bạn nên:

- Theo dõi node embedding diversity qua các layer
- Thử so sánh 8, 16, 32 layer xem có cải thiện thật không

### 4.3. Công thức Message Passing tổng quát

$$
\mathbf{h}_v^l = \gamma_\Theta\!\left(\mathbf{h}_v^{l-1},\; \bigoplus_{u \in \mathcal{N}(v)} \phi_\Theta(\mathbf{h}_v^{l-1}, \mathbf{h}_u^{l-1}, \mathbf{e}_{u,v})\right)
$$

Đọc theo trực giác:

- $\mathbf{h}_v^l$ = "hiểu biết của node $v$ tại vòng $l$"
- $\bigoplus$ = phép tổng hợp (sum/mean/max) — **bất biến với hoán vị** (hàng xóm không có thứ tự!)
- $\phi_\Theta$ = "biến đổi thông điệp" trước khi gửi đi (message function)
- $\gamma_\Theta$ = "cập nhật" hiểu biết của bản thân (update function)

Quá trình có **3 giai đoạn** trong mỗi layer:

1. **Message** — node $u$ tạo message gửi cho node $v$, là $\phi_\Theta(h_v, h_u, e_{u,v})$
2. **Aggregate** — node $v$ gom tất cả message từ hàng xóm bằng $\bigoplus$
3. **Update** — kết hợp với hiểu biết cũ và cập nhật

### 4.4. Cụ thể trong paper: GraphSAGE với "sum" aggregator

$$
\mathbf{h}_v^l = \sigma\!\left(\mathbf{W}^l \left(\mathbf{h}_v^{l-1} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{l-1}\right)\right)
$$

Phân tích từng mảnh:

| Mảnh công thức | Ý nghĩa |
|---|---|
| $\sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{l-1}$ | cộng tất cả "hiểu biết" của hàng xóm |
| $\mathbf{h}_v^{l-1} + \sum$ | cộng thêm hiểu biết bản thân (đây là "self-loop" implicit) |
| $\mathbf{W}^l \cdot$ | ma trận trọng số (learnable) "trộn" thông tin |
| $\sigma = \tanh$ | kích hoạt phi tuyến để cho phép biểu diễn quan hệ phi tuyến |

#### Phân tích sâu hơn về aggregator: sum vs mean vs max

**Lý do paper chọn sum** (paper trích Xu et al. 2019 — "How powerful are GNNs?"): sum là aggregator *injective* nhất — nó phân biệt được nhiều cấu hình hàng xóm hơn mean/max.

| Aggregator | Phân biệt được gì? | Ví dụ trong stiffened panel |
|---|---|---|
| `sum` | Số lượng + giá trị hàng xóm | Plate có 4 stiffener vs 2 stiffener khác hẳn |
| `mean` | Chỉ giá trị trung bình hàng xóm | 4 stiffener giá trị X = 2 stiffener giá trị 2X — sai! |
| `max` | Chỉ "đại biểu mạnh nhất" | Mất hoàn toàn thông tin về phân bố hàng xóm |

**Ví dụ cụ thể về sự khác biệt:**

Xét 2 panel:

- Panel A: plate ở giữa giáp 2 stiffener cao 100mm.
- Panel B: plate ở giữa giáp 4 stiffener cao 50mm.

Vật lý: Panel A và Panel B có *tổng "khả năng đỡ"* khác nhau (vì độ cứng phụ thuộc cubic vào height).

- Với `sum`: aggregated = $2 \times h_{100}$ vs $4 \times h_{50}$ → khác nhau → mạng phân biệt được.
- Với `mean`: aggregated = $h_{100}$ vs $h_{50}$ — vẫn khác, nhưng KHÔNG có thông tin "có bao nhiêu stiffener".

Trường hợp tệ hơn:

- Panel A: 2 stiffener đều cao 100mm → mean = $h_{100}$.
- Panel B: 4 stiffener đều cao 100mm → mean = $h_{100}$ (giống panel A!).

→ `mean` không phân biệt được. `sum` thì $2 h_{100}$ vs $4 h_{100}$ — phân biệt rõ.

> **Đây là lý do bản chất tại sao paper chọn sum.** Trong cơ học, **số lượng kết nối** ảnh hưởng trực tiếp đến độ cứng — phải bảo toàn.

#### Vì sao $h_v + \sum h_u$ thay vì $\text{concat}(h_v, \sum h_u)$?

Cả hai đều khả thi. Concat giữ được thông tin tốt hơn (không bị "lẫn") nhưng tăng kích thước. Paper chọn cộng vì:

- Đơn giản, nhanh.
- Có hiệu ứng "residual connection" — ổn định gradient flow trong deep GNN (32 layer!).

#### Vì sao tanh thay vì ReLU?

Ít paper GNN dùng tanh, nhưng có lý do:

- Output là *stress* — giá trị thực, có thể âm hoặc dương.
- ReLU "kill" gradient ở vùng âm — không phù hợp với regression có range rộng.
- Tanh smoother → gradient ổn định hơn trong 32 layer sâu.

Trade-off: tanh saturate ở 2 đầu → cần normalize input cẩn thận.

### 4.5. Sau nhiều layer thì node biết được gì?

Paper dùng **32 layer**, mỗi layer **64 hidden neuron**. Sau **batch normalization** mỗi layer để ổn định.

Tiến trình "hiểu biết":

| Layer | Node biết gì |
|---|---|
| 0 | Chỉ thông tin bản thân (8 features) |
| 1 | Bản thân + hàng xóm cấp 1 |
| 2 | + hàng xóm cấp 2 |
| ... | ... |
| ~10 | Toàn graph (vì panel ~16-30 node) |
| ~32 | Toàn graph + nhiều layer biến đổi phi tuyến → biểu diễn phức tạp |

Sau layer cuối, mỗi node có một vector hiểu biết (64-dim) → đi qua một head MLP (64 → 200) → ra **200 giá trị stress** trên grid 10×20 của thành phần đó.

> **Cảm giác cuối cùng:** tưởng tượng mỗi node ban đầu là một người tù trong phòng kín, chỉ biết kích thước phòng mình. Sau 32 vòng thì thầm qua vách tường, nó hiểu được toàn bộ tòa nhà — và từ đó *suy ra được* áp lực mà tường phòng nó đang chịu.

### 4.6. Mối liên hệ ngầm với phương trình cơ học

Đây là phần triết lý thú vị nhất. Có thể bạn đã nhận ra:

- Phương trình cân bằng cơ học (PDE): $\nabla \cdot \sigma + b = 0$ — *local* (chỉ phụ thuộc lân cận).
- Lời giải FEM: ma trận $K \cdot u = f$ — sparse, mỗi node chỉ ghép với hàng xóm.
- GNN message passing: $h_v^l = f(h_v^{l-1}, \{h_u^{l-1}\}_{u \in \mathcal{N}(v)})$ — *local* cập nhật.

→ **Cấu trúc toán học của GNN khớp với cấu trúc toán học của PDE cơ học.** Đây không phải ngẫu nhiên — đây là lý do GNN học hiệu quả: nó *implicit* tôn trọng tính địa phương của vật lý.

Trong nghiên cứu sâu hơn, người ta gọi đây là **physics-informed inductive bias** — và là lý do GNN thường *outperform* MLP/CNN với cùng lượng dữ liệu trong các bài toán physics.

### 4.7. Bridge → Phần 5

Bạn đã hiểu:

1. Message passing = lan truyền tin đồn = lan truyền ứng suất.
2. 32 layer đủ sâu để mỗi node "thấy" toàn graph + biểu diễn phi tuyến phức tạp.
3. Sum aggregator giữ thông tin về số hàng xóm (= cường độ kết nối cơ học).
4. Cấu trúc GNN khớp tự nhiên với cấu trúc PDE cơ học.

→ Câu hỏi tiếp theo: **Hiểu cơ chế rồi — quy trình toàn cục từ raw data đến predict ra sao? Có những bước nào dễ sai?**

---

## 5. Pipeline đầy đủ — và những điểm dễ sai khi reproduce

Hình dung như một dây chuyền:

```
ABAQUS FEM ───▶ Graph Builder ───▶ GraphSAGE ───▶ Stress Map
  (sinh data)    (hình → graph)    (huấn luyện)    (predict)
```

Tài liệu này không chỉ liệt kê các bước — mà còn phân tích **vai trò vật lý**, **lý do thiết kế**, và **chỗ dễ sai** của mỗi bước.

### 5.1. Bước 1 — Sinh dữ liệu (FEM/ABAQUS)

#### 5.1.1. Cài đặt cơ bản

- Tạo **2.000 đến 8.000** thiết kế ngẫu nhiên (random sampling trong miền tham số: độ dày plate 10–20 mm, số stiffener 2–8, …).
- Mỗi thiết kế chạy ABAQUS → ra trường ứng suất von Mises trên toàn panel.
- Mesh dày: 15 elements giữa các stiffener, 15 cho web height, 8 cho flange width.
- Phần tử shell `S4R`.

#### 5.1.2. Vai trò

Đây là "ground truth" — không có nó, mạng không có gì để học. Nếu thiếu bước này, GNN không biết "ứng suất đúng" trông như thế nào.

> **Đây cũng là bước tốn nhiều thời gian nhất khi reproduce — chiếm ~80% effort.** Paper viết rất ngắn gọn về phần này (chỉ 1 dòng "parametric modeling"), nhưng thực tế đây là chỗ bạn sẽ debug nhiều nhất.

#### 5.1.3. "Random" cụ thể là gì?

Paper dùng **uniform random sampling** đơn giản — không phải Latin Hypercube, không phải Sobol. Mỗi tham số được bốc độc lập trong miền cho phép (Bảng 2 của paper):

```python
import numpy as np

def sample_one_design(seed):
    """Bốc một thiết kế ngẫu nhiên, deterministic theo seed."""
    rng = np.random.default_rng(seed)
    return {
        # Tham số liên tục — uniform
        'plate_thickness':   rng.uniform(10, 20),    # mm
        'stiff_thickness':   rng.uniform(5, 20),
        'stiff_height':      rng.uniform(100, 400),
        'flange_thickness':  rng.uniform(5, 20),
        'flange_width':      rng.uniform(50, 150),
        'pressure':          rng.uniform(0.05, 0.1), # MPa

        # Tham số rời rạc — random integer
        'n_stiffeners':      int(rng.integers(2, 9)),    # 2..8

        # Tham số categorical — random choice (chỉ với case 1, 5)
        'bc_plate_edges':    rng.choice(['fixed', 'simply'], size=4).tolist(),
        'bc_stiff_edges':    rng.choice(['free', 'fixed', 'simply'], size=4).tolist(),
    }
```

→ Lặp 2000 lần với 2000 seed khác nhau là có 2000 thiết kế. **Quan trọng:** dùng `seed=i` để reproduce được — nếu không, không thể debug khi có sample fail.

#### 5.1.4. Phân tích sâu: Sampling strategy có quan trọng không?

Uniform sampling là choice đơn giản nhưng có vấn đề:

- **Curse of dimensionality**: Nếu có $d$ tham số, miền tham số $d$-chiều. Uniform sampling sẽ thưa khi $d$ lớn.
- **Vùng quan trọng vật lý** (vd gần buckling threshold) có thể bị under-sample.
- **Combinations rời rạc** (categorical BC × N) có thể không phủ đều.

**Cách cải tiến (cho thesis của bạn):**

| Strategy | Ưu | Nhược |
|---|---|---|
| Uniform random (paper dùng) | Đơn giản, không bias | Phủ thưa khi $d$ lớn |
| Latin Hypercube Sampling | Phủ đều miền tốt hơn | Cần biết trước số sample |
| Sobol sequence (quasi-random) | Phủ đều, low-discrepancy | Implementation phức tạp |
| Adaptive sampling | Tập trung vùng quan trọng | Cần model preliminary |
| Active learning | Tối ưu data | Cần loop train-sample-train |

#### 5.1.5. Có tự động được không? — 3 mức độ

##### Mức 1 — Bán-thủ-công (KHÔNG nên làm)

Mở ABAQUS CAE bằng tay, mỗi lần đổi tham số rồi nhấn Submit → 2000 lần. **Tuyệt đối không khả thi**, mỗi thiết kế mất 10–30 phút thao tác = 1 năm chỉ để click chuột.

##### Mức 2 — ABAQUS Python scripting (chuẩn industry)

ABAQUS có **Python interpreter built-in** (Python 2.7 hoặc 3.x tùy version). Mọi thứ làm được trên GUI đều làm được bằng code — kể cả tạo geometry, mesh, BC, submit job, extract result.

Chạy bằng:

```bash
abaqus cae noGUI=script.py        # chạy script trong CAE backend
abaqus python extract_stress.py   # chạy script Python thuần (cho post-processing)
```

##### Mức 3 — Hybrid orchestrator (recommended cho paper này)

Một script Python "bên ngoài" (regular Python 3) làm orchestrator, gọi ABAQUS Python qua `subprocess`. Cho phép:

- **Parallel execution** (nhiều job chạy đồng thời)
- **Recovery from crash** (resume từ chỗ fail)
- **Better debugging** (Python 3 features, modern libraries)
- **Resource management** (timeout, memory limits)

→ **Paper chắc chắn dùng Mức 2 hoặc Mức 3.** Họ chạy trên HPC cluster (UBC ARC) với thousand-sample datasets — không có cách nào làm bằng tay.

#### 5.1.6. Workflow Mức 3 — Cấu trúc 3 file

```
sample_designs.py     # Master orchestrator (Python 3 thường)
build_model.py        # Chạy bên trong ABAQUS — xây model + submit
extract_stress.py     # Chạy bên trong ABAQUS — đọc .odb
```

##### File 1: Master orchestrator

```python
# sample_designs.py — chạy bằng Python 3 thường
import numpy as np, subprocess, json
from pathlib import Path

OUT = Path('./dataset'); OUT.mkdir(exist_ok=True)
N_SAMPLES = 2000

for i in range(N_SAMPLES):
    out_npz = OUT / f"design_{i:04d}_stress.npz"
    if out_npz.exists():
        continue  # đã chạy → bỏ qua (resume sau crash)

    # 1) Bốc tham số ngẫu nhiên (deterministic theo seed=i)
    params = sample_one_design(seed=i)
    params_file = OUT / f"design_{i:04d}_params.json"
    params_file.write_text(json.dumps(params))

    # 2) Gọi ABAQUS để xây model + submit job
    try:
        subprocess.run([
            'abaqus', 'cae', 'noGUI=build_model.py',
            '--', f'--id={i}', f'--params={params_file}'
        ], check=True, timeout=1800)  # 30 phút timeout
    except subprocess.TimeoutExpired:
        print(f"[{i}] timeout, skip"); continue
    except subprocess.CalledProcessError as e:
        print(f"[{i}] FEM failed: {e}"); continue

    # 3) Trích stress từ .odb
    subprocess.run([
        'abaqus', 'python', 'extract_stress.py',
        f'--odb={OUT}/design_{i:04d}.odb',
        f'--out={out_npz}',
        f'--params={params_file}'
    ], check=True)

    # 4) Dọn file lớn để tiết kiệm dung lượng (.odb 100MB+, không cần giữ)
    for ext in ['.odb', '.dat', '.msg', '.com', '.sta', '.prt']:
        f = OUT / f"design_{i:04d}{ext}"
        if f.exists(): f.unlink()
```

##### File 2: Build model (chạy trong ABAQUS)

```python
# build_model.py — chạy bằng `abaqus cae noGUI=build_model.py`
from abaqus import *
from abaqusConstants import *
import json, sys

# Parse args sau dấu '--'
args = sys.argv[sys.argv.index('--')+1:]
params_file = next(a.split('=')[1] for a in args if a.startswith('--params='))
design_id   = next(a.split('=')[1] for a in args if a.startswith('--id='))
params = json.load(open(params_file))

m = mdb.Model(name='Panel')

# 1) Plate (3m × 3m)
sk = m.ConstrainedSketch(name='plate', sheetSize=5000)
sk.rectangle(point1=(0,0), point2=(3000, 3000))
plate = m.Part(name='Plate', dimensionality=THREE_D, type=DEFORMABLE_BODY)
plate.BaseShell(sketch=sk)

# 2) Stiffeners — loop tạo web + flange
n = int(params['n_stiffeners'])
spacing = 3000.0 / (n + 1)
for i in range(n):
    x_pos = spacing * (i + 1)
    # Web: extrude shell từ đường thẳng (x_pos, 0) → (x_pos, 3000)
    # cao params['stiff_height']
    # ... (chi tiết ABAQUS API)

    # Flange (nếu có)
    # ... đặt phẳng tại đỉnh web, rộng params['flange_width']

# 3) Material + Section assignment
m.Material(name='Steel')
m.materials['Steel'].Elastic(table=((200000., 0.3),))
# Plate section:  thickness = params['plate_thickness']
m.HomogeneousShellSection(name='PlateSec', material='Steel',
                          thickness=params['plate_thickness'])
# Web section, Flange section tương tự...

# 4) Mesh — 15 elements giữa stiffeners → seed size = spacing/15
plate.seedPart(size=spacing/15.0)
plate.generateMesh()
# Mesh stiffeners tương tự với appropriate seed size

# 5) Assembly + BC + Load + Step
a = m.rootAssembly
inst = a.Instance(name='Panel-1', part=plate, dependent=ON)

# IMPORTANT: đặt tên các Set/Surface NGAY khi tạo, không tạo sau
# vì extract_stress.py sẽ truy cập theo tên
for edge_idx, bc_type in enumerate(params['bc_plate_edges']):
    edge_set = a.Set(name=f'PlateEdge_{edge_idx}', edges=...)
    if bc_type == 'fixed':
        m.EncastreBC(name=f'BC_plate_{edge_idx}',
                     createStepName='Initial', region=edge_set)
    elif bc_type == 'simply':
        m.PinnedBC(name=f'BC_plate_{edge_idx}',
                   createStepName='Initial', region=edge_set)

# Step + Load
m.StaticStep(name='Load', previous='Initial', nlgeom=OFF)
m.Pressure(name='P', createStepName='Load',
           region=top_surface, magnitude=params['pressure'])

# 6) Đặt tên Set cho từng structural unit (phục vụ extract sau)
# QUAN TRỌNG: tên phải nhất quán với graph builder
for i, ps in enumerate(plate_spans):
    a.Set(name=f'plate_span_{i}', elements=ps.elements)
for i in range(n):
    a.Set(name=f'web_{i}',    elements=...)
    a.Set(name=f'flange_{i}', elements=...)

# 7) Submit job
job_name = f'design_{int(design_id):04d}'
job = mdb.Job(name=job_name, model='Panel', numCpus=4, memory=8000)
job.submit()
job.waitForCompletion()
```

##### File 3: Extract stress

```python
# extract_stress.py — chạy bằng `abaqus python extract_stress.py`
from odbAccess import openOdb
import numpy as np, json, sys

odb_path = next(a.split('=')[1] for a in sys.argv if a.startswith('--odb='))
out_path = next(a.split('=')[1] for a in sys.argv if a.startswith('--out='))

odb = openOdb(odb_path)
step = odb.steps['Load']
frame = step.frames[-1]  # last frame của analysis
stress_field = frame.fieldOutputs['S']  # stress tensor

# Lấy von Mises trên từng element set (theo tên đã đặt ở build_model.py)
data = {}
for elset_name in odb.rootAssembly.elementSets.keys():
    if not elset_name.startswith(('plate_span_', 'web_', 'flange_')):
        continue
    elset = odb.rootAssembly.elementSets[elset_name]
    sub = stress_field.getSubset(region=elset, position=ELEMENT_NODAL)

    # Chú ý surface: paper quy định
    #   plate, flange  → lower z-surface (loaded side)
    #   web           → higher y-surface hoặc lower x-surface
    # Phải filter section point cho đúng surface!

    von_mises = np.array([v.mises for v in sub.values])
    coords    = np.array([(v.nodeLabel,) + tuple(v.instance.nodes[v.nodeLabel-1].coordinates)
                          for v in sub.values])
    data[elset_name + '_mises']  = von_mises
    data[elset_name + '_coords'] = coords

np.savez(out_path, **data)
odb.close()
```

> **Bước interpolation về grid 10×20 nên làm sau** (trong main pipeline pre-processing), KHÔNG làm trong extract_stress — vì interpolation dùng scipy nhưng ABAQUS python (2.7) có thể không có scipy mới.

#### 5.1.7. Phân tích thời gian thực tế

##### Thời gian mỗi sample

| Bước | Thời gian |
|---|---|
| Build model trong ABAQUS | 5–30 giây |
| Submit + run analysis (mesh dày) | **3–15 phút** ← bottleneck |
| Extract stress từ .odb | 5–10 giây |
| **Tổng/sample** | **~5–20 phút** |

##### Toàn dataset

- 2000 samples × 10 phút trung bình = **~14 ngày sequential**.
- 8000 samples (Case 5) = **~55 ngày sequential**.
- → Phải **parallel** để không tốn cả tháng.

##### Cách parallel:

| Cách | Tốc độ | Lưu ý |
|---|---|---|
| 1 máy, 4 CPU/job, 1 job/lúc | 1× | Baseline |
| 1 máy, 4 jobs đồng thời (1 CPU/job) | ~3× | Cần check ABAQUS license token đủ |
| Cluster HPC (paper dùng UBC ARC) | 50–200× | **Cần** cho 8000 samples |

> Paper acknowledge rõ: *"computational resources and services provided by Advanced Research Computing at The University of British Columbia"*. Họ chạy trên HPC cluster, không phải PC cá nhân.

#### 5.1.8. Mẹo quan trọng — Macro Recorder

Đừng học ABAQUS Python API từ đầu — dùng **Macro Recorder** trong ABAQUS CAE:

1. Mở `File → Macro Manager → Create...`
2. Đặt tên macro, chọn level (Session/User/Work).
3. ABAQUS bắt đầu record mọi thao tác.
4. Bạn build 1 panel hoàn chỉnh trên GUI (geometry → mesh → BC → load → submit).
5. Stop record → ABAQUS xuất ra script Python tương ứng.
6. Bạn dùng script đó làm template, parameterize bằng `params` dict.

→ Tiết kiệm **2-3 tuần** học API. Đây là shortcut industry-standard mà paper không nói.

#### 5.1.9. Pitfalls điển hình khi reproduce

| Lỗi | Hậu quả | Cách phòng |
|---|---|---|
| ABAQUS license token cạn (chạy parallel quá nhiều) | Job fail random | Throttle: max 4 job đồng thời |
| Một thiết kế "lạ" (gân quá nhỏ) khiến mesh fail | Crash cả batch | `try/except` quanh mỗi sample, log fail |
| `.odb` file 100MB × 8000 = 800GB | Hết ổ cứng | Xóa `.odb` ngay sau khi extract |
| ABAQUS tự đổi tên element set → script extract sai | Stress lấy sai unit | Đặt tên *trước* khi mesh, dùng `Set` API |
| Memory leak khi loop trong ABAQUS Python | Crash sau 200 sample | Restart ABAQUS sau mỗi 100 sample |
| Quên fix random seed | Không reproduce được | `seed=i` cho từng sample |
| BC không apply đúng cạnh | Toàn bộ dataset sai | Sanity check 1 sample trước khi run 2000 |
| Sample không đủ phủ miền | Model fail ở thiết kế ngoài training domain | Visualize distribution các tham số |
| Mesh size không nhất quán giữa sample | Ground truth khác nhau cho cùng 1 thiết kế | Cố định seed size trong build_model |
| Quên xuất stress tại đúng surface (lower-z cho plate) | Stress sai do dùng surface đối diện | Verify với 1 sample bằng GUI |
| Không kiểm tra hội tụ FEM | Ground truth có lỗi mà không biết | Chạy mesh sensitivity 1 lần (Appendix B paper) |

#### 5.1.10. Khả thi không cho bạn? — Câu trả lời thẳng

**CÓ thể tự động hoàn toàn — với điều kiện:**

1. **Có ABAQUS license đầy đủ** (Student edition giới hạn 1000 nodes — KHÔNG đủ cho mesh 15 element/span).
2. **Có máy đủ mạnh** — tối thiểu 16 GB RAM, 8 cores. Lý tưởng: HPC cluster.
3. **Dành 2–3 tuần đầu chỉ để viết + debug pipeline data generation**.
4. **Sanity check kỹ trước khi run batch** — 1 sample sai pattern = 2000 sample sai.

**Khó / không khả thi nếu:**

- Chỉ có ABAQUS Student → mesh không đủ dày.
- Không có cluster → 2000 sample mất vài tuần.
- Geometry phức tạp (Case 6: curved + bi-directional) → graph builder rất khó tự động hóa hoàn toàn cho mọi cấu hình.

#### 5.1.11. Lộ trình thực tế cho bạn (5 tuần)

| Tuần | Công việc | Output |
|---|---|---|
| 1 | Build 1 panel cố định trong ABAQUS GUI bằng tay; ghi lại toàn bộ thao tác | Hiểu workflow, có file `.cae` mẫu |
| 2 | Bật Macro Recorder, build lại 1 panel → có script template | `template.py` |
| 3 | Parameterize `template.py` bằng `params` dict; viết `sample_one_design()` | `build_model.py` |
| 4 | Viết orchestrator + extract_stress; chạy 10 sample test; verify với ABAQUS GUI | Pipeline working |
| 5 | Scale 100 → 500 → 2000 sample; monitor, fix sample fail | Dataset đầy đủ |

> **Mẹo cuối:** *Bắt đầu với Case 1* (BC vary, hình học đều, vật liệu tuyến tính). Đừng cố làm Case 5 (tổng hợp tất cả biến) ngay — sẽ debug điên đảo. Sau khi Case 1 chạy ổn, mở rộng dần.

### 5.2. Bước 2 — Chuyển kết cấu thành graph

#### Cài đặt

- Phát hiện các structural unit (plate-span, web, flange) bằng cách phân tích hình học.
- Xây ma trận kề: 2 unit chia sẻ một cạnh hàn → có edge.

#### Vai trò

Đây là "ngôn ngữ" để GNN hiểu kết cấu. Nếu sai bước này (vd thiếu edge), thông tin không truyền đúng → ứng suất predict sai chỗ.

#### Phân tích sâu: Bước này khó hơn vẻ ngoài

Trong panel đơn giản (uniform stiffener), graph builder dễ — chỉ cần đếm. Nhưng trong **case 5 và 6** (geometric complexity, curved panel với bi-directional stiffener), graph builder phải:

- Phát hiện chỗ giao nhau giữa longitudinal và transverse stiffener.
- Phân biệt edge "plate–web" với edge "web–web" (intersection).
- Xử lý curved geometry — cạnh không thẳng.

**Đây là lý do code không public dễ chạy.** Phần graph builder rất phụ thuộc vào convention của bạn.

#### Pitfall điển hình

| Lỗi | Hậu quả |
|---|---|
| Quên edge giữa plate-edge và stiffener-end | Stress concentration ở góc không được học |
| Edge không đối xứng (forgot undirected) | Message passing chỉ chạy 1 chiều → sai |
| Graph builder giả định T-stiffener khi panel có L-stiffener | Crash hoặc predict sai |

### 5.3. Bước 3 — Xây feature cho node

#### Cài đặt

- Với mỗi unit, tính **8 feature**: width, length, thickness, BC×4, pressure.
- Thêm tọa độ patch loading nếu áp dụng.
- Chuẩn hóa (normalize) các feature về cùng thang đo.

#### Vai trò

Đây là "input" cho mạng. Nếu không có feature về BC, mạng sẽ predict sai khi đổi điều kiện biên.

#### Phân tích sâu: Normalization quan trọng đến đâu?

Cực quan trọng. Lý do:

- `width`, `length` (mm) có range ~100–3000.
- `thickness` (mm) có range ~5–20.
- `pressure` (MPa) có range ~0.05–0.3.

Nếu KHÔNG normalize, gradient của feature lớn (length) sẽ dominate → mạng phớt lờ feature nhỏ (pressure). Mà pressure là *driver chính* của stress!

**Cách normalize phổ biến:**

- *Min-Max scaling*: $x' = (x - x_{min}) / (x_{max} - x_{min})$ → range [0, 1].
- *Z-score*: $x' = (x - \mu) / \sigma$ → mean 0, std 1.

Paper không nói rõ dùng cái nào, nhưng với regression, **Min-Max thường ổn định hơn** vì giữ được range.

#### Pitfall điển hình

| Lỗi | Hậu quả |
|---|---|
| Quên normalize | Loss không converge, hoặc converge về local minimum tệ |
| Normalize trên tập train+test (data leakage) | Test accuracy cao giả tạo |
| BC encode bằng số liên tục thay vì categorical | Mạng học sai (vì free=0, simply=1, fixed=2 không có nghĩa "fixed cao gấp đôi simply") |

### 5.4. Bước 4 — Xây label (target stress)

#### Cài đặt

- Lấy ứng suất ABAQUS trên unit đó.
- **Sample về grid 10×20** bằng *scattered interpolation* (vì các unit có hình dạng/kích thước khác nhau, phải đưa về cùng "form factor").
- Flatten thành vector 200 chiều.

#### Vai trò

Đây là "đáp án". Cách sampling 10×20 là điểm tinh tế — vừa đủ chi tiết, vừa không tốn memory. Phụ lục B của paper chứng minh đây là choice tối ưu.

#### Phân tích sâu: Information loss trong interpolation

Đây là một điểm yếu *hidden* của paper. Khi interpolate stress field từ FEM mesh (50.000 điểm) xuống grid 10×20 (200 điểm), thông tin bị mất:

- Stress concentration nhọn (vd ở góc) có thể bị "smooth out".
- Maximum stress thực có thể nằm giữa 2 grid point → không được capture.

Paper *cố ý* chọn MSE loss để "phạt mạnh giá trị lớn" — bù trừ cho mất mát này. Nhưng đây vẫn là một limitation.

#### Pitfall điển hình

| Lỗi | Hậu quả |
|---|---|
| Interpolation method sai (linear vs cubic) | Smoothing quá hoặc thiếu |
| Quên align grid orientation với local axis của unit | Stress map xoay sai → predict sai |
| Lấy stress trên surface sai (web có 2 surface, x-axis vs y-axis) | Hệ thống sai mà không phát hiện |

### 5.5. Bước 5 — Huấn luyện GNN

#### Hyperparameters đầy đủ

| Thông số | Giá trị | Lý do/Comment |
|---|---|---|
| Framework | PyTorch Geometric | Standard cho GNN research |
| Mạng | GraphSAGE | Spatial, sum-friendly |
| Số layer | 32 | Deep enough for receptive field |
| Hidden size | 64 | Balance capacity/overfit |
| Activation | tanh | Smooth, fit regression |
| Optimizer | Adam | Standard |
| Learning rate | 0.02 | Hơi cao — có thể vì batch size lớn |
| Batch size | 512 | Tận dụng GPU |
| L2 regularization | 1e-4 | Standard |
| Loss | MSE | Phạt mạnh max stress |
| Aggregator | sum | Bảo toàn thông tin số hàng xóm |
| Hardware | NVIDIA GTX 3090 | 24GB VRAM |
| Train/val/test | 80% / 10% / 10% | Standard |
| Training time | 30–120 phút mỗi case | Scale theo data size |

#### Phân tích sâu: Tại sao MSE thay vì MAE/Huber?

| Loss | Ưu | Nhược |
|---|---|---|
| **MSE** ✅ | Phạt mạnh giá trị lớn → model ưu tiên đúng max stress | Nhạy với outlier; predict tệ ở vùng stress thấp |
| MAE | Robust với outlier | Phạt yếu giá trị lớn → có thể bỏ qua điểm critical |
| Huber | Cân bằng MSE+MAE | Thêm hyperparameter |

→ Paper chọn MSE *cố ý* vì mục tiêu là dự đoán đúng max stress (chỗ failure). Nhưng đây cũng là lý do ở vùng stress thấp model không chính xác bằng vùng cao (paper thừa nhận).

> **Bài học:** Choice of loss function reflects engineering priority. Chọn loss = chọn what error matters most.

#### Phân tích về batch size 512 và lr 0.02

Hai hyperparameter này **cùng thay đổi theo nguyên tắc Linear Scaling Rule**: batch lớn → lr lớn (cùng tỉ lệ). Paper đi theo rule này.

Khi reproduce với GPU nhỏ hơn:

- Nếu giảm batch xuống 64, nên giảm lr theo: $0.02 \times 64/512 = 0.0025$.
- Hoặc dùng *gradient accumulation*: tích lũy gradient qua 8 mini-batch trước khi step.

#### Pitfall điển hình

| Lỗi | Hậu quả |
|---|---|
| Train không đủ epoch | Predict tệ ở vùng có stress concentration |
| Train quá lâu | Overfit (đặc biệt với case 3 — geometric complexity) |
| Quên BatchNorm sau mỗi SAGE layer | Gradient explode/vanish trong 32 layer |
| Lr cao quá | Loss dao động không giảm |

### 5.6. Bước 6 — Dự đoán

#### Cài đặt

- Với một thiết kế mới: build graph → forward pass → ra 200 stress values cho mỗi node.
- Ráp các unit lại → contour stress 3D toàn panel.
- **Thời gian: gần như instantaneous** (so với phút/giờ của ABAQUS).

→ Đây chính là "ROM" — sau khi đầu tư thời gian train, mỗi prediction chỉ tốn ms.

#### Phân tích về tốc độ thực tế

| Phương pháp | Thời gian/thiết kế |
|---|---|
| ABAQUS FEM | 5–30 phút |
| GNN inference | ~10 ms |
| **Speedup** | **~30.000–180.000 lần** |

→ Trong vòng tối ưu hóa 10.000 thiết kế: ABAQUS ~1 năm, GNN ~2 phút.

### 5.7. Tổng quan các case study trong paper

| Case | Mô tả | Accuracy max stress |
|---|---|---|
| 1 | Vary boundary conditions, panel thẳng, stiffener đều | 94.93% |
| 2 | Material phi tuyến (đàn-dẻo) | 96.31% |
| 3 | Hình học phức tạp (stiffener không đồng đều) | 88.01% |
| 4 | Patch loading (tải tập trung) | — |
| 5 | Tổng hợp tất cả biến | 89.69% |
| 6 | Panel cong + stiffener 2 chiều | 93.93% (test 12) |
| **Trung bình** | toàn bộ test set | **92.3%** |

#### Phân tích thú vị: Tại sao Case 3 (88%) tệ hơn Case 1 (95%)?

Case 3 có **stiffener không đồng đều** — chiều cao, độ dày khác nhau giữa các stiffener. Lý do model làm tệ hơn:

- Không gian thiết kế **lớn hơn nhiều** lần Case 1 → cần thêm data nhưng paper vẫn dùng 2000 sample.
- Stress field có "jump" mạnh giữa các stiffener với kích thước khác nhau → khó interpolate.
- Topology graph vẫn giống nhau (mạng không "thấy" được sự khác biệt qua structure, chỉ qua features).

→ Bài học: GNN learning capacity bị giới hạn khi geometric variation lớn mà không đi kèm với topology variation.

### 5.8. Bridge → Phần 6

Bạn đã hiểu:

1. Pipeline 6 bước: data gen → graph → features → labels → train → predict.
2. Mỗi bước có vai trò vật lý + lý do thiết kế cụ thể.
3. Có những pitfall điển hình ở mỗi bước.

→ Câu hỏi tiếp theo: **Hiểu rồi — bây giờ tôi tự code lại như thế nào? Cần chuẩn bị gì? Pseudo-code ra sao?**

---

## 6. Hướng dẫn tự reproduce — kèm phân tích pitfalls

### 6.1. Chuẩn bị

#### Cần có

| Mục | Yêu cầu |
|---|---|
| FEM software | ABAQUS (trả phí) hoặc CalculiX/Code_Aster (free) |
| Python | ≥ 3.9 với PyTorch ≥ 1.12 |
| **PyTorch Geometric (PyG)** | thư viện GNN chính |
| GPU | tối thiểu 4 GB cho unit-vertex; 24 GB+ nếu thử FE-vertex |
| Skill cơ học | hiểu được FEM output, biết xuất stress field từ ABAQUS |

#### Dữ liệu cần

- Tối thiểu **1.000 design samples** (paper test với 1.000 – 8.000).
- Mỗi sample: hình học + BC + load + stress field từ FEM.

#### Lộ trình thực tế (8 tuần)

| Tuần | Công việc | Khó khăn dự kiến |
|---|---|---|
| 1 – 2 | Viết script Python parametric tạo input ABAQUS (file `.inp`), chạy batch | Học ABAQUS scripting |
| 3 | Viết parser đọc kết quả ABAQUS (file `.odb` hoặc `.rpt`) → trích stress | ABAQUS Python API tricky |
| 4 | Xây graph builder từ hình học | Phân loại unit + edge logic |
| 5 – 6 | Code GraphSAGE, training loop, validation | Debug NaN loss, OOM |
| 7+ | Experiment với từng case | So sánh với paper baseline |

> **Cảnh báo:** Paper KHÔNG public code (`Data will be made available on request`). Bạn sẽ phải tự viết phần graph builder và data pipeline. Phần GraphSAGE thì PyG đã có sẵn `SAGEConv`.

### 6.2. Phân tích "phần dễ / phần khó" cho người reproduce

| Phần | Độ khó | Lý do |
|---|---|---|
| Mạng GraphSAGE (PyTorch) | ⭐ Dễ | PyG có `SAGEConv` sẵn |
| Training loop | ⭐ Dễ | Standard PyTorch |
| Data normalization | ⭐⭐ Trung bình | Cần thử nghiệm scaling |
| **Graph builder** | ⭐⭐⭐⭐ Khó | Logic phụ thuộc geometry, không có chuẩn |
| **ABAQUS parametric scripting** | ⭐⭐⭐⭐ Khó | API khó, debug khổ |
| **Stress extraction từ .odb** | ⭐⭐⭐⭐⭐ Rất khó | Phải hiểu element type, surface, local axis |

→ **80% effort sẽ rơi vào FEM data pipeline, không phải GNN.** Đây là sự thật mà paper không nói rõ.

### 6.3. Pseudo-code

#### A. Tạo graph từ dữ liệu cơ học

```python
def build_graph(panel_geometry, loading):
    """
    panel_geometry: dict với plate dims, list of stiffeners
                    (mỗi stiffener: web height/thickness, flange w/t, vị trí)
    loading: pressure value, [optional] patch position
    """
    nodes = []          # list of feature vectors
    node_id_map = {}    # tên unit -> node index
    edges = []          # list of (i, j) pairs

    # 1) Tạo node cho mỗi plate span (giữa các stiffener)
    plate_spans = split_plate_by_stiffeners(panel_geometry)
    for ps in plate_spans:
        feat = [ps.width, ps.length, ps.thickness,
                ps.bc_top, ps.bc_bot, ps.bc_left, ps.bc_right,
                loading.pressure]
        node_id_map[ps.name] = len(nodes)
        nodes.append(feat)

    # 2) Tạo node cho mỗi web và flange của stiffener
    for st in panel_geometry.stiffeners:
        # web
        feat_web = [st.web_thickness, st.web_height, st.length,
                    *st.bc_web_edges_4_values, loading.pressure]
        node_id_map[st.web_name] = len(nodes)
        nodes.append(feat_web)
        # flange (nếu có)
        if st.has_flange:
            feat_fl = [st.flange_width, st.flange_thickness, st.length,
                       *st.bc_flange_edges_4_values, loading.pressure]
            node_id_map[st.flange_name] = len(nodes)
            nodes.append(feat_fl)

    # 3) Tạo edge theo mối hàn vật lý
    for ps in plate_spans:
        for st in adjacent_stiffeners(ps):
            edges.append((node_id_map[ps.name], node_id_map[st.web_name]))
    for st in panel_geometry.stiffeners:
        if st.has_flange:
            edges.append((node_id_map[st.web_name], node_id_map[st.flange_name]))

    # 4) Trả về dạng PyG Data
    import torch
    from torch_geometric.data import Data
    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # đối xứng cho undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(x=x, edge_index=edge_index)
```

#### B. Mô hình GraphSAGE

```python
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class StressGNN(nn.Module):
    def __init__(self, in_dim=8, hidden=64, n_layers=32, out_dim=200):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        # First layer: in_dim -> hidden
        self.layers.append(SAGEConv(in_dim, hidden, aggr='sum'))
        self.bns.append(nn.BatchNorm1d(hidden))
        # Remaining layers: hidden -> hidden
        for _ in range(n_layers - 1):
            self.layers.append(SAGEConv(hidden, hidden, aggr='sum'))
            self.bns.append(nn.BatchNorm1d(hidden))
        # Output head: predict 10x20 = 200 stress values per node
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.layers, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.tanh(x)
        return self.head(x)  # shape: (N_nodes, 200)
```

#### C. Training loop

```python
from torch_geometric.loader import DataLoader

def train(model, train_data, val_data, epochs=500, device='cuda'):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=512)

    for epoch in range(epochs):
        # ----- TRAIN -----
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch.x, batch.edge_index)   # (N, 200)
            loss = loss_fn(pred, batch.y)             # batch.y = stress labels
            loss.backward()
            opt.step()
            train_loss += loss.item() * batch.num_graphs

        # ----- VALIDATE -----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index)
                val_loss += loss_fn(pred, batch.y).item() * batch.num_graphs

        print(f"Epoch {epoch}: train={train_loss:.3f} val={val_loss:.3f}")
```

#### D. Inference (suy luận)

```python
def predict_stress(model, panel_geometry, loading, device='cuda'):
    model.eval()
    graph = build_graph(panel_geometry, loading).to(device)
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)   # (N_nodes, 200)
    # Reshape mỗi node thành lưới 10x20
    stress_grids = out.view(-1, 10, 20).cpu().numpy()
    return stress_grids
```

### 6.4. Debugging strategy — khi mọi thứ không hoạt động

| Triệu chứng | Nguyên nhân thường gặp | Cách kiểm tra |
|---|---|---|
| Loss = NaN sau vài epoch | LR quá cao; gradient explosion | Giảm LR; thêm gradient clipping |
| Loss giảm nhưng val accuracy thấp | Overfitting; data leakage | Tăng L2; kiểm tra train/test split |
| Predict ra const stress everywhere | Over-smoothing trong deep GNN | Giảm số layer; thêm residual; tăng diversity loss |
| Predict tốt train, tệ test | Distribution shift | Augment data; check normalization consistency |
| OOM | Batch quá lớn; graph quá lớn | Giảm batch; gradient accumulation |
| Output range sai (vd negative stress) | Activation cuối tanh + không scale | Scale output về MPa range thật |

### 6.5. Validation approach — làm sao biết bạn reproduce đúng?

Ngay cả khi bạn code đúng, vẫn cần verify. Recommended:

1. **Test case sanity**: tạo 1 panel cực đơn giản (1 stiffener, BC fixed all, uniform load) → ABAQUS tính ra stress, so với GNN predict. Sai lệch nên < 5%.

2. **Compare với baseline đơn giản**: so MAE của GNN vs MLP với input fixed-size. GNN nên *thua* MLP trên panel cố định (vì MLP có thể overfit) nhưng *thắng* khi panel thay đổi.

3. **Convergence study**: tăng dần data size từ 100 → 500 → 1000 → 2000. Accuracy nên cải thiện monotonically. Nếu không → có bug trong pipeline.

4. **Reproduce một số hyperparameter của paper**: lr=0.02, batch=512, 32 layer. Nếu kết quả khác xa paper → có bug trong implementation.

### 6.6. Checklist trước khi chạy

- [ ] Đã chạy đủ ABAQUS để có ≥ 1000 sample
- [ ] Đã viết parser .odb → trích stress đúng surface (lower-z cho plate/flange, higher-y/lower-x cho web)
- [ ] Đã interpolate stress về grid 10×20 cho mỗi unit
- [ ] Đã build graph với edge đúng theo mối hàn vật lý
- [ ] Đã normalize features (vd Min-Max scaling) nhất quán train/val/test
- [ ] Đã chia 80/10/10 đúng quy trình (split theo design, KHÔNG split theo node!)
- [ ] Đã chuẩn bị GPU đủ memory cho batch=512
- [ ] Đã có sanity check trên 1 panel đơn giản

### 6.7. Bridge → Phần 7

Bạn đã có đủ kiến thức và công cụ để reproduce. Nhưng đọc xong KHÔNG có nghĩa hiểu sâu — phải tự kiểm tra.

→ Phần 7 sẽ test bạn bằng 7 câu hỏi (5 cơ bản + 2 nâng cao). Trả lời mà không cần nhìn lại tài liệu này.

---

## 7. Kiểm tra hiểu sâu — 7 câu hỏi (5 cơ bản + 2 nâng cao)

Trả lời mà không nhìn lại các phần trên. Mục tiêu là *tự suy luận lại*, không phải *nhớ lại*.

### Q1. Trực giác về biểu diễn graph

Tại sao paper chọn cách "1 structural unit = 1 node" thay vì "1 finite element = 1 node"? Hãy giải thích cả về mặt **độ phức tạp tính toán** lẫn **trực giác vật lý**.

**Mở rộng:** Nếu mai bạn phải áp dụng GNN cho bài toán *biểu đồ ứng suất trong cánh máy bay* (wing với spar và rib), bạn sẽ định nghĩa "unit" là gì?

### Q2. Hiểu message passing

Mạng có **32 layer**. Nếu bạn giảm xuống **2 layer**, dự đoán sẽ kém ở chỗ nào? Tại sao?

**Mở rộng:** Cho ví dụ cụ thể về một loại stiffened panel mà việc thiếu layer sẽ gây sai lầm nghiêm trọng (gợi ý: nghĩ về panel dài, có nhiều stiffener, tải đặt 1 đầu).

### Q3. Hiểu lựa chọn aggregator

Paper chọn aggregator là **`sum`** thay vì `mean`. Hãy nghĩ ra một tình huống cụ thể trong stiffened panel mà nếu dùng `mean` sẽ cho kết quả sai.

**Hint:** nghĩ về plate có *nhiều* stiffener vs *ít* stiffener. Cường độ "tín hiệu" mà plate "nghe được" có nên phụ thuộc vào số stiffener tiếp xúc không?

### Q4. Hiểu pipeline tổng thể

Giả sử bạn đã train xong mạng với panel có **stiffener chữ T** (web + flange). Bây giờ bạn cần dự đoán cho panel có **stiffener chữ L** (chỉ web, flange ngắn ở 1 bên).

- Bạn có cần train lại không?
- Nếu cần, train lại từ đầu hay chỉ fine-tune?
- Lý luận của bạn dựa trên **cấu trúc graph** như thế nào?

### Q5. Suy luận trade-off output dimension

Output của mỗi node là vector 200 chiều = stress trên grid **10×20**.

- Tại sao paper không dùng grid **30×60** (chi tiết hơn) hay **5×10** (gọn hơn)?
- Trade-off ở đây là gì?
- Bạn sẽ chọn gì nếu *bài toán của bạn* yêu cầu localization rất sắc nét cho ứng suất concentration (vd dự đoán crack initiation)?

### Q6 (Nâng cao). Inductive bias và over-smoothing

GNN sâu (>20 layer) thường gặp vấn đề **over-smoothing** — mọi node embedding trở nên giống nhau. Paper dùng 32 layer mà vẫn predict tốt.

- Theo bạn, có những yếu tố nào trong thiết kế mạng này giúp tránh over-smoothing? (Gợi ý: nhìn lại công thức GraphSAGE, đặc biệt phần $h_v + \sum h_u$).
- Nếu graph của bạn có **đường kính lớn hơn** (vd 50 — panel rất dài), 32 layer còn đủ không? Tại sao?

### Q7 (Nâng cao). Loss function và mục tiêu kỹ thuật

Paper chọn MSE — phạt mạnh giá trị lớn — vì mục tiêu *kỹ thuật* là dự đoán đúng max stress (chỗ failure).

- Nếu mục tiêu của bạn là **dự đoán đúng phân bố ứng suất toàn cục** (cho mục đích thiết kế weight optimization), bạn sẽ chọn loss nào? Tại sao?
- Nếu mục tiêu là **bảo toàn cân bằng lực** (physics-consistent prediction), bạn có thể thêm loss term gì?

---

## 8. Đánh giá tới hạn paper — góc nhìn của reviewer

Phần này dành cho bạn nếu đang chuẩn bị seminar/thesis — bạn cần đánh giá paper *có suy nghĩ phản biện*, không chỉ ca ngợi.

### 8.1. Điểm mạnh thực sự

1. **Sáng tạo ở level abstraction** (structural unit → node). Đây là cái paper *thực sự đóng góp*.
2. **So sánh fair với baseline** (FE-vertex embedding, cùng GraphSAGE, cùng data) → kết quả 27× faster có ý nghĩa.
3. **Phân tích đa-case** (BC, nonlinearity, geometric complexity, patch loading, curvature) → robust evidence.
4. **Validation FEM cẩn thận** (Appendix A so với literature, Appendix B mesh sensitivity).
5. **Practical**: instantaneous inference → có giá trị thực tế trong optimization loop.

### 8.2. Điểm yếu/hạn chế

1. **Code không public** → khó reproduce, đặc biệt graph builder.
2. **Không thảo luận over-smoothing** trong 32-layer deep GNN — đây là một thiếu sót về phân tích.
3. **MSE loss làm predict tệ ở vùng stress thấp** — paper thừa nhận nhưng không offer solution.
4. **Generalization tới loại stiffener mới** (L, I, hat-shaped) chưa được test.
5. **Không physics-informed** — mạng có thể predict stress vi phạm cân bằng lực mà không bị penalize.
6. **Sampling strategy đơn giản** (uniform random) — có thể cải tiến.
7. **Không quantify uncertainty** — chỉ predict point estimate, không biết khi nào predict không chắc chắn.
8. **Edge không có feature** — bỏ lỡ thông tin về connection type (full weld vs bolted vs partial).

### 8.3. Câu hỏi reviewer thường hỏi

| Câu hỏi | Cách trả lời |
|---|---|
| "Tại sao 32 layer?" | "Receptive field + capacity. Nhưng có thể giảm xuống ~10 với panel size hiện tại." |
| "MSE bias toward max — fair với optimization?" | "Vâng, vì optimization quan tâm max stress. Nhưng cho monitoring có thể cần MAE." |
| "Generalization out-of-domain?" | "Paper không test. Đây là future work." |
| "So với physics-informed neural network (PINN)?" | "Bài toán khác. PINN solve PDE; paper này là surrogate from data." |

### 8.4. Hướng cải tiến nếu bạn làm paper tiếp theo

| Hướng | Ý tưởng |
|---|---|
| **Edge feature** | Thêm "stiffness của connection" làm edge attribute → dùng GNN có edge feature (NNConv) |
| **Physics-informed** | Thêm loss term: $\|\nabla \cdot \sigma\|^2$ trên mỗi unit → đảm bảo cân bằng lực local |
| **Uncertainty** | Bayesian GNN hoặc ensemble → output mean + std stress |
| **Active learning** | Sample mới ở vùng predict không chắc → giảm data needed |
| **Multi-fidelity** | Train với mix of dense FEM + coarse FEM → tận dụng cheap data |
| **Transfer learning** | Pretrain trên straight panel → fine-tune cho curved → giảm data cho case mới |
| **Generalize stiffener types** | Encode stiffener cross-section topology → generalize T → L → I |

---

## 9. Tổng kết và hướng nghiên cứu tiếp

### 9.1. Ý tưởng chính của paper trong 1 câu

> **Thay vì đại diện stiffened panel bằng từng phần tử mesh (như cách GNN-CFD truyền thống), paper biểu diễn mỗi structural unit (plate-span / web / flange) bằng 1 node — tôn trọng cấu trúc rời rạc tự nhiên của kết cấu — đạt 27× nhanh hơn và 92.3% accuracy max stress.**

### 9.2. Các quyết định thiết kế cốt lõi (tóm tắt)

| Quyết định | Giá trị | Tại sao |
|---|---|---|
| Loại GNN | GraphSAGE | Spatial method, scale tốt với graph lớn |
| Aggregator | sum | Giữ thông tin về số hàng xóm (= cường độ kết nối) |
| Activation | tanh | Smooth, phù hợp với regression (stress = giá trị thực) |
| Loss | MSE | Phạt mạnh giá trị lớn → ưu tiên đúng điểm nguy hiểm |
| Số layer | 32 | Đủ để thông tin lan toàn graph + capacity phi tuyến |
| Hidden size | 64 | Cân bằng capacity vs overfit |
| Output dim/node | 200 (10×20) | Đủ chi tiết để vẽ contour, không quá tốn |
| Node = unit | Plate-span/web/flange | Tôn trọng cấu trúc rời rạc của panel |

### 9.3. Hiệu năng

| Chỉ số | Giá trị |
|---|---|
| Trung bình max-stress accuracy | **92.3%** |
| Memory savings vs FE-vertex | **~98%** |
| Time savings vs FE-vertex | **~96%** |
| Inference time (sau train) | **gần như instantaneous** |

### 9.4. Ba bài học sâu nhất từ paper

1. **Inductive bias đúng > capacity lớn.** Mạng nhỏ với inductive bias đúng (GNN) thắng mạng lớn với inductive bias sai (CNN/MLP).

2. **Abstraction level đúng > brute-force resolution.** Cùng vấn đề, 16 node đại diện đúng > 50.000 node đại diện sai.

3. **Cấu trúc toán học của mạng nên khớp cấu trúc toán học của vật lý.** GNN message passing tự nhiên khớp với PDE local — đó là lý do nó học hiệu quả hơn.

### 9.5. Câu hỏi cuối

> Sau khi đọc xong tài liệu này, bạn có thể đóng nó lại và tự giải thích bằng lời nói cho một người bạn không biết gì về GNN/FEM, trong vòng 10 phút, theo đúng 9 phần trên không?
>
> - Nếu **CÓ** → bạn đã hiểu đến mức tái hiện được. Bắt đầu code.
> - Nếu **CHƯA** → quay lại phần đang vướng, đọc lại với câu hỏi cụ thể trong đầu. Đừng đọc passive — đọc với câu hỏi.

### 9.6. Tài liệu liên quan để đọc tiếp

| Topic | Paper/Resource |
|---|---|
| GraphSAGE gốc | Hamilton et al. 2017, "Inductive Representation Learning on Large Graphs" |
| GNN cho CFD (FE-vertex) | Pfaff et al. 2020, "Learning Mesh-Based Simulation with Graph Networks" (MeshGraphNet) |
| Over-smoothing trong deep GNN | Li et al. 2018, "Deeper Insights into GCN" |
| Physics-informed GNN | Sanchez-Gonzalez et al. 2020, "Learning to Simulate Complex Physics" |
| GNN survey | Wu et al. 2021, "A Comprehensive Survey on Graph Neural Networks" |

---

*Tài liệu được biên soạn trong vai trò research mentor, mục tiêu giúp bạn hiểu sâu đến mức tự tái hiện được nghiên cứu. Nếu phần nào còn vướng, hãy hỏi cụ thể — tôi sẽ giải thích lại bằng góc nhìn khác hoặc ví dụ khác.*
