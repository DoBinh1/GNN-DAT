# Cơ chế hoạt động của Graph Neural Network: Lý thuyết, dữ liệu và quy trình vận hành

## Tóm tắt

Báo cáo này trình bày cơ sở lý thuyết và quy trình thực hành của Graph Neural Network (GNN) trong vai trò mô hình thay thế (surrogate model) cho phương pháp phần tử hữu hạn (FEM) trong bài toán dự đoán trường ứng suất trên các kết cấu panel có gân (stiffened panel). Nội dung được tổ chức thành ba phần chính: cơ sở toán học của GNN, cấu trúc dữ liệu cùng quy trình huấn luyện, và quy trình suy luận với mô tả tường minh các phép biến đổi dữ liệu ở mỗi giai đoạn.

---

## 1. Giới thiệu và đặt vấn đề

Phân tích ứng suất trong panel có gân là một bài toán cơ học cơ bản trong thiết kế các kết cấu mỏng chịu tải, từ vỏ tàu đến vỏ máy bay và sàn cầu. Phương pháp tiêu chuẩn hiện nay là FEM với chi phí tính toán dao động từ vài phút đến vài giờ cho mỗi cấu hình panel, tùy độ tinh của lưới và độ phức tạp của bài toán. Trong bối cảnh tối ưu thiết kế, nơi cần đánh giá hàng nghìn đến hàng vạn cấu hình ứng viên, chi phí tính toán này trở thành rào cản chính.

Một mô hình thay thế dựa trên học máy có thể giải quyết rào cản này nếu thỏa mãn ba yêu cầu. Thứ nhất, mô hình phải dự đoán trường ứng suất với độ chính xác đủ tốt cho mục đích sàng lọc thiết kế. Thứ hai, mô hình phải nhanh hơn FEM nhiều bậc để vòng lặp tối ưu trở nên khả thi. Thứ ba, mô hình phải có khả năng tổng quát hóa sang các thiết kế chưa thấy trong quá trình huấn luyện, vì không gian thiết kế rộng lớn khiến việc bao phủ toàn bộ trong dataset là không thể.

GNN đáp ứng cả ba yêu cầu này nhờ một đặc điểm cốt lõi: kiến trúc của nó hoạt động trên cấu trúc đồ thị có kích thước thay đổi, phản ánh trực tiếp bản chất của bài toán panel — nơi mỗi thiết kế có số lượng và cách bố trí gân khác nhau. Phần còn lại của báo cáo đi sâu vào cách GNN thực hiện điều này.

---

## 2. Biểu diễn dữ liệu dạng đồ thị

### 2.1 Định nghĩa hình thức

Một panel được biểu diễn bởi một đồ thị vô hướng `G = (V, E, X, A)`, trong đó:

- `V = {v₁, v₂, …, v_N}` là tập `N` đỉnh (node), mỗi đỉnh tương ứng với một đơn vị kết cấu của panel: một mảnh tấm, một đoạn gân, hoặc một vùng kết cấu được phân chia theo quy ước của bài toán.
- `E ⊆ V × V` là tập cạnh, mỗi cạnh `(v_i, v_j)` thể hiện một mối nối vật lý giữa hai đơn vị kết cấu — thường là vị trí hàn giữa gân và tấm, hoặc giao điểm giữa hai gân.
- `X ∈ ℝ^{N × d_in}` là ma trận đặc trưng đỉnh, hàng `i` tương ứng vector đặc trưng `x_i ∈ ℝ^{d_in}` của đỉnh `v_i`. Trong bài toán panel, `x_i` chứa các thuộc tính vật lý của đơn vị kết cấu thứ `i`, ví dụ: chiều dài, chiều rộng, độ dày, module đàn hồi `E`, hệ số Poisson `ν`, cờ điều kiện biên, và tải áp dụng cục bộ.
- `A ∈ {0,1}^{N × N}` là ma trận kề (adjacency matrix), với `A_ij = 1` khi `(v_i, v_j) ∈ E` và `A_ij = 0` ngược lại.

Với tập hàng xóm của một đỉnh `v` ký hiệu `N(v) = {u ∈ V : (u, v) ∈ E}`, ta có thể thay thế ma trận kề bằng danh sách kề (edge index) `E_idx ∈ ℤ^{2 × M}` trong đó `M = |E|`. Cột thứ `m` của `E_idx` là cặp chỉ số `(u_m, v_m)` mô tả cạnh thứ `m`. Biểu diễn danh sách kề tiết kiệm bộ nhớ hơn ma trận kề khi đồ thị thưa, và là dạng được dùng phổ biến trong các thư viện như PyTorch Geometric.

### 2.2 Ánh xạ từ panel sang đồ thị

Quá trình biến một thiết kế panel thành đồ thị là tiền xử lý, không thuộc mô hình học. Bài toán chia thành các đơn vị kết cấu theo quy ước cố định: với panel hình chữ nhật được tăng cứng bằng các gân dọc và ngang, mỗi mảnh tấm con bị bao quanh bởi các gân tạo thành một đỉnh, mỗi đoạn gân giữa hai giao điểm tạo thành một đỉnh khác. Cạnh được tạo ra giữa hai đỉnh khi chúng có chung một biên vật lý.

Quy ước này quan trọng cho ba lý do. Một là số đỉnh `N` được giữ nhỏ — chỉ vài chục đến vài trăm — so với phép phân chia kiểu finite element nơi `N` có thể lên đến hàng vạn. Hai là cấu trúc đồ thị phản ánh trung thực mối liên kết cơ học giữa các phần của panel. Ba là quy ước phải nhất quán giữa giai đoạn huấn luyện và suy luận để các trọng số đã học áp được cho thiết kế mới.

### 2.3 Đầu ra của mô hình

Trường ứng suất trong mỗi đơn vị kết cấu được sampled trên một lưới đều `n_x × n_y` (paper Cai & Jelovica chọn `10 × 20 = 200` điểm), sau đó các giá trị này được flatten thành vector `y_i ∈ ℝ^{d_out}` với `d_out = 200`. Toàn bộ trường ứng suất của panel được biểu diễn bởi ma trận `Y ∈ ℝ^{N × d_out}`. Bài toán dự đoán quy về việc học một hàm `f : (V, E, X) ↦ Y`.

---

## 3. Cơ sở toán học của Graph Neural Network

### 3.1 Hàm mục tiêu và nguyên lý chia sẻ trọng số

Vì các đồ thị trong dataset có kích thước thay đổi (`N_i` khác nhau giữa các sample) và việc đánh số đỉnh là tùy ý, hàm `f` cần thỏa mãn hai tính chất toán học. Tính chất thứ nhất là **bất biến với kích thước đồ thị**: hàm phải áp dụng được cho `N` bất kỳ. Tính chất thứ hai là **đẳng biến với hoán vị (permutation equivariance)**: nếu hoán vị các đỉnh bởi một phép hoán vị `π`, ma trận đầu ra cũng được hoán vị tương ứng, cụ thể `f(P_π X, P_π A P_π^T) = P_π f(X, A)` với `P_π` là ma trận hoán vị tương ứng `π`.

Cả hai tính chất được đảm bảo nếu `f` được xây dựng từ các phép toán cục bộ tại từng đỉnh, sử dụng phép gộp đối xứng trên tập hàng xóm. Đây là khung khái niệm của Graph Neural Network: học một hàm cục bộ `φ` chia sẻ giữa mọi đỉnh, áp dụng lặp đi lặp lại để mô hình bao phủ thông tin từ vùng xa.

### 3.2 Khung Message Passing

Một lớp GNN thực hiện hai bước nối tiếp tại mọi đỉnh, được gọi chung là message passing. Gọi `h_v^{(k)} ∈ ℝ^{d_k}` là biểu diễn của đỉnh `v` ở lớp `k`, với quy ước `h_v^{(0)} = x_v` là feature thô.

**Bước aggregate** tính thông điệp tổng hợp từ hàng xóm:

```
m_v^{(k)} = AGG^{(k)} ( { h_u^{(k-1)} : u ∈ N(v) } )
```

trong đó `AGG^{(k)}` là một hàm đối xứng, không phụ thuộc thứ tự các phần tử của tập đầu vào. Ba lựa chọn phổ biến là:

```
AGG_mean( {z_u} ) = (1 / |N(v)|) · Σ_{u ∈ N(v)} z_u
AGG_sum ( {z_u} ) =                Σ_{u ∈ N(v)} z_u
AGG_max ( {z_u} ) = element-wise max của { z_u }
```

Phép trung bình thường được dùng khi muốn chuẩn hóa theo bậc của đỉnh, phép tổng phù hợp khi số hàng xóm mang ý nghĩa, phép max phù hợp khi muốn nổi bật đặc trưng cực trị. Tính đối xứng của `AGG` chính là cơ chế đảm bảo tính đẳng biến với hoán vị của toàn bộ mô hình.

**Bước update** tính biểu diễn mới:

```
h_v^{(k)} = σ ( W^{(k)} · [ h_v^{(k-1)} ; m_v^{(k)} ] + b^{(k)} )
```

trong đó `[ ; ]` là phép concatenation hai vector, `W^{(k)} ∈ ℝ^{d_k × (d_{k-1} + d_{k-1})}` và `b^{(k)} ∈ ℝ^{d_k}` là tham số học được, và `σ` là hàm phi tuyến (thường là ReLU). Bộ tham số `(W^{(k)}, b^{(k)})` được dùng chung cho mọi đỉnh trong lớp `k`, đây chính là nguyên lý chia sẻ trọng số.

### 3.3 Dạng ma trận của một lớp

Khi viết cho cả đồ thị một lúc, các phép toán trên gộp lại thành phép nhân ma trận. Đặt `H^{(k-1)} ∈ ℝ^{N × d_{k-1}}` là ma trận biểu diễn của tất cả đỉnh ở lớp `k-1`. Phép aggregate trung bình tương ứng với:

```
M^{(k)} = D^{-1} A H^{(k-1)}
```

với `D` là ma trận đường chéo có `D_ii = |N(v_i)|`. Mỗi hàng `i` của `M^{(k)}` chính là `m_{v_i}^{(k)}`. Bước update ghi gọn:

```
H^{(k)} = σ ( [ H^{(k-1)} ; M^{(k)} ] · W^{(k) T} + 1_N (b^{(k)})^T )
```

Toàn bộ lớp được thực hiện bằng vài phép nhân ma trận thưa, có độ phức tạp tính toán `O(M · d_k)` thay vì `O(N² · d_k)`. Đây là lý do GNN scale tốt trên đồ thị thưa và tận dụng được phần cứng GPU hiệu quả.

Một biến thể chuẩn hóa đối xứng phổ biến (Kipf & Welling 2017) thay `D^{-1} A` bằng `D̃^{-1/2} Ã D̃^{-1/2}` với `Ã = A + I` và `D̃` là ma trận bậc của `Ã`. Việc thêm self-loop `I` đảm bảo đỉnh không quên feature của chính mình, và phép chuẩn hóa căn bậc hai đối xứng có nguồn gốc từ giải tích phổ trên Laplacian đồ thị, cho training ổn định hơn.

### 3.4 Tính song song trong một lớp

Trong cùng một lớp `k`, công thức cập nhật `h_v^{(k)}` chỉ phụ thuộc vào `h_v^{(k-1)}` và `{h_u^{(k-1)} : u ∈ N(v)}` — tất cả đều là biểu diễn ở lớp trước. Không có sự phụ thuộc giữa các đỉnh tại cùng một lớp, do đó toàn bộ `N` đỉnh được cập nhật song song. Sự tuần tự duy nhất là giữa các lớp với nhau: lớp `k` chỉ bắt đầu sau khi lớp `k-1` hoàn thành cho mọi đỉnh.

Tính song song này là điều kiện tiên quyết để biểu diễn mô hình dưới dạng phép nhân ma trận, đồng thời là cơ sở toán học của tính đẳng biến với hoán vị: vì không có thứ tự xử lý đỉnh, kết quả không thể phụ thuộc vào cách đánh số.

### 3.5 Stack nhiều lớp và mở rộng vùng nhìn

Khi stack `K` lớp GNN nối tiếp, biểu diễn `h_v^{(K)}` của một đỉnh chứa thông tin của các đỉnh trong vùng `K-hop neighborhood` quanh `v`. Cụ thể, sau lớp 1, mỗi đỉnh hấp thụ thông tin từ hàng xóm trực tiếp; sau lớp 2, các hàng xóm này đã chứa thông tin từ hàng xóm của họ ở lớp 1, do đó đỉnh trung tâm gián tiếp tiếp cận thông tin cách 2 bước. Quy nạp cho thấy sau `K` lớp, vùng tiếp cận có bán kính `K`.

Mỗi lớp có bộ tham số riêng `(W^{(k)}, b^{(k)})`, được học độc lập với các lớp khác. Cấu trúc này cho phép các lớp chuyên môn hóa: lớp đầu mã hóa pattern cục bộ, lớp giữa mã hóa pattern trung gian, lớp sâu tích hợp pattern toàn cục.

Việc tăng `K` không phải là không có hạn chế. Hiện tượng over-smoothing xảy ra khi `K` quá lớn: các vector `h_v^{(K)}` của các đỉnh khác nhau hội tụ về cùng một giá trị, làm mất tính đặc thù cần thiết cho dự đoán node-level. Trong thực hành, `K ∈ {2, 3, 4}` là dải hợp lý cho hầu hết bài toán panel.

### 3.6 Kiến trúc encoder-decoder hoàn chỉnh

Một mô hình GNN hoàn chỉnh gồm hai thành phần. Encoder là chuỗi `K` lớp message passing biến đổi feature thô `H^{(0)} = X` thành biểu diễn ẩn `H^{(K)} ∈ ℝ^{N × d_h}`, với `d_h` là chiều của không gian ẩn. Decoder là một MLP nhỏ áp dụng cho từng đỉnh, biến đổi `h_v^{(K)}` thành dự đoán đầu ra:

```
ŷ_v = MLP_dec ( h_v^{(K)} ) = W_2^{dec} σ ( W_1^{dec} h_v^{(K)} + b_1^{dec} ) + b_2^{dec}
```

với `W_1^{dec} ∈ ℝ^{d_h × d_h}` và `W_2^{dec} ∈ ℝ^{d_h × d_out}`. Toàn bộ mô hình có thể viết dưới dạng hàm tổng hợp `f_θ(X, A) = Ŷ`, trong đó `θ` gồm tất cả tham số học được:

```
θ = { W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}, ..., W^{(K)}, b^{(K)}, W_1^{dec}, b_1^{dec}, W_2^{dec}, b_2^{dec} }
```

### 3.7 Bảng kích thước tensor tham khảo

Với cấu hình `K = 3, d_in = 7, d_h = 64, d_out = 200`, kích thước cụ thể của các tensor như sau:

| Đối tượng | Kích thước | Vai trò |
|---|---|---|
| `X = H^{(0)}` | `N × 7` | Feature thô đầu vào |
| `H^{(1)}` | `N × 64` | Biểu diễn ẩn sau lớp 1 |
| `H^{(2)}` | `N × 64` | Biểu diễn ẩn sau lớp 2 |
| `H^{(3)}` | `N × 64` | Biểu diễn ẩn sau lớp 3 |
| `Ŷ` | `N × 200` | Dự đoán ứng suất |
| `W^{(1)}` | `64 × 14` | (vì input concat = 7 + 7 = 14) |
| `W^{(2)}` | `64 × 128` | (vì input concat = 64 + 64 = 128) |
| `W^{(3)}` | `64 × 128` | |
| `W_1^{dec}` | `64 × 64` | Lớp ẩn của decoder |
| `W_2^{dec}` | `200 × 64` | Lớp output của decoder |

Tổng số tham số học được trong cấu hình này khoảng `30,000` — nhỏ so với các mô hình deep learning lớn, nhưng đủ để học các pattern cơ học phức tạp khi dataset đủ phong phú.

---

## 4. Chuẩn hóa và ổn định numerical

Việc training GNN ổn định trong thực tế đòi hỏi chuẩn hóa ở ba tầng nối tiếp.

**Chuẩn hóa đặc trưng đầu vào** áp dụng trên `X` trước khi đi vào mô hình. Các thuộc tính vật lý có dải giá trị chênh lệch lớn: độ dày khoảng `10⁻²` mét, module đàn hồi `E` khoảng `2 × 10¹¹` Pa, tải khoảng `10⁶` Pa. Nếu nạp thẳng các giá trị này, gradient bị thống trị bởi chiều có magnitude lớn nhất. Phép standardize từng chiều `j`:

```
x'_{ij} = (x_{ij} - μ_j) / σ_j
```

với `μ_j, σ_j` là trung bình và độ lệch chuẩn của chiều `j` tính trên toàn bộ tập huấn luyện. Sau standardize, mọi chiều có cùng thang giá trị với mean ≈ 0 và std ≈ 1.

**Chuẩn hóa trong AGG** thực hiện ngầm khi chọn `AGG_mean`, vì phép trung bình tự chia cho số hàng xóm. Khi dùng `AGG_sum`, magnitude của thông điệp tỷ lệ với bậc đỉnh, nên cần thêm một tầng chuẩn hóa khác hoặc chuẩn hóa đối xứng kiểu GCN.

**Chuẩn hóa giữa các lớp** áp dụng LayerNorm sau mỗi lớp GNN:

```
LayerNorm(h_v^{(k)}) = γ ⊙ (h_v^{(k)} - mean(h_v^{(k)})) / std(h_v^{(k)}) + β
```

với `γ, β ∈ ℝ^{d_k}` là tham số học được. LayerNorm tính trung bình và độ lệch chuẩn theo các chiều của một vector duy nhất, do đó hoạt động độc lập cho từng đỉnh và phù hợp với GNN. Tầng này giúp duy trì activation trong dải hợp lý qua nhiều lớp, cho phép dùng learning rate cao hơn.

---

## 5. Giai đoạn huấn luyện

### 5.1 Cấu trúc dataset

Dataset huấn luyện là tập hợp `D = { (G_i, Y_i) : i = 1, …, S }` gồm `S` mẫu, mỗi mẫu là một cặp đồ thị-nhãn. Trong thực hành, mỗi mẫu được lưu thành các tensor riêng:

```
D[i] = {
    X_i        : tensor float kích thước [N_i, d_in],
    edge_idx_i : tensor int kích thước [2, M_i],
    Y_i        : tensor float kích thước [N_i, d_out]
}
```

`S` thường rơi vào khoảng vài nghìn đến vài chục nghìn, tùy mức độ phong phú của không gian thiết kế. `N_i` và `M_i` thay đổi giữa các mẫu vì mỗi panel có cấu hình khác nhau.

Quy trình sinh dataset gồm các bước. Bước một, định nghĩa không gian thiết kế: dải các giá trị có thể có cho số gân, vị trí gân, độ dày tấm, tiết diện gân, vật liệu, tải, điều kiện biên. Bước hai, lấy mẫu các điểm trong không gian này — thường bằng Latin hypercube sampling hoặc sampling đều — để có `S` cấu hình ứng viên. Bước ba, với mỗi cấu hình, chạy FEM trong ABAQUS để thu được trường ứng suất chính xác. Bước bốn, trích xuất ứng suất tại các điểm sampling theo lưới `n_x × n_y` cho mỗi đơn vị kết cấu, flatten thành vector `y_i` chiều `d_out`. Bước năm, lưu cặp `(G_i, Y_i)` thành tensor.

### 5.2 Tách train, validation, test

Dataset được tách thành ba tập rời nhau với tỉ lệ thông thường `70:15:15`. Tập train dùng để cập nhật tham số, tập validation dùng để theo dõi tiến độ huấn luyện và tinh chỉnh hyperparameter, tập test dùng để đánh giá độc lập sau khi training kết thúc. Việc tách phải ngẫu nhiên trên các mẫu, không trên các đỉnh trong cùng một mẫu.

Các tham số chuẩn hóa `(μ_j, σ_j)` cho feature đầu vào và `(μ_y, σ_y)` cho output được tính chỉ trên tập train. Cụ thể:

```
μ_j = (1 / N_train_total) · Σ_i Σ_v x_{i,v,j}
σ_j² = (1 / N_train_total) · Σ_i Σ_v (x_{i,v,j} - μ_j)²
```

trong đó `N_train_total = Σ_i N_i` là tổng số đỉnh trong tập train. Các giá trị này được lưu thành file (thường là JSON hoặc NumPy `.npz`) và sẽ được tải lại trong giai đoạn suy luận.

### 5.3 Hàm mất mát

Vì đầu ra là vector liên tục, hàm mất mát tự nhiên là Mean Squared Error tính trên toàn bộ chiều của ứng suất và toàn bộ đỉnh:

```
L(θ) = (1 / (N_total · d_out)) · Σ_i Σ_v Σ_j ( ŷ_{i,v,j} - y_{i,v,j} )²
```

trong đó tổng chạy qua mọi mẫu `i` trong batch, mọi đỉnh `v` của mẫu, và mọi chiều `j` của vector ứng suất. Khi output đã được normalize, loss được tính trên không gian normalized, đảm bảo các chiều có cùng trọng số trong gradient.

Trong một số biến thể, loss có thể bổ sung term L1 cho tính bền vững với outlier, hoặc trọng số hóa theo vùng quan trọng của panel khi muốn ưu tiên độ chính xác ở vùng ứng suất cao. Trong báo cáo này, ta xét MSE thuần túy.

### 5.4 Mini-batch của đồ thị

Để tận dụng GPU, các mẫu được nhóm thành mini-batch kích thước `B`. Khác với dữ liệu ảnh nơi các sample có cùng kích thước, các đồ thị trong batch có `N_i, M_i` khác nhau, không thể stack thành một tensor đa chiều thông thường. Giải pháp chuẩn là gộp tất cả đồ thị của batch thành một đồ thị siêu lớn có cấu trúc khối đường chéo:

```
X_batch        : [N_total, d_in],         N_total = Σ_{i=1}^B N_i
edge_idx_batch : [2, M_total],            M_total = Σ_{i=1}^B M_i
Y_batch        : [N_total, d_out]
batch_vec      : [N_total]   chỉ số mẫu cho mỗi đỉnh
```

Cụ thể, `X_batch` được tạo bằng cách concat `X_1, X_2, …, X_B` theo chiều đỉnh. `edge_idx_batch` được tạo bằng cách offset các chỉ số: với mẫu `i`, mọi chỉ số đỉnh được cộng thêm `Σ_{j<i} N_j`. Kết quả là một đồ thị có `N_total` đỉnh, trong đó các đỉnh thuộc các mẫu khác nhau không có cạnh nối với nhau — tạo nên cấu trúc khối đường chéo trong ma trận kề tương ứng.

Vector `batch_vec[v]` ghi nhớ đỉnh `v` thuộc mẫu nào, hữu ích khi cần thực hiện phép gộp ở mức graph (không cần thiết cho bài toán node-level prediction nhưng vẫn được lưu để tổ chức code thống nhất).

Cấu trúc khối đường chéo đảm bảo message passing trong batch không "lẫn" giữa các mẫu: vì không có cạnh giữa các khối, thông điệp chỉ truyền trong phạm vi mỗi mẫu. Đây là lý do batch graph cho kết quả giống hệt như xử lý từng mẫu một.

### 5.5 Forward pass trong huấn luyện

Với một batch, forward pass thực hiện lần lượt:

```
H^{(0)} = X_batch                                                     # [N_total, d_in]

for k = 1, 2, ..., K:
    M^{(k)} = aggregate(H^{(k-1)}, edge_idx_batch)                    # [N_total, d_{k-1}]
    H_concat = concat(H^{(k-1)}, M^{(k)}, axis=1)                     # [N_total, 2 d_{k-1}]
    H^{(k)} = activation( H_concat @ W^{(k)}.T + b^{(k)} )            # [N_total, d_k]
    H^{(k)} = LayerNorm(H^{(k)})                                       # [N_total, d_k]

Z = activation( H^{(K)} @ W_1^{dec}.T + b_1^{dec} )                   # [N_total, d_h]
Ŷ = Z @ W_2^{dec}.T + b_2^{dec}                                       # [N_total, d_out]
```

Phép `aggregate` được thực hiện bằng phép nhân ma trận thưa giữa ma trận chuẩn hóa (`D^{-1} A` hoặc `D̃^{-1/2} Ã D̃^{-1/2}`) và `H^{(k-1)}`. Thư viện như PyTorch Geometric cung cấp các operator chuyên biệt như `scatter_mean`, `scatter_add` để thực hiện phép này hiệu quả trên GPU, không cần xây ma trận `A` đầy đủ.

### 5.6 Backpropagation và cập nhật tham số

Loss `L(θ)` được tính từ `Ŷ` và `Y_batch`. Gradient `∇_θ L` được lan truyền ngược qua đồ thị tính toán bằng autograd của framework (PyTorch hoặc TensorFlow). Vì các phép trong forward pass đều là phép nhân ma trận và activation chuẩn, autograd xử lý tự động.

Tham số được cập nhật bằng optimizer. Adam là lựa chọn mặc định với hyperparameter:

```
learning_rate    = 1e-3
β_1, β_2          = 0.9, 0.999
ε                 = 1e-8
weight_decay     = 1e-4    (regularization L2)
```

Adam thích nghi learning rate cho từng tham số dựa trên momentum bậc nhất và bậc hai của gradient, phù hợp với các bài toán có gradient không đồng đều giữa các thành phần tham số.

### 5.7 Quy trình huấn luyện một epoch

Một epoch là một lượt duyệt toàn bộ tập train. Pseudocode đầy đủ:

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        X, edge_idx, Y = batch.X, batch.edge_idx, batch.Y
        Y_pred = model(X, edge_idx)
        loss = MSE(Y_pred, Y)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            Y_pred = model(batch.X, batch.edge_idx)
            val_loss += MSE(Y_pred, batch.Y) * batch.size

    val_loss /= len(val_dataset)
    scheduler.step(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(model)
```

Các thực hành bổ sung:

`clip_grad_norm_` giới hạn L2-norm của gradient ở ngưỡng cố định (thường 1.0) để tránh hiện tượng gradient nổ, đặc biệt khi đồ thị có đỉnh bậc cao. Scheduler giảm learning rate khi validation loss không cải thiện (chẳng hạn `ReduceLROnPlateau`), giúp hội tụ ổn định ở giai đoạn cuối. Early stopping dừng training khi validation loss không cải thiện trong một số epoch liên tiếp (thường 20-30), tránh overfitting.

### 5.8 Tinh chỉnh hyperparameter

Các hyperparameter chính cần tinh chỉnh trên tập validation gồm số lớp `K ∈ {2, 3, 4}`, kích thước không gian ẩn `d_h ∈ {32, 64, 128, 256}`, lựa chọn AGG `∈ {mean, sum, max}`, learning rate `∈ [10⁻⁴, 10⁻²]`, kích thước batch `∈ {16, 32, 64, 128}`. Việc tinh chỉnh thường dùng grid search nếu không gian nhỏ, hoặc Bayesian optimization (vd: thư viện Optuna) khi không gian rộng.

Tiêu chí lựa chọn cấu hình tốt nhất là validation loss thấp nhất sau early stopping. Cần chú ý cấu hình tốt trên validation chưa chắc tốt trên test — đây là lý do giữ tập test riêng và chỉ đánh giá một lần ở cuối.

### 5.9 Lưu mô hình

Sau khi training kết thúc, các artifact sau được lưu:

```
checkpoint.pt                 # toàn bộ tham số θ
normalization_stats.json      # μ_j, σ_j cho input; μ_y, σ_y cho output
config.yaml                   # K, d_h, d_in, d_out, AGG choice, ...
```

Bộ ba này là tất cả những gì cần để chạy suy luận, không cần truy cập lại dataset huấn luyện.

---

## 6. Giai đoạn suy luận

Giai đoạn suy luận sử dụng mô hình đã huấn luyện để dự đoán trường ứng suất cho một thiết kế panel mới. Khác với huấn luyện, suy luận chỉ thực hiện forward pass một lần, không tính gradient, không cập nhật tham số.

### 6.1 Tải mô hình và thống kê chuẩn hóa

Bước đầu là tải các artifact đã lưu sau training:

```python
model = GNNModel(config)
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()

stats = json.load(open("normalization_stats.json"))
μ_x, σ_x = stats["input_mean"], stats["input_std"]    # shape [d_in]
μ_y, σ_y = stats["output_mean"], stats["output_std"]  # shape [d_out]
```

Sau bước này, mô hình ở trạng thái sẵn sàng. Mọi tensor `μ, σ` được lưu cố định và không đổi trong toàn bộ giai đoạn suy luận.

### 6.2 Đọc thiết kế và xây đồ thị

Một thiết kế panel mới được mô tả bởi các thông số kỹ thuật:

```
panel_design = {
    "geometry": {
        "length": ...,
        "width": ...,
        "thickness": ...,
        "stiffener_layout": [
            {"type": "T", "position": (...), "section": (...)},
            {"type": "T", "position": (...), "section": (...)},
            ...
        ]
    },
    "material": {"E": ..., "nu": ...},
    "boundary": {"edges_clamped": [...]},
    "load": {"pressure_field": ..., "magnitude": ...}
}
```

Bước phân tích hình học chia panel thành các đơn vị kết cấu theo quy ước đã dùng khi train. Kết quả là danh sách `N` đỉnh, mỗi đỉnh có thuộc tính vật lý cụ thể, và danh sách cạnh `M` mô tả các mối nối. Quá trình này thuần túy là tiền xử lý hình học, không liên quan đến mô hình học.

Sau bước này, ta có:

```
X       : [N, d_in]    feature thô của các đỉnh
edge_idx: [2, M]       danh sách cạnh
```

Số `N` và `M` có thể khác hoàn toàn so với mọi đồ thị trong tập train — đây là điểm mạnh của GNN.

### 6.3 Chuẩn hóa đầu vào

Feature thô được chuẩn hóa bằng `μ_x, σ_x` đã tải:

```
X_norm[v, j] = (X[v, j] - μ_x[j]) / σ_x[j]    với mọi v, j
```

Đây là bước hay bị bỏ sót khi mới làm. Việc dùng `μ, σ` từ training (không tính lại từ panel mới) là tuyệt đối quan trọng vì mô hình đã học được trong không gian đã chuẩn hóa theo các thống số đó. Nếu chuẩn hóa bằng `μ, σ` khác, đầu vào của mô hình sẽ rơi ra ngoài phân phối học, dự đoán sai lệch.

### 6.4 Forward pass không gradient

Mô hình thực hiện một forward pass duy nhất:

```python
with torch.no_grad():
    H = X_norm                                          # [N, d_in]
    for k in range(1, K + 1):
        M_k = aggregate(H, edge_idx)                    # [N, d_{k-1}]
        H_concat = concat(H, M_k, dim=1)                # [N, 2 d_{k-1}]
        H = relu(H_concat @ W[k].T + b[k])              # [N, d_k]
        H = LayerNorm[k](H)                              # [N, d_k]

    Z = relu(H @ W1_dec.T + b1_dec)                     # [N, d_h]
    Y_pred_norm = Z @ W2_dec.T + b2_dec                 # [N, d_out]
```

Việc bao toàn bộ phép toán trong `torch.no_grad()` có hai lợi ích. Thứ nhất, không lưu lại các tensor trung gian cho backprop, tiết kiệm bộ nhớ đáng kể. Thứ hai, các phép kiểm tra liên quan đến gradient được bỏ qua, tăng tốc execution. Đặt `model.eval()` thêm một bước nữa: một số layer như Dropout và BatchNorm có hành vi khác trong eval mode (Dropout không drop, BatchNorm dùng thống kê running thay vì batch).

### 6.5 Khử chuẩn hóa đầu ra

`Y_pred_norm` ở trong không gian normalized. Để có giá trị ứng suất vật lý, cần khử chuẩn hóa:

```
Y_pred[v, j] = Y_pred_norm[v, j] · σ_y[j] + μ_y[j]    với mọi v, j
```

Sau bước này, `Y_pred` có đơn vị Pascal hoặc MPa tùy quy ước của bài toán, sẵn sàng để so sánh với kết quả FEM hoặc đưa vào phân tích tiếp.

### 6.6 Hậu xử lý và sử dụng

Trường ứng suất `Y_pred ∈ ℝ^{N × 200}` cung cấp `200` giá trị ứng suất cho mỗi đơn vị kết cấu trên lưới `10 × 20`. Các phép hậu xử lý phổ biến:

- Tính ứng suất cực đại trên toàn panel: `σ_max = max(Y_pred)`.
- Xác định đơn vị kết cấu chịu ứng suất cao nhất: `v_critical = argmax_v max_j Y_pred[v, j]`.
- Kiểm tra ràng buộc thiết kế: `σ_max ≤ σ_yield · safety_factor`.
- Tái tạo trường ứng suất 2D bằng cách reshape `Y_pred[v]` từ vector 200 chiều thành ma trận `10 × 20` cho mỗi đơn vị kết cấu, sau đó assemble theo hình học panel.

### 6.7 Đặc điểm hiệu năng

Trên một panel có vài chục đỉnh, toàn bộ giai đoạn suy luận từ bước đọc thiết kế đến bước xuất kết quả tốn vài đến vài chục mili-giây trên GPU, vài chục đến vài trăm mili-giây trên CPU. So với FEM truyền thống tốn vài phút đến vài giờ, đây là cải thiện hàng nghìn đến hàng vạn lần. Bộ nhớ tiêu thụ thấp, chủ yếu chứa các tensor trung gian; với mô hình trong cấu hình tham khảo, tổng bộ nhớ inference dưới 100 MB — chạy được trên thiết bị edge nếu cần.

### 6.8 Vai trò trong vòng lặp tối ưu thiết kế

Một lần suy luận đơn lẻ chỉ là một mẩu của ứng dụng lớn hơn. Trong bài toán tối ưu thiết kế panel, một thuật toán bên ngoài — chẳng hạn giải thuật di truyền hoặc Bayesian optimization — đề xuất các cấu hình ứng viên. Mỗi cấu hình được đánh giá bằng cách:

1. Sinh thiết kế ứng viên `panel_design_t` từ thuật toán tối ưu ở vòng lặp `t`.
2. Chạy GNN suy luận để thu trường ứng suất `Y_pred`.
3. Tính các chỉ tiêu thiết kế: ứng suất cực đại, hệ số an toàn, khối lượng panel.
4. Đưa các chỉ tiêu này ngược lại cho thuật toán tối ưu để cập nhật thiết kế ứng viên.

Vòng lặp tiếp tục cho đến khi tìm được thiết kế thỏa mãn ràng buộc và tối ưu mục tiêu (thường là tối thiểu khối lượng dưới ràng buộc bền). Trong vòng lặp này, GNN đóng vai trò máy đánh giá nhanh — bản thân nó không quyết định thiết kế tốt nhất, nhưng làm cho việc đánh giá đủ rẻ để thuật toán tối ưu khám phá không gian thiết kế đủ rộng.

### 6.9 Khả năng tổng quát hóa và giới hạn áp dụng

Khả năng tổng quát hóa của mô hình phụ thuộc vào mức độ tương đồng giữa thiết kế mới và phân phối tập huấn luyện. Khi thiết kế mới cùng họ — cùng loại gân, cùng dải kích thước, cùng dải tải, cùng dải vật liệu — mô hình thường cho dự đoán chính xác. Khi thiết kế rơi ra ngoài phân phối, mô hình vẫn chạy được vì luật cục bộ vẫn áp được, nhưng độ chính xác giảm.

Trong thực hành, cần kiểm tra phân phối của thiết kế mới so với tập train trước khi tin tưởng dự đoán. Các kiểm tra cụ thể: kích thước panel có nằm trong dải train không, số gân có nằm trong dải train không, loại gân có xuất hiện trong train không, tải có vượt dải train không. Khi các kiểm tra này thất bại, dự đoán cần được xem là sơ bộ và xác nhận lại bằng FEM ở các điểm thiết kế quan trọng.

---

## 7. Tổng quát hóa qua các đồ thị có cấu trúc khác nhau

Một câu hỏi thường gặp về GNN là: tại sao một mô hình huấn luyện trên các đồ thị có `N_i` thay đổi lại áp dụng được cho một đồ thị mới có `N` hoàn toàn khác? Câu trả lời nằm ở cấu trúc tham số của mô hình.

Toàn bộ tham số học được `θ` gồm các ma trận `W^{(k)}` của lớp GNN và các ma trận của decoder. Kích thước của các ma trận này được quyết định bởi `d_in, d_h, d_out` — các hyperparameter cố định, không phụ thuộc `N`. Số đỉnh `N` chỉ quyết định số lần mà các phép message passing được áp dụng tại lớp aggregate. Với đồ thị có 12 đỉnh, mỗi lớp GNN áp luật cục bộ 12 lần (song song); với đồ thị 19 đỉnh, áp 19 lần. Cùng một bộ tham số, chỉ khác số lần áp.

Cơ sở lý thuyết của hiện tượng này là tính đẳng biến với hoán vị và bất biến với kích thước đã thảo luận ở Mục 3.1. Các tính chất này được đảm bảo bởi việc xây dựng mô hình từ các phép toán cục bộ và phép gộp đối xứng. Đặc điểm của bài toán cơ học — nơi các định luật vật lý cũng là cục bộ — làm cho việc học một luật cục bộ từ dữ liệu trở thành cách tiếp cận tự nhiên và phù hợp với bản chất bài toán.

---

## 8. Ví dụ minh họa: một bước message passing trên đồ thị 4 đỉnh

Để cụ thể hóa các phép toán đã trình bày, ta xét một đồ thị nhỏ có cấu trúc hình vuông với bốn đỉnh và bốn cạnh `(1,2), (1,3), (2,4), (3,4)`. Mỗi đỉnh có vector feature 2 chiều:

```
h_1^{(0)} = [1.0, 0.5]
h_2^{(0)} = [1.0, 0.0]
h_3^{(0)} = [2.0, 0.5]
h_4^{(0)} = [2.0, 0.0]
```

Tập hàng xóm: `N(1) = {2,3}, N(2) = {1,4}, N(3) = {1,4}, N(4) = {2,3}`.

Bước aggregate dạng trung bình tính được:

```
m_1^{(1)} = (h_2^{(0)} + h_3^{(0)}) / 2 = [1.5, 0.25]
m_2^{(1)} = (h_1^{(0)} + h_4^{(0)}) / 2 = [1.5, 0.25]
m_3^{(1)} = (h_1^{(0)} + h_4^{(0)}) / 2 = [1.5, 0.25]
m_4^{(1)} = (h_2^{(0)} + h_3^{(0)}) / 2 = [1.5, 0.25]
```

Bước concatenate ghép feature gốc với thông điệp, tạo vector `c_v ∈ ℝ^4`:

```
c_1 = [1.0, 0.5, 1.5, 0.25]
c_2 = [1.0, 0.0, 1.5, 0.25]
c_3 = [2.0, 0.5, 1.5, 0.25]
c_4 = [2.0, 0.0, 1.5, 0.25]
```

Giả sử ma trận trọng số `W^{(1)} ∈ ℝ^{3 × 4}` (tức `d_1 = 3`):

```
W^{(1)} = | 0.5  -0.2   0.1 |
          | 0.3   0.4  -0.1 |
          | 0.1   0.2   0.5 |
          |-0.4   0.1   0.3 |
```

Bước update với `b^{(1)} = 0` cho ra `h_v^{(1)} = ReLU(W^{(1) T} c_v)`. Tính cho đỉnh 1:

```
W^{(1) T} c_1 = [
    0.5(1.0) + 0.3(0.5) + 0.1(1.5) + (-0.4)(0.25),
   -0.2(1.0) + 0.4(0.5) + 0.2(1.5) + 0.1(0.25),
    0.1(1.0) + (-0.1)(0.5) + 0.5(1.5) + 0.3(0.25)
] = [0.7, 0.325, 0.875]

h_1^{(1)} = ReLU([0.7, 0.325, 0.875]) = [0.7, 0.325, 0.875]
```

Tính tương tự cho các đỉnh khác. Quan sát rằng trong đồ thị đối xứng này, các thông điệp `m_v^{(1)}` đồng nhất ở cả bốn đỉnh, do đó sự khác biệt giữa các `h_v^{(1)}` chỉ đến từ sự khác biệt trong feature gốc `h_v^{(0)}`. Trong các bài toán thực tế nơi cần phân biệt các đỉnh có cấu trúc lân cận giống nhau, edge feature hoặc positional encoding thường được thêm vào để phá tính đối xứng.

---

## 9. Tổng kết

Graph Neural Network cung cấp một khung lý thuyết và thực hành để xây dựng các mô hình thay thế cho FEM trong các bài toán cơ học có cấu trúc đồ thị thay đổi giữa các mẫu. Cốt lõi của GNN là một bộ tham số nhỏ mã hóa luật cục bộ tại từng đỉnh, được áp dụng song song qua nhiều lớp với trọng số riêng cho từng lớp, và được kết hợp với một decoder để dịch sang đại lượng vật lý đầu ra. Vì luật cục bộ độc lập với kích thước đồ thị, cùng một mô hình áp được cho mọi panel cùng họ, kể cả những panel chưa thấy trong dataset huấn luyện.

Trong giai đoạn huấn luyện, dataset gồm hàng nghìn cặp đồ thị-nhãn được sinh từ FEM, được tách thành ba tập train, validation, test. Mô hình được huấn luyện bằng cách tối thiểu MSE giữa dự đoán và ground truth qua backpropagation và optimizer Adam, với các kỹ thuật bổ sung gồm gradient clipping, learning rate scheduling, và early stopping. Các thống kê chuẩn hóa được tính chỉ trên tập train và lưu cùng checkpoint.

Trong giai đoạn suy luận, một thiết kế panel mới được biến thành đồ thị, feature được chuẩn hóa bằng các thống số từ training, đẩy qua mô hình một lần với tốc độ vài mili-giây, và đầu ra được khử chuẩn hóa thành trường ứng suất vật lý. Quy trình này được dùng lặp đi lặp lại hàng nghìn lần trong các vòng lặp tối ưu thiết kế, biến bài toán tối ưu vốn không khả thi với FEM thành khả thi với GNN.

Mọi biến thể GNN — GCN, GraphSAGE, GAT, MPNN — đều có thể đọc dưới ngôn ngữ chung của message passing, khác nhau ở chi tiết hàm aggregate hoặc cách kết hợp feature cạnh. Khi đã nắm chắc khung này, việc đọc paper hay cài đặt mô hình mới trở thành công việc tinh chỉnh chứ không còn là tìm hiểu khái niệm mới.

---

## Tài liệu tham khảo chính

- Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.
- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). *Graph Attention Networks*. ICLR.
- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). *Neural Message Passing for Quantum Chemistry*. ICML.
- Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2019). *A Comprehensive Survey on Graph Neural Networks*. arXiv:1901.00596.
- Cai, Y., & Jelovica, J. (2024). *Efficient Graph Representation in Graph Neural Networks for Stress Predictions in Stiffened Panels*. Thin-Walled Structures, 203, 112157.
