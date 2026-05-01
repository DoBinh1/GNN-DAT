# Báo cáo khoa học: Mạng nơ-ron đồ thị cho dự đoán ứng suất trong tấm có gân gia cường — phương pháp biểu diễn đồ thị hiệu quả

> **Báo cáo dựa trên paper:** Yuecheng Cai, Jasmin Jelovica (2024). *Efficient graph representation in graph neural networks for stress predictions in stiffened panels*. Thin-Walled Structures 203, 112157.
>
> **Mục tiêu báo cáo:** trình bày toàn bộ phương pháp luận, cơ sở lý thuyết, chi tiết kỹ thuật và kết quả của nghiên cứu một cách liền mạch, đủ để người đọc có thể mô tả lại, thiết kế lại pipeline và triển khai lại bằng mã nguồn.

---

## 1. Giới thiệu bài toán

Trong kỹ thuật kết cấu hiện đại, tấm có gân gia cường (stiffened panel) đóng vai trò là khối xây dựng cơ bản của hầu hết các kết cấu vỏ mỏng quy mô lớn — từ thân tàu thủy, vỏ máy bay, sàn cầu thép cho đến bồn chứa áp lực. Một tấm có gân gia cường về bản chất là sự kết hợp giữa một tấm phẳng (plate) tương đối mỏng và một hệ thống các gân (stiffeners) được hàn cứng vào tấm đó nhằm tăng độ cứng chống cong vênh. Triết lý thiết kế này cho phép kết cấu vừa nhẹ nhờ tấm mỏng, vừa cứng nhờ các gân chịu tải, do đó nó gần như xuất hiện trong mọi ứng dụng yêu cầu tỷ lệ cứng-trên-trọng-lượng cao. Khi tải trọng bên ngoài — chẳng hạn áp lực nước, sóng biển hay hàng hóa trên boong — tác dụng lên panel, mỗi điểm trong kết cấu sẽ chịu một mức ứng suất khác nhau. Điều này dẫn đến các vùng tập trung ứng suất, đặc biệt tại những giao tuyến hàn giữa tấm và gân, nơi nguy cơ phá hủy là cao nhất. Bài toán đặt ra cho người thiết kế là phải dự đoán chính xác trường ứng suất trên toàn bộ kết cấu, từ đó kiểm chứng độ bền và tối ưu hóa hình học.

Phương pháp tiêu chuẩn để giải quyết bài toán này là **Phương pháp Phần tử Hữu hạn** (Finite Element Method — FEM), được hiện thực trong các phần mềm thương mại như ABAQUS, ANSYS hay NASTRAN. Trực giác của FEM là chia nhỏ kết cấu liên tục thành hàng nghìn đến hàng triệu phần tử rời rạc, viết phương trình cân bằng cho từng phần tử, rồi giải đồng thời hệ phương trình tuyến tính khổng lồ thu được. Phương pháp này có ưu điểm vượt trội về độ chính xác toán học và tính linh hoạt: nó xử lý được hình học phức tạp, vật liệu phi tuyến và các loại tải trọng đa dạng, đồng thời được đảm bảo hội tụ về lời giải đúng khi mesh đủ mịn. Nhờ những tính chất này, FEM đã trở thành tiêu chuẩn vàng (ground truth) trong cộng đồng cơ học kết cấu suốt nhiều thập kỷ.

Tuy nhiên, FEM bộc lộ những hạn chế nghiêm trọng khi đặt vào bối cảnh **tối ưu hóa thiết kế**. Trong thực tế kỹ thuật, người ta hiếm khi chỉ phân tích một thiết kế đơn lẻ; thay vào đó, các thuật toán tối ưu — như giải thuật di truyền, tối ưu hóa bầy đàn hay tối ưu Bayesian — đòi hỏi đánh giá hàng nghìn đến hàng vạn phương án thiết kế khác nhau để tìm ra cấu hình tối ưu về trọng lượng, độ bền hoặc chi phí. Khi mỗi đánh giá riêng lẻ tiêu tốn vài phút đến vài giờ chạy FEM, tổng chi phí tính toán có thể lên tới hàng tháng đến hàng năm CPU-time. Hơn nữa, mỗi khi hình học thay đổi — chẳng hạn khi số gân tăng từ bốn lên năm — toàn bộ lưới mesh phải được tạo lại từ đầu, và kết quả của các thiết kế trước đó không thể tái sử dụng. Ba điểm yếu cốt lõi này không phải là khuyết tật cài đặt mà là bản chất của FEM: nó giải bài toán cụ thể chứ không *học* từ kinh nghiệm.

Để vượt qua những giới hạn ấy, cộng đồng nghiên cứu đã phát triển khái niệm **Mô hình bậc rút gọn** (Reduced-Order Model — ROM) — những mô hình thay thế nhanh hơn FEM nhiều lần nhưng vẫn duy trì độ chính xác chấp nhận được. Các thế hệ ROM đầu tiên dựa trên phương pháp thống kê như Kriging, Radial Basis Function, hay Multivariate Adaptive Regression Splines, nhưng những phương pháp này vấp phải hạn chế khi số biến thiết kế lớn hoặc khi hình học thay đổi tô-pô. Sự ra đời của học máy và đặc biệt là học sâu mang đến hướng đi mới: các mạng nơ-ron có khả năng học những ánh xạ phi tuyến phức tạp từ dữ liệu, và do đó hứa hẹn xử lý được các bài toán cơ học có tính phi tuyến cao.

Tuy nhiên, không phải mọi kiến trúc mạng đều phù hợp với bài toán stiffened panel. Mạng đa lớp truyền thẳng (Multi-Layer Perceptron — MLP) yêu cầu vector đầu vào có kích thước cố định, điều bất khả thi khi số gân thay đổi giữa các thiết kế. Mạng tích chập (Convolutional Neural Network — CNN) lại giả định dữ liệu nằm trên lưới đều, do đó buộc phải "ép" kết cấu ba chiều thành ảnh hai chiều — quá trình mất mát thông tin và không tôn trọng quan hệ kết nối tô-pô giữa các thành phần. Chính từ những hạn chế này, **Mạng nơ-ron đồ thị** (Graph Neural Network — GNN) nổi lên như một lựa chọn tự nhiên: vì một stiffened panel vốn dĩ đã có cấu trúc đồ thị (các thành phần là đỉnh, các mối hàn là cạnh), việc dùng GNN không phải là sự ép buộc mà là sự tôn trọng cấu trúc vật lý sẵn có. Báo cáo này trình bày một nghiên cứu áp dụng GNN cho bài toán dự đoán ứng suất trong stiffened panel, với điểm sáng tạo cốt lõi là một cách biểu diễn đồ thị mới — biểu diễn theo đơn vị cấu trúc (structural unit-vertex embedding) — giảm đáng kể chi phí tính toán mà vẫn duy trì độ chính xác cao.

---

## 2. Cơ sở lý thuyết

### 2.1. Khái niệm đồ thị và biểu diễn dữ liệu dưới dạng đồ thị

Để hiểu được cơ chế hoạt động của GNN, trước tiên cần làm rõ khái niệm đồ thị về mặt toán học và cách dữ liệu thực tế được mã hóa vào cấu trúc này. Một đồ thị được định nghĩa là bộ ba $G = (V, E, A)$, trong đó $V$ là tập hợp các đỉnh (vertices, đôi khi gọi là nodes), $E$ là tập hợp các cạnh (edges) nối giữa hai đỉnh, và $A \in \mathbb{R}^{N \times N}$ là ma trận kề mã hóa toàn bộ thông tin kết nối giữa các đỉnh. Cụ thể, nếu $|V| = N$ là tổng số đỉnh, thì $A_{ij} = 1$ nếu tồn tại cạnh giữa đỉnh $i$ và đỉnh $j$, và bằng $0$ trong trường hợp ngược lại. Trong nghiên cứu này, các đồ thị được giả định là vô hướng, do đó $A_{ij} = A_{ji}$ với mọi cặp đỉnh.

Mỗi đỉnh trong đồ thị có thể mang theo một vector đặc trưng (feature vector) $x_i \in \mathbb{R}^D$ mô tả thuộc tính của đỉnh đó. Khi gộp đặc trưng của tất cả $N$ đỉnh, ta thu được ma trận đặc trưng $X \in \mathbb{R}^{N \times D}$. Như vậy, một đồ thị có gắn thuộc tính được mô tả đầy đủ bởi cặp $(X, A)$ — ma trận đặc trưng quy định "nội dung" của từng đỉnh, còn ma trận kề quy định "cấu trúc" liên kết giữa chúng.

Cách biểu diễn này đặc biệt phù hợp với các hệ thống có cấu trúc rời rạc và quan hệ kết nối phức tạp — từ mạng xã hội, mạng phân tử trong hóa học, hệ thống giao thông cho đến kết cấu cơ khí. Đối với một stiffened panel, ta có thể tự nhiên ánh xạ các thành phần kết cấu thành đỉnh, các mối hàn giữa chúng thành cạnh, và các thuộc tính hình học cũng như điều kiện biên của mỗi thành phần thành vector đặc trưng tại đỉnh tương ứng. Sự tự nhiên của ánh xạ này chính là lý do GNN trở thành công cụ phù hợp cho bài toán đang xét.

### 2.2. Bản chất của Mạng nơ-ron đồ thị

Mạng nơ-ron đồ thị là một họ kiến trúc mạng được thiết kế để xử lý dữ liệu có cấu trúc đồ thị, kế thừa tinh thần của các mạng truyền thống nhưng tôn trọng đặc thù tô-pô của dữ liệu đầu vào. Khác với CNN — vốn tận dụng tính bất biến với phép tịnh tiến trên lưới đều — GNN tận dụng hai thiên kiến cấu trúc khác có ý nghĩa hơn đối với dữ liệu đồ thị: tính bất biến với hoán vị (permutation invariance) và tính địa phương của kết nối (local connectivity). Tính bất biến với hoán vị đảm bảo rằng kết quả tính toán không phụ thuộc vào thứ tự đánh số đỉnh — một thuộc tính quan trọng vì các đỉnh trong đồ thị không có thứ tự tự nhiên. Tính địa phương của kết nối đảm bảo rằng mỗi đỉnh chỉ "nhìn thấy" thông tin từ những đỉnh kề nó, qua đó phản ánh đúng cách thông tin (hoặc trong bài toán cơ học, ứng suất) lan truyền cục bộ trong hệ thống.

Hai đặc tính này khiến GNN có khả năng tổng quát hóa tốt với dữ liệu mà MLP và CNN gặp khó khăn. Một mạng GNN có thể xử lý các đồ thị với số đỉnh khác nhau mà không cần thay đổi tham số, vì mỗi đỉnh được xử lý theo cùng một quy tắc cục bộ dựa trên hàng xóm của nó. Đây là tính chất "inductive" — mạng học được quy luật cục bộ rồi áp dụng cho mọi kích thước đồ thị, không phải học cho từng cấu hình cụ thể.

### 2.3. Cơ chế truyền thông tin (Message Passing)

Cốt lõi của GNN là cơ chế **Message Passing** (truyền thông tin) — một khung lý thuyết thống nhất bao quát hầu hết các biến thể GNN hiện đại. Trực giác của cơ chế này có thể được mô tả qua một ẩn dụ đơn giản: hãy hình dung mỗi đỉnh là một người dân trong một ngôi làng, và các cạnh là những con đường nối giữa các nhà. Ban đầu, mỗi người chỉ biết về bản thân mình — tức là vector đặc trưng đầu vào. Trong vòng tin đồn thứ nhất, mỗi người gặp những người hàng xóm trực tiếp, hỏi thăm và tổng hợp thông tin từ họ vào hiểu biết của bản thân. Trong vòng thứ hai, vì các hàng xóm đã biết về hàng xóm-của-hàng-xóm rồi, thông tin đó cũng được truyền tiếp. Sau $L$ vòng (tương ứng $L$ lớp mạng), mỗi đỉnh đã thu thập được thông tin từ tất cả các đỉnh nằm trong khoảng cách $L$ bước đi trên đồ thị.

Về mặt toán học, một lớp message passing tổng quát được biểu diễn bằng phương trình:

$$
\mathbf{h}_v^{l} = \gamma_\Theta\!\left(\mathbf{h}_v^{l-1},\;\bigoplus_{u \in \mathcal{N}(v)} \phi_\Theta\!\left(\mathbf{h}_v^{l-1}, \mathbf{h}_u^{l-1}, \mathbf{e}_{u,v}\right)\right),
$$

trong đó $\mathbf{h}_v^l$ là vector ẩn của đỉnh $v$ tại lớp $l$, $\mathcal{N}(v)$ là tập các đỉnh kề $v$, $\phi_\Theta$ là hàm tạo "thông điệp" gửi từ đỉnh láng giềng, $\bigoplus$ là phép gộp bất biến với hoán vị (chẳng hạn tổng, trung bình hay cực đại), và $\gamma_\Theta$ là hàm cập nhật trạng thái mới của đỉnh dựa trên trạng thái cũ và thông tin gộp từ láng giềng. Quá trình này lặp lại qua nhiều lớp, mỗi lớp mở rộng "vùng tiếp nhận" (receptive field) của mỗi đỉnh thêm một bước.

Sự thanh lịch của cơ chế này nằm ở chỗ nó tự nhiên khớp với cấu trúc của các phương trình vi phân địa phương trong vật lý. Phương trình cân bằng cơ học có dạng $\nabla \cdot \boldsymbol{\sigma} + \mathbf{b} = \mathbf{0}$, vốn là một phương trình địa phương — ứng suất tại một điểm chỉ phụ thuộc vào trạng thái các điểm lân cận. Khi giải bằng FEM, ta thu được hệ phương trình $\mathbf{K} \mathbf{u} = \mathbf{f}$ với ma trận độ cứng $\mathbf{K}$ thưa, mỗi đỉnh chỉ ghép cặp với các đỉnh kề nó. Như vậy, cấu trúc toán học của message passing trong GNN khớp tự nhiên với cấu trúc toán học của bài toán cơ học mà ta muốn xấp xỉ — đây là lý do sâu xa giải thích vì sao GNN học hiệu quả với dữ liệu vật lý.

### 2.4. GraphSAGE trong bối cảnh bài toán

Trong rừng các biến thể GNN, nghiên cứu này lựa chọn **GraphSAGE** (Graph Sampling and Aggregation) — một kiến trúc thuộc nhóm phương pháp không gian (spatial method) được Hamilton và cộng sự đề xuất năm 2017. Lý do của lựa chọn này có thể lý giải qua một vài tiêu chí thực tiễn. Thứ nhất, GraphSAGE thực hiện tích chập trực tiếp trên không gian đỉnh, tránh được sự đắt đỏ tính toán của các phương pháp phổ (như Spectral CNN hay ChebNet) vốn yêu cầu phân rã trị riêng của ma trận Laplacian. Thứ hai, GraphSAGE cho phép linh hoạt chọn hàm gộp — một tính chất quan trọng vì, như sẽ phân tích trong các phần tiếp theo, hàm gộp ảnh hưởng trực tiếp đến khả năng phân biệt các cấu hình kết cấu khác nhau. Thứ ba, GraphSAGE có khả năng "inductive" — mạng học từ một tập đồ thị này có thể tổng quát hóa cho các đồ thị mới có kích thước khác, đáp ứng yêu cầu xử lý các stiffened panel với số gân thay đổi.

Trong nghiên cứu này, biến thể GraphSAGE được sử dụng có dạng cụ thể:

$$
\mathbf{h}_v^{l} = \sigma\!\left(\mathbf{W}^{l}\!\left(\mathbf{h}_v^{l-1} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{l-1}\right)\right),
$$

với $\mathbf{W}^l$ là ma trận trọng số học được tại lớp $l$, $\sigma$ là hàm kích hoạt phi tuyến (paper chọn $\tanh$), và phép gộp được chọn là phép cộng (sum). Công thức này, dù trông đơn giản, ẩn chứa nhiều quyết định thiết kế quan trọng sẽ được phân tích chi tiết trong Mục 4.

---

## 3. Phương pháp đề xuất

Đóng góp cốt lõi của nghiên cứu này là một cách biểu diễn đồ thị mới cho stiffened panel — gọi là **biểu diễn theo đơn vị cấu trúc** (structural unit-vertex embedding) — trái ngược với cách biểu diễn truyền thống dựa trên phần tử hữu hạn (finite element-vertex embedding). Phần này trình bày chi tiết hai cách tiếp cận, ý nghĩa vật lý của các thành phần đồ thị, và phân tích vì sao phương pháp đề xuất vừa giảm chi phí tính toán mà vẫn duy trì độ chính xác.

### 3.1. Cách biểu diễn đồ thị theo phần tử hữu hạn

Cách biểu diễn truyền thống, vốn đã được áp dụng thành công trong các bài toán cơ học chất lỏng (CFD) sử dụng GNN — chẳng hạn MeshGraphNet của DeepMind — gán mỗi phần tử hữu hạn trong mesh ABAQUS thành một đỉnh trong đồ thị. Hai phần tử kề nhau trong mesh sẽ được nối bằng một cạnh trong đồ thị tương ứng. Phương pháp này có ưu điểm là *trực tiếp*: ta đơn giản chuyển dữ liệu mesh có sẵn từ ABAQUS thành đồ thị mà không cần xử lý phức tạp.

Tuy nhiên, hệ quả ngay lập tức của cách tiếp cận này là số đỉnh trong đồ thị tỷ lệ thuận với độ mịn của mesh. Trong nghiên cứu này, mesh được sử dụng có 15 phần tử giữa hai gân, 15 phần tử cho chiều cao gân, và 8 phần tử cho chiều rộng cánh gân. Một panel điển hình với năm gân sẽ có hàng vạn phần tử và do đó hàng vạn đỉnh trong đồ thị. Khi đồ thị lớn như vậy đi qua một mạng GraphSAGE 32 lớp, bộ nhớ GPU cần thiết tăng cao và thời gian huấn luyện kéo dài đáng kể. Cụ thể, với batch size 64, cách biểu diễn này tiêu tốn 23.4 GB VRAM và mỗi epoch huấn luyện mất khoảng 6.94 giây — những con số gây trở ngại cho việc nghiên cứu trên các máy tính thông thường.

### 3.2. Cách biểu diễn đồ thị theo đơn vị cấu trúc

Nhận ra rằng số lượng đỉnh khổng lồ là không cần thiết, các tác giả đề xuất một cách biểu diễn ở mức độ trừu tượng cao hơn nhưng vẫn bảo toàn cấu trúc vật lý. Trong cách tiếp cận mới này, mỗi *đơn vị cấu trúc* được ánh xạ thành một đỉnh duy nhất. Đối với stiffened panel, một đơn vị cấu trúc được định nghĩa là một trong ba loại thành phần: (i) tấm con (plate-span) nằm giữa hai gân hoặc giữa một gân và mép tấm, (ii) phần đứng của gân (stiffener web), và (iii) phần ngang của gân, tức cánh gân (stiffener flange) — chỉ tồn tại với gân chữ T.

Với cách định nghĩa này, một panel có năm gân chữ T phân bố trên tấm sẽ chỉ chứa khoảng 16 đỉnh: sáu tấm con, năm web và năm flange. Đây là sự rút gọn đáng kể so với hàng vạn đỉnh ở cách truyền thống. Triết lý của cách tiếp cận này là: *kết cấu rời rạc vốn đã có cấu trúc tự nhiên — hãy tôn trọng cấu trúc đó*. Mỗi thành phần kết cấu trong stiffened panel được sản xuất riêng và hàn lại với nhau, vì vậy chúng tự thân là những đơn vị có ý nghĩa vật lý độc lập, và việc gán mỗi đơn vị một đỉnh là sự tương ứng tự nhiên giữa biểu diễn toán học và bản chất kết cấu.

### 3.3. Ý nghĩa vật lý của đỉnh, cạnh và đặc trưng

Khi đã định nghĩa được "đỉnh là gì", câu hỏi tiếp theo là mỗi đỉnh mang theo thông tin gì. Mỗi đỉnh trong cách biểu diễn đề xuất chứa một vector đặc trưng tám chiều, gồm: chiều rộng, chiều dài, độ dày của đơn vị cấu trúc; bốn giá trị mô tả điều kiện biên trên bốn cạnh của nó (free, simply supported hoặc fixed); và áp suất tải tác dụng. Khi sử dụng tải dạng patch (tải tập trung trên một vùng nhỏ), thêm tọa độ vị trí áp tải làm đặc trưng bổ sung. Điều đáng lưu ý là vector đặc trưng *không* chứa thông tin về vị trí tương đối của đơn vị trong panel — bởi vì thông tin đó được mã hóa ngầm thông qua tô-pô của đồ thị: một tấm con nằm ở mép có ít hàng xóm hơn một tấm ở giữa, và mạng có thể tự suy ra điều này thông qua message passing. Đây là một minh họa cho nguyên tắc rằng cấu trúc đồ thị có thể "miễn phí" cung cấp những thông tin mà nếu dùng MLP thì phải mã hóa thủ công.

Cạnh trong đồ thị mang ý nghĩa vật lý là *mối hàn* — đường truyền ứng suất giữa hai thành phần kết cấu kề nhau. Cụ thể, một cạnh được tạo giữa hai đỉnh nếu và chỉ nếu hai đơn vị cấu trúc tương ứng chia sẻ một đường hàn. Trong cách hiện thực của paper, các cạnh không có thuộc tính riêng (edge feature) — tô-pô của chúng đã đủ để mô tả kết nối, vì mọi mối hàn được giả định là cứng và đồng nhất. Đây là một đơn giản hóa hợp lý cho stiffened panel hàn cứng truyền thống, mặc dù nó cũng là điểm có thể cải tiến trong các nghiên cứu tương lai cho những kết cấu có nhiều loại liên kết khác nhau.

Đầu ra của mỗi đỉnh không phải là một con số ứng suất duy nhất, mà là một trường ứng suất hai chiều được rời rạc hóa thành lưới $10 \times 20$, sau đó được làm phẳng thành vector 200 chiều. Tức là, mạng dự đoán ứng suất tại 200 điểm sampling phân bố đều trên bề mặt của mỗi đơn vị cấu trúc. Đây là sự dung hòa giữa độ chi tiết và chi phí tính toán: 200 điểm đủ để vẽ contour ứng suất với độ trung thực tốt, nhưng không quá tốn bộ nhớ. Phụ lục B của paper chứng minh rằng việc tăng độ phân giải lưới lên $30 \times 60$ không cải thiện độ chính xác đáng kể, trong khi giảm xuống $5 \times 10$ thì mất chi tiết tại các vùng tập trung ứng suất.

### 3.4. Phân tích so sánh và lý giải hiệu quả

Sự khác biệt giữa hai cách biểu diễn không đơn thuần là vấn đề số lượng — nó phản ánh hai triết lý đối lập về mức độ trừu tượng. Cách biểu diễn theo phần tử hữu hạn giữ nguyên độ phân giải mesh và xử lý mọi điểm như nhau; cách biểu diễn theo đơn vị cấu trúc thừa nhận rằng kết cấu có cấu trúc rời rạc tự nhiên và rằng mỗi đơn vị có thể được mô tả gọn bằng vài tham số hình học. Để hiểu tại sao cách thứ hai vẫn duy trì được độ chính xác, cần lưu ý rằng trong stiffened panel, trường ứng suất trên mỗi đơn vị cấu trúc thường khá "trơn" — nó biến thiên một cách có thể dự đoán được dựa trên hình học và điều kiện biên của đơn vị, và chỉ "nhảy bậc" tại biên giữa hai đơn vị (nơi có mối hàn). Do đó, việc nén toàn bộ đơn vị thành một đỉnh không làm mất bản chất của trường ứng suất, miễn là đầu ra cho phép biểu diễn được sự biến thiên trong đơn vị (chính là vai trò của lưới $10 \times 20$).

Nguồn gốc của hiệu quả tính toán có thể được phân tích định lượng. Độ phức tạp của một mạng GNN trong huấn luyện tỷ lệ tuyến tính với số đỉnh $N$:

$$
\mathcal{O}\!\left(N \times \sum_{i=1}^L F_{i-1}\,F_i\right) + \mathcal{O}\!\left(N \times k \times \sum_{i=1}^L F_i\right) \;\propto\; \mathcal{O}(N),
$$

trong đó $L$ là số lớp, $F_i$ là số chiều ẩn tại lớp $i$, và $k$ là số hàng xóm trung bình. Số hạng đầu ứng với chi phí biến đổi đặc trưng tại mỗi đỉnh, số hạng thứ hai ứng với chi phí gộp thông tin từ hàng xóm. Vì các tổng trong dấu ngoặc là hằng số đối với một kiến trúc cố định, toàn bộ chi phí tỷ lệ tuyến tính với $N$. Khi $N$ giảm từ khoảng 10.000 xuống còn khoảng 16 — tức 600 lần — về lý thuyết chi phí cũng giảm cỡ ấy. Trong thực tế đo đạc, do còn nhiều chi phí phụ trợ (overhead trong PyTorch Geometric, batch normalization, bộ nhớ trung gian), tốc độ tăng được khoảng 27 lần — vẫn là một cải thiện đáng kể, đủ để chuyển một bài toán không khả thi trên máy tính cá nhân thành bài toán chạy được trong vài chục phút.

---

## 4. Kiến trúc mô hình và thuật toán

Phần này đi sâu vào kiến trúc cụ thể của mạng được sử dụng, công thức của lớp GraphSAGE, các quyết định thiết kế đi kèm, hàm mất mát và bộ siêu tham số.

### 4.1. Công thức GraphSAGE và phân tích từng thành phần

Như đã nêu ở Mục 2, lớp GraphSAGE trong nghiên cứu này có dạng:

$$
\mathbf{h}_v^{l} = \sigma\!\left(\mathbf{W}^{l}\!\left(\mathbf{h}_v^{l-1} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{l-1}\right)\right).
$$

Để hiểu sâu sắc công thức này, cần phân tích vai trò của từng thành phần. Tổng $\sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{l-1}$ là phép gộp thông tin từ các đỉnh kề: tại lớp $l$, đỉnh $v$ thu thập trạng thái của tất cả hàng xóm tại lớp $l-1$ và cộng chúng lại. Phép cộng này có tính chất bất biến với hoán vị — nếu ta đảo thứ tự các hàng xóm, kết quả không đổi — và do đó đảm bảo mạng tôn trọng tính chất "không có thứ tự" của các đỉnh kề trong đồ thị vô hướng. Việc cộng thêm trạng thái của bản thân $\mathbf{h}_v^{l-1}$ trước khi đi qua biến đổi tuyến tính $\mathbf{W}^l$ tương đương với một dạng "vòng lặp tự thân" (self-loop) ngầm: đỉnh $v$ luôn coi chính nó như một hàng xóm của mình. Cấu trúc cộng này còn có tác dụng phụ quan trọng — nó tạo ra một liên kết phần dư (residual-like connection) giúp gradient lan truyền ổn định trong mạng sâu 32 lớp, một điểm sẽ được thảo luận thêm dưới đây.

Ma trận $\mathbf{W}^l$ chứa các tham số có thể học được tại lớp $l$, có nhiệm vụ "trộn" thông tin gộp được thành biểu diễn mới. Hàm kích hoạt $\sigma = \tanh$ đem lại tính phi tuyến — không có nó, toàn bộ mạng dù sâu vẫn chỉ là một biến đổi tuyến tính lớn. Lựa chọn $\tanh$ thay vì ReLU trong nghiên cứu này có lý do cụ thể: đầu ra của mạng là giá trị ứng suất, một đại lượng thực có thể dương hoặc âm, và do đó hàm kích hoạt mượt như $\tanh$ phù hợp hơn ReLU vốn cắt giá trị âm. Thêm vào đó, $\tanh$ cho gradient ổn định hơn trong mạng sâu, giảm nguy cơ "neuron chết" mà ReLU đôi khi gặp phải.

### 4.2. Ba giai đoạn của message passing và phân tích lựa chọn aggregator

Có thể phân tích quá trình tính toán trong mỗi lớp thành ba giai đoạn rõ ràng. **Giai đoạn aggregation** (gộp): mỗi đỉnh thu thập các vector trạng thái của hàng xóm và gộp chúng bằng một hàm bất biến với hoán vị. **Giai đoạn propagation** (lan truyền): kết quả gộp được kết hợp với trạng thái cũ của đỉnh — trong công thức này là phép cộng trực tiếp, mặc dù có những biến thể GNN khác sử dụng phép nối (concatenation) hoặc cơ chế attention. **Giai đoạn update** (cập nhật): trạng thái mới được tạo ra qua biến đổi tuyến tính cộng với hàm kích hoạt phi tuyến.

Trong các hàm gộp khả dĩ — tổng (sum), trung bình (mean) và cực đại (max) — paper chọn tổng. Lựa chọn này không tùy tiện mà có cơ sở lý thuyết và vật lý rõ ràng. Theo Xu và cộng sự (2019) trong công trình *"How Powerful Are Graph Neural Networks?"*, tổng là hàm gộp có tính chất "đơn ánh" (injective) mạnh nhất — tức là nó phân biệt được nhiều cấu hình hàng xóm hơn các hàm khác. Cụ thể, hãy xét hai panel: panel A có một tấm con kẹp giữa hai gân cao 100 mm, panel B có cùng tấm con kẹp giữa bốn gân cao 100 mm. Về mặt cơ học, hai panel có độ cứng chống cong rất khác nhau vì số gân khác nhau. Nếu sử dụng hàm gộp trung bình, kết quả gộp tại tấm con sẽ giống nhau ở cả hai panel (cùng là vector đặc trưng của một gân cao 100 mm), và mạng sẽ không phân biệt được hai trường hợp. Ngược lại, với hàm gộp tổng, kết quả là $2 \mathbf{h}$ trong trường hợp đầu và $4 \mathbf{h}$ trong trường hợp sau — sự khác biệt được bảo toàn. Như vậy, lựa chọn hàm tổng phản ánh một sự thật vật lý: số lượng kết nối ảnh hưởng trực tiếp đến độ cứng của hệ, và mạng phải có khả năng phân biệt thông tin này.

### 4.3. Vấn đề mạng sâu và vai trò của Batch Normalization

Một câu hỏi thường nảy sinh khi xem xét kiến trúc paper là vì sao chọn 32 lớp khi đồ thị chỉ có khoảng 16–30 đỉnh. Về lý thuyết, sau khoảng 5–10 lớp, mỗi đỉnh đã có thể "thấy" toàn bộ đồ thị thông qua message passing. Lý do chọn số lớp lớn hơn nhiều có thể được lý giải qua hai khía cạnh. Thứ nhất, độ sâu của mạng không chỉ liên quan đến phạm vi tiếp nhận (receptive field) mà còn đến năng lực biểu diễn (representation capacity). Mỗi lớp thêm vào mạng cho phép biểu diễn các quan hệ phi tuyến phức tạp hơn giữa hình học và ứng suất, và với bài toán có sự thay đổi mạnh về điều kiện biên, vật liệu phi tuyến và hình học, năng lực biểu diễn cao là cần thiết. Thứ hai, mạng sâu tạo cơ hội học các đặc trưng có tính tổ hợp — tức là biểu diễn các tương tác bậc cao giữa nhiều thành phần — điều mà mạng nông không thể đạt được.

Tuy nhiên, mạng GNN sâu thường gặp phải vấn đề **over-smoothing** — sau quá nhiều lớp, các vector trạng thái của các đỉnh khác nhau hội tụ về cùng một giá trị, làm mất khả năng phân biệt giữa chúng. Để chống vấn đề này, paper áp dụng **Batch Normalization** sau mỗi lớp GraphSAGE. Batch normalization chuẩn hóa các vector trạng thái có mean 0 và variance 1 trong mỗi mini-batch, qua đó duy trì sự đa dạng giữa các đỉnh và ổn định gradient trong huấn luyện. Hơn nữa, công thức cộng $\mathbf{h}_v + \sum \mathbf{h}_u$ trong GraphSAGE đóng vai trò gần như một liên kết phần dư (residual connection), giúp gradient không bị triệt tiêu khi lan truyền ngược qua nhiều lớp. Sự kết hợp giữa batch normalization và cấu trúc cộng này là chìa khóa cho phép mạng 32 lớp huấn luyện ổn định.

### 4.4. Hàm mất mát và ý nghĩa kỹ thuật

Hàm mất mát được sử dụng là Trung bình Bình phương Sai số (Mean Squared Error — MSE):

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^n (\mathbf{y}_i - \hat{\mathbf{y}}_i)^2,
$$

trong đó $\mathbf{y}_i$ và $\hat{\mathbf{y}}_i$ lần lượt là giá trị ứng suất thực (từ FEM) và giá trị dự đoán (từ GNN) tại điểm dữ liệu thứ $i$. Lựa chọn MSE thay vì các hàm thay thế như MAE (Mean Absolute Error) hay Huber không phải là quy ước mặc định mà phản ánh ưu tiên kỹ thuật cụ thể. MSE phạt mạnh các sai số lớn — bình phương sai số 10 lớn gấp 100 lần bình phương sai số 1 — qua đó hướng mạng tới việc dự đoán chính xác các điểm có ứng suất cao. Vì trong thiết kế kết cấu, các vùng tập trung ứng suất cao chính là nơi xảy ra phá hủy, ưu tiên này là hợp lý: thà sai một chút ở vùng ứng suất thấp (không nguy hiểm) còn hơn sai ở vùng ứng suất cực đại (quyết định độ bền). Đây là minh chứng cho nguyên tắc rằng việc chọn hàm mất mát luôn nên xuất phát từ mục tiêu kỹ thuật, chứ không phải từ thói quen.

### 4.5. Bộ siêu tham số đầy đủ

Để hỗ trợ tái hiện nghiên cứu, dưới đây là bộ siêu tham số được sử dụng trong toàn bộ thí nghiệm:

| Siêu tham số | Giá trị |
|---|---|
| Số lớp ẩn | 32 |
| Số nơ-ron mỗi lớp ẩn | 64 |
| Hàm kích hoạt | $\tanh$ |
| Optimizer | Adam |
| Learning rate | 0.02 |
| Batch size | 512 |
| Hệ số chính quy hóa L2 | $10^{-4}$ |
| Hàm gộp | sum |
| Hàm mất mát | MSE |
| Framework | PyTorch Geometric |
| Phần cứng | NVIDIA GTX 3090 |

Bộ siêu tham số này được xác định thông qua tìm kiếm gần ngẫu nhiên (quasi-random search) và áp dụng nhất quán cho toàn bộ các trường hợp nghiên cứu. Learning rate 0.02 tương đối cao nhưng được cân bằng bởi batch size lớn 512, theo nguyên tắc Linear Scaling Rule: khi batch size tăng, learning rate có thể tăng tỷ lệ thuận để giữ động lực gradient ổn định. Khi tái hiện nghiên cứu trên phần cứng yếu hơn (vd GPU 4 GB không thể chứa batch 512), người dùng nên giảm batch size đồng thời giảm learning rate theo tỷ lệ tương ứng.

---

## 5. Chuẩn bị dữ liệu

Phần dữ liệu là phần ít được paper trình bày kỹ nhất nhưng lại chiếm phần lớn công sức khi tái hiện thực tế. Phần này trình bày chi tiết cách dữ liệu được sinh, các biến thiết kế được lấy mẫu, cách xuất ứng suất từ FEM và cách biểu diễn nó dưới dạng phù hợp cho GNN.

### 5.1. Sinh dữ liệu bằng FEM trong ABAQUS

Mọi dữ liệu huấn luyện trong nghiên cứu này đều được sinh ra bằng phần mềm FEM thương mại ABAQUS, sử dụng phần tử shell tứ giác giảm tích phân `S4R` — phần tử tiêu chuẩn cho phân tích tấm-vỏ chịu uốn. Mesh được chọn dày: 15 phần tử nằm giữa hai gân kề nhau, 15 phần tử dọc theo chiều cao của web, và 8 phần tử dọc theo chiều rộng của flange, với điều kiện các phần tử cố gắng giữ dạng vuông hoặc gần vuông để tránh sai số do méo lưới. Tính hội tụ của mesh này được khẳng định qua phân tích nhạy cảm trong Phụ lục B của paper, cho thấy kết quả ứng suất ổn định và sai khác không đáng kể khi tăng độ mịn lên gấp đôi. Mô hình FEM cũng được kiểm chứng bằng cách so sánh với một nghiên cứu trước đây (Avi và cộng sự, 2015), với độ chệch nhỏ trong cả độ võng và phân bố ứng suất.

Dữ liệu được tổ chức thành sáu trường hợp nghiên cứu (case studies), mỗi trường hợp tập trung vào một loại biến thiết kế: trường hợp 1 thay đổi điều kiện biên, trường hợp 2 đưa vào vật liệu phi tuyến, trường hợp 3 cho phép gân không đồng đều (chiều cao, độ dày khác nhau giữa các gân), trường hợp 4 thay tải đều bằng tải patch (tải tập trung trên một vùng nhỏ), trường hợp 5 kết hợp toàn bộ các biến trên cho panel thẳng, và trường hợp 6 mở rộng sang panel cong với hệ gân hai chiều. Số mẫu sinh ra cho mỗi trường hợp dao động từ 2.000 đến 8.000 — trường hợp 5 cần nhiều dữ liệu nhất do không gian thiết kế lớn nhất. Dữ liệu được chia theo tỷ lệ 80% cho huấn luyện, 10% cho kiểm chứng (validation), và 10% cho kiểm tra (test).

### 5.2. Các biến đầu vào và miền lấy mẫu

Việc lấy mẫu được thực hiện theo phân phối đều (uniform random) trong các miền tham số đã được xác định trước. Bảng dưới đây tóm tắt các biến và miền giá trị tương ứng:

| Biến | Cận dưới | Cận trên | Đơn vị |
|---|---|---|---|
| Độ dày tấm | 10 | 20 | mm |
| Độ dày web gân | 5 | 20 | mm |
| Chiều cao web gân | 100 | 400 | mm |
| Độ dày flange | 5 | 20 | mm |
| Chiều rộng flange | 50 | 150 | mm |
| Số gân dọc | 2 | 8 | – |
| Số gân ngang (chỉ với panel cong) | 0 | 3 | – |
| Độ cong (Gaussian curvature) | 0.005 | 0.015 | $1/\text{m}^2$ |

Kích thước tổng thể của mỗi panel được giữ cố định 3 m × 3 m, đại diện cho một đơn vị lặp lại tiêu biểu trong các kết cấu lớn. Đối với panel cong, độ võng ban đầu tối đa lên tới 139 mm. Điều kiện biên được mã hóa dưới dạng phân loại: trên các cạnh của tấm có thể là *fixed* (ngàm) hoặc *simply supported* (gối tựa đơn giản); trên các cạnh của web và flange gân có thể là *free* (tự do), *simply supported* hoặc *fixed*. Trong các trường hợp 1 và 5, điều kiện biên được sinh ngẫu nhiên giữa các thiết kế; trong các trường hợp còn lại, điều kiện biên được giữ cố định *fixed* để cô lập ảnh hưởng của các biến khác.

Vật liệu được sử dụng là thép với mô-đun đàn hồi 200 GPa và hệ số Poisson 0.3. Trong các trường hợp có vật liệu phi tuyến (trường hợp 2 và 5), mô hình đàn-dẻo được áp dụng với ứng suất chảy ban đầu $\sigma_0 = 355$ MPa và quy luật làm cứng có dạng:

$$
\sigma_f(\varepsilon) = \begin{cases}
\sigma_0 & \text{nếu } \varepsilon \le \varepsilon_L, \\
K(\varepsilon_0 + \varepsilon)^n & \text{nếu } \varepsilon > \varepsilon_L,
\end{cases}
$$

trong đó $\varepsilon_L = 0.006$ là biến dạng tới hạn, $K = 530$ MPa và $n = 0.26$ là các tham số làm cứng, và $\varepsilon_0 = (\sigma_0/K)^{1/n} - \varepsilon_L$. Tải trọng dạng đều có biên độ trong khoảng 0.05–0.1 MPa cho các trường hợp đàn hồi thông thường, tăng lên 0.1–0.2 MPa cho trường hợp 2 (để buộc nhiều panel rơi vào miền chảy dẻo), và 0.2–0.3 MPa cho các trường hợp có tải patch. Vùng patch chiếm $\frac{1}{4} \times \frac{1}{4}$ diện tích tấm chính, tức 0.75 m × 0.75 m, và vị trí của nó được sinh ngẫu nhiên trên tấm.

### 5.3. Biểu diễn đầu ra: từ trường ứng suất sang vector

Đầu ra mong muốn của mạng là trường ứng suất von Mises trên toàn bộ panel. Tuy nhiên, mạng GNN gán mỗi đỉnh một vector trạng thái có kích thước cố định, vì vậy cần một cơ chế biểu diễn trường ứng suất hai chiều dưới dạng vector. Cách tiếp cận trong paper là, với mỗi đơn vị cấu trúc, chọn một bề mặt đại diện và lấy mẫu ứng suất tại các điểm phân bố đều trên bề mặt đó. Cụ thể, đối với tấm và flange, bề mặt được chọn là bề mặt có giá trị tọa độ z thấp hơn (đối với tấm, đây là phía chịu tải); đối với web, bề mặt được chọn dựa trên tọa độ y cao hơn hoặc x thấp hơn, tùy theo định hướng của web. Quy ước này cần được tuân thủ nhất quán để đảm bảo dữ liệu giữa các mẫu so sánh được với nhau.

Vì các đơn vị cấu trúc có hình dạng và kích thước khác nhau giữa các panel, ứng suất từ mesh FEM được nội suy về một lưới chuẩn $10 \times 20$: 10 điểm dọc theo chiều ngắn (chiều rộng) và 20 điểm dọc theo chiều dài. Việc nội suy được thực hiện bằng phương pháp scattered interpolation, với hướng của lưới được xác định bởi sự kề nhau cục bộ của đơn vị đang xét với các hàng xóm của nó. Sau khi thu được lưới $10 \times 20$, ma trận hai chiều này được làm phẳng (reshape) thành vector 200 chiều, đóng vai trò là nhãn đầu ra cho đỉnh tương ứng trong đồ thị.

Sự cần thiết của việc reshape xuất phát từ ràng buộc cấu trúc của GNN: mỗi đỉnh chỉ chấp nhận vector làm trạng thái, không chấp nhận trực tiếp tensor hai chiều. Do đó, để biểu diễn được bản chất hai chiều của trường ứng suất, paper áp dụng nguyên lý "encode rồi decode": trong huấn luyện, vector 200 chiều được làm phẳng từ lưới gốc; trong dự đoán, vector đầu ra được reshape ngược về lưới $10 \times 20$ và ráp lại thành contour ba chiều cho toàn panel. Việc lựa chọn kích thước lưới $10 \times 20$ là kết quả của một sự cân bằng: lưới thưa hơn làm mất chi tiết tại các vùng tập trung ứng suất, trong khi lưới dày hơn tăng số chiều đầu ra mà không cải thiện độ chính xác tương xứng. Phụ lục B của paper trình bày thí nghiệm so sánh các kích thước lưới và xác nhận rằng $10 \times 20$ là lựa chọn tối ưu cho miền bài toán đang xét.

### 5.4. Quy mô dữ liệu và chiến lược chia tập

Nghiên cứu phân tích ảnh hưởng của quy mô dữ liệu đến độ chính xác. Đối với các trường hợp 1, 2, 3, 4 và 6, 2.000 mẫu là đủ để đạt độ chính xác cao. Tuy nhiên, đối với trường hợp 5 — kết hợp toàn bộ các biến — cần đến 8.000 mẫu để mạng học được mapping phức tạp; với 1.000 hoặc 2.000 mẫu, độ chính xác giảm đáng kể, đặc biệt tại các vùng có hình học phức tạp (xem Phụ lục C của paper). Đây là minh chứng cho nguyên tắc rằng số mẫu cần thiết tăng tỷ lệ thuận (và đôi khi siêu tuyến tính) với độ phức tạp của không gian thiết kế. Tỷ lệ chia 80% / 10% / 10% được áp dụng nhất quán cho mọi trường hợp; điểm cần lưu ý là việc chia phải được thực hiện ở cấp *thiết kế* (mỗi panel là một mẫu độc lập) chứ không phải ở cấp đỉnh, nhằm tránh rò rỉ dữ liệu giữa tập huấn luyện và tập kiểm tra.

---

## 6. Quy trình huấn luyện và pipeline tổng thể

Để hiểu được toàn cảnh nghiên cứu và có thể triển khai lại, cần hình dung pipeline như một chuỗi sáu bước liên kết, trong đó đầu ra của bước này là đầu vào của bước kế tiếp. Phần này mô tả từng bước với phân tích vai trò và sự kết nối logic.

Bước thứ nhất, **sinh dữ liệu thô bằng FEM**, là nền móng của toàn bộ pipeline. Trong bước này, một script Python tự động hóa sinh hàng nghìn cấu hình thiết kế ngẫu nhiên trong miền tham số đã định, mỗi cấu hình tương ứng với một input file ABAQUS. ABAQUS sau đó chạy phân tích tĩnh cho từng cấu hình và xuất ra trường ứng suất von Mises trên toàn panel. Vai trò của bước này là tạo "ground truth" mà mạng sẽ học bắt chước; nếu bỏ qua bước này, mạng không có chuẩn để so sánh và do đó không thể được huấn luyện. Đây cũng là bước tiêu tốn nhiều tài nguyên nhất — paper đã sử dụng cluster tính toán hiệu năng cao của Đại học British Columbia để hoàn thành trong thời gian hợp lý.

Bước thứ hai, **chuyển kết cấu thành đồ thị**, biến đổi dữ liệu hình học thô thành cấu trúc đồ thị mà GNN có thể xử lý. Một module phân tích hình học sẽ phát hiện các đơn vị cấu trúc (tấm con, web, flange) và xác định các mối hàn giữa chúng. Mỗi đơn vị trở thành một đỉnh, mỗi mối hàn trở thành một cạnh, và ma trận kề được xây dựng tương ứng. Vai trò của bước này là cung cấp "ngôn ngữ" mà GNN có thể hiểu — cụ thể là tô-pô của kết cấu. Nếu bước này thực hiện sai (ví dụ thiếu cạnh giữa hai đơn vị thực sự được hàn), thông tin sẽ không lan truyền đúng cách trong mạng, dẫn đến dự đoán ứng suất sai chỗ.

Bước thứ ba, **xây dựng đặc trưng đỉnh**, tính toán vector tám chiều cho mỗi đỉnh dựa trên hình học và điều kiện biên của đơn vị tương ứng, cộng thêm áp suất tải trọng. Trong các trường hợp có tải patch, tọa độ vị trí lực được thêm vào. Bước này cũng bao gồm việc chuẩn hóa các đặc trưng — thường là Min-Max scaling — để mọi đặc trưng có cùng thang đo, tránh tình trạng những đặc trưng có biên độ lớn (như chiều dài) lấn át những đặc trưng có biên độ nhỏ nhưng quan trọng (như áp suất). Vai trò của bước này là cung cấp "nội dung" mà mạng sẽ xử lý; nếu thiếu một đặc trưng quan trọng (chẳng hạn điều kiện biên), mạng không thể dự đoán đúng khi điều kiện biên thay đổi.

Bước thứ tư, **xây dựng nhãn đầu ra**, xuất trường ứng suất từ kết quả ABAQUS, nội suy về lưới $10 \times 20$ cho mỗi đơn vị, và làm phẳng thành vector 200 chiều. Đây là bước "đáp án" mà mạng sẽ học bắt chước. Cách nội suy quyết định chất lượng của nhãn — nội suy quá thô sẽ làm mất chi tiết tại các điểm tập trung ứng suất và do đó hạn chế độ chính xác cuối cùng của mạng. Sự kết hợp giữa các bước thứ ba và thứ tư tạo nên dữ liệu huấn luyện hoàn chỉnh: mỗi mẫu là một đồ thị có đặc trưng đỉnh và nhãn đỉnh.

Bước thứ năm, **huấn luyện GNN**, là trung tâm của pipeline. Tại bước này, dữ liệu được chia thành tập huấn luyện, kiểm chứng và kiểm tra; mạng GraphSAGE 32 lớp được khởi tạo; vòng lặp huấn luyện chạy qua nhiều epoch, trong mỗi epoch các đồ thị được đưa qua mạng theo từng mini-batch, mất mát MSE được tính, gradient được lan truyền ngược, và optimizer Adam cập nhật trọng số. Quá trình lặp tiếp tục cho đến khi mất mát kiểm chứng không cải thiện trong nhiều epoch liên tiếp, biểu thị mạng đã hội tụ. Thời gian huấn luyện điển hình là 30–40 phút cho hầu hết các trường hợp, kéo dài đến 120 phút cho trường hợp 5 do quy mô dữ liệu lớn.

Bước thứ sáu, **dự đoán cho thiết kế mới**, là bước thu hoạch giá trị thực tế. Khi đã có mạng được huấn luyện, người dùng có thể đưa vào một panel thiết kế mới, chuyển nó thành đồ thị bằng cùng pipeline ở bước hai và ba, rồi chạy forward pass qua mạng để thu được các vector đầu ra 200 chiều cho mỗi đỉnh. Các vector này được reshape về lưới $10 \times 20$ và ráp lại thành contour ứng suất ba chiều trên toàn panel. Quá trình này gần như tức thời — chỉ vài mili-giây — so với hàng phút đến hàng giờ của một lần chạy ABAQUS đầy đủ. Đây là điểm mà mô hình bậc rút gọn thể hiện giá trị: chi phí huấn luyện một lần được bù đắp bởi vô số lần dự đoán nhanh sau đó.

---

## 7. Kết quả và phân tích

Phần này trình bày kết quả thực nghiệm trên sáu trường hợp nghiên cứu, so sánh với FEM, và phân tích các điểm mạnh-yếu của mô hình.

### 7.1. Kết quả tổng quan

Trên toàn bộ tập kiểm tra của sáu trường hợp, mô hình GraphSAGE với biểu diễn đồ thị đề xuất đạt độ chính xác trung bình **92.3%** trong dự đoán ứng suất von Mises cực đại. Đây là một kết quả đáng kể nếu xét đến phạm vi rộng của các biến thiết kế — từ điều kiện biên đa dạng, vật liệu đàn hồi và phi tuyến, hình học đơn giản và phức tạp, tải đều và tải patch, panel thẳng và panel cong với hệ gân một và hai chiều. Kết quả chi tiết theo từng trường hợp cho thấy độ chính xác đạt 94.93% trong trường hợp 1 (điều kiện biên thay đổi), 96.31% trong trường hợp 2 (vật liệu phi tuyến), 88.01% trong trường hợp 3 (hình học phức tạp), 89.69% trong trường hợp 5 (tổng hợp), và 93.93% trong trường hợp 6 (panel cong, ví dụ test 12).

### 7.2. So sánh với FEM trên các trường hợp đại diện

Trong trường hợp 1, hai test example được phân tích sâu — một panel với điều kiện biên đơn giản tại các cạnh tấm và tự do tại các cạnh gân, và một panel với điều kiện biên ngàm tại tấm và đơn giản tại gân. Trong cả hai, GraphSAGE bám sát phân bố ứng suất của FEM ở các vị trí chịu ứng suất cao, đặc biệt tại giao tuyến giữa tấm và web gân — nơi ứng suất cực đại thường xuất hiện. Sai khác giữa GNN và FEM tại điểm cực đại lần lượt là 5.98% và 0.93%. Tuy nhiên, ở các vùng ứng suất thấp như giữa tấm trong test example 1, mô hình có sai số tương đối lớn hơn — một hiện tượng nhất quán xuất hiện trong toàn bộ nghiên cứu và có lý giải kỹ thuật rõ ràng (xem Mục 7.3).

Trong trường hợp 2, mô hình tiếp tục cho kết quả tốt với panel chịu tải lớn dẫn đến chảy dẻo. Ứng suất cực đại tại giao tuyến tấm-gân được dự đoán với sai số khoảng 3.2% trong test example 3. Trong test example 4, một panel có gân nhỏ chịu ứng suất cao trong cả web và flange, mô hình bắt đúng phân bố ứng suất chung với sai số tối thiểu 6.36% tại web. Đáng chú ý, chỉ 15% dữ liệu huấn luyện trong trường hợp này thực sự rơi vào miền chảy dẻo, điều này đặt ra thách thức về cân bằng dữ liệu, nhưng mạng vẫn xử lý hợp lý phần lớn các test example.

Trong trường hợp 3, khi gân không đồng đều — chiều cao và độ dày khác nhau giữa các gân — mô hình đối mặt với không gian thiết kế lớn hơn và độ chính xác giảm xuống 88%. Sai số lớn nhất xuất hiện tại các cạnh tấm trong những cấu hình hình học bất đối xứng, nơi trường ứng suất nhảy bậc đáng kể giữa các đơn vị có kích thước khác nhau.

Đối với panel cong với hệ gân hai chiều (trường hợp 6), test example 12 cho thấy mô hình bắt được các pattern ứng suất tại cả tấm và gân, với sai khác trung bình tuyệt đối toàn panel là 3.45 MPa và sai số tại cực đại là 6.07%. Đây là một kết quả ấn tượng vì panel cong đại diện cho các kết cấu thực tế phức tạp như phần mũi tàu hoặc hông tàu, nơi hình học và phân bố ứng suất phi tuyến hơn nhiều so với panel phẳng.

### 7.3. Khi mô hình hoạt động tốt và khi có sai số

Phân tích chéo các trường hợp cho thấy mô hình đặc biệt mạnh ở việc dự đoán **ứng suất cực đại** — chính là đại lượng quan trọng nhất trong thiết kế kết cấu. Lý do nằm ở lựa chọn hàm mất mát MSE, vốn phạt mạnh sai số lớn và do đó khuyến khích mạng tập trung độ chính xác vào các vùng có ứng suất cao. Đây là một thiết kế có chủ đích: trong tối ưu hóa kết cấu, độ bền được quyết định bởi ứng suất cực đại, và một mô hình thay thế dự đoán đúng giá trị này có giá trị thực tế cao hơn nhiều so với một mô hình dự đoán đều cả hai miền.

Đối lập với điểm mạnh trên, mô hình cho sai số tương đối lớn hơn ở các **vùng ứng suất thấp** — chẳng hạn vùng trung tâm tấm trong những panel có nhiều gân lớn. Hiện tượng này có hai nguyên nhân: (i) MSE phạt nhẹ các sai số nhỏ, do đó mạng không học được các pattern tinh tế ở vùng ứng suất thấp; (ii) việc nội suy về lưới $10 \times 20$ và làm phẳng thành vector có thể làm mất chi tiết tại những vùng có gradient ứng suất nhỏ. Tuy nhiên, vì ứng suất thấp ít liên quan đến phá hủy, hạn chế này chấp nhận được trong bối cảnh thiết kế.

Một điểm cần lưu ý nữa là mô hình bị giới hạn bởi **phạm vi không gian thiết kế trong huấn luyện**. Khi đầu vào nằm ngoài miền lấy mẫu — chẳng hạn một panel có chiều cao gân vượt quá 400 mm hoặc tải trọng vượt 0.3 MPa — không có đảm bảo nào về độ chính xác. Đây là tính chất chung của các mô hình học máy: chúng nội suy tốt trong miền huấn luyện nhưng ngoại suy kém. Trong thực tế triển khai, người dùng cần đảm bảo các thiết kế cần đánh giá nằm trong miền đã huấn luyện, hoặc mở rộng tập huấn luyện khi miền thiết kế thay đổi.

---

## 8. Phân tích hiệu quả tính toán

Một trong những đóng góp thực tế quan trọng nhất của nghiên cứu là việc giảm đáng kể nhu cầu tài nguyên tính toán so với cách biểu diễn đồ thị truyền thống. Để hiểu được nguồn gốc của hiệu quả này, cần phân tích chi phí tính toán theo cấu trúc đồ thị.

### 8.1. Liên hệ với số đỉnh trong đồ thị

Như đã chứng minh trong Mục 3.4, độ phức tạp tính toán của một mạng GNN trong huấn luyện tỷ lệ tuyến tính với số đỉnh $N$. Cụ thể, tổng chi phí gồm hai thành phần: chi phí biến đổi đặc trưng tại mỗi đỉnh — bằng $\mathcal{O}(N \cdot \sum_i F_{i-1} F_i)$ — và chi phí gộp thông tin từ hàng xóm — bằng $\mathcal{O}(N \cdot k \cdot \sum_i F_i)$. Vì $\sum_i F_{i-1} F_i$ và $\sum_i F_i$ là hằng số một khi kiến trúc mạng cố định, và $k$ (số hàng xóm trung bình) cũng tương đối ổn định giữa các kích thước đồ thị khác nhau trong cùng loại bài toán, toàn bộ chi phí phụ thuộc tuyến tính vào $N$. Đây là hệ quả cơ bản: giảm $N$ là cách trực tiếp nhất để giảm chi phí.

Trong cách biểu diễn truyền thống dựa trên phần tử hữu hạn, $N$ tỷ lệ thuận với độ mịn mesh — và mesh càng dày để có độ chính xác FEM cao thì $N$ càng lớn. Trong nghiên cứu này, mesh được chọn dày để phục vụ ground truth (15 phần tử giữa hai gân), dẫn đến số đỉnh ở cách truyền thống lên đến hơn mười nghìn cho một panel điển hình. Ngược lại, cách biểu diễn theo đơn vị cấu trúc giữ $N$ ở mức 16–30 cho mọi panel — số này phụ thuộc vào số gân chứ không phụ thuộc vào độ mịn mesh. Tỷ lệ giảm $N$ do đó là khoảng 600 lần về lý thuyết.

### 8.2. Tại sao GPU memory giảm tới 98%

Bộ nhớ GPU cần thiết cho huấn luyện chủ yếu được sử dụng cho ba mục đích: lưu trữ ma trận đặc trưng đỉnh trong mỗi mini-batch, lưu trữ các trung gian (intermediate activations) qua các lớp để lan truyền ngược, và lưu trữ ma trận kề. Tất cả đều tỷ lệ thuận với $N$ — và một số thậm chí với $N^2$ nếu ma trận kề được lưu dày. Khi $N$ giảm 600 lần, bộ nhớ giảm theo tỷ lệ tương ứng. Số đo thực tế trong paper xác nhận điều này: cách truyền thống tiêu tốn 23.4 GB VRAM cho batch size 64, trong khi cách đề xuất chỉ cần 0.5 GB, tương ứng giảm khoảng 98%. Hệ quả thực tiễn của con số này là: cách truyền thống yêu cầu GPU cấp doanh nghiệp với 24 GB VRAM trở lên, còn cách đề xuất chạy thoải mái trên một GPU thông thường có 4 GB.

### 8.3. Tại sao thời gian huấn luyện giảm tới 96%

Tương tự, thời gian huấn luyện giảm tỷ lệ với $N$. Cụ thể, paper báo cáo rằng cách truyền thống tốn 6.94 giây mỗi epoch trong khi cách đề xuất chỉ tốn 0.26 giây — nhanh hơn khoảng 27 lần. Lưu ý rằng tỷ lệ này thấp hơn so với tỷ lệ giảm $N$ lý thuyết (khoảng 600 lần), phản ánh thực tế rằng còn nhiều chi phí phụ trợ không phụ thuộc trực tiếp vào $N$ — chẳng hạn chi phí khởi tạo batch, chi phí gọi hàm trong PyTorch Geometric, chi phí thao tác bộ nhớ giữa CPU và GPU. Nhưng ngay cả mức 27 lần nhanh hơn cũng đã đủ để chuyển một bài toán không khả thi (cần hàng giờ hoặc hàng ngày huấn luyện) thành bài toán hoàn tất trong vài phút đến vài chục phút.

### 8.4. Ý nghĩa thực tiễn

Sự giảm đồng thời của cả bộ nhớ và thời gian không chỉ là thuận tiện — nó mở ra khả năng nghiên cứu trên phần cứng phổ thông. Một sinh viên với GPU tiêu dùng có thể huấn luyện và thử nghiệm các mô hình GNN cho stiffened panel mà không cần truy cập cluster tính toán cao cấp. Đối với các kỹ sư công nghiệp, điều này có nghĩa là chu trình thiết kế-thử-tối ưu-thiết kế lại có thể chạy trong cùng ngày, thay vì mất tuần. Nhìn xa hơn, cùng triết lý "abstraction đúng level" có thể được áp dụng cho các loại kết cấu khác — cánh máy bay, khung xe, vỏ tên lửa — nơi cấu trúc rời rạc tự nhiên cũng có thể được biểu diễn hiệu quả qua đồ thị đơn vị cấu trúc.

---

## 9. Kết luận

Nghiên cứu này trình bày một mô hình bậc rút gọn dựa trên Mạng nơ-ron đồ thị — cụ thể là kiến trúc GraphSAGE — cho bài toán dự đoán trường ứng suất trong tấm có gân gia cường. Đóng góp cốt lõi không nằm ở việc áp dụng GNN cho kết cấu (vốn đã có một số nghiên cứu trước cho hệ giàn), mà nằm ở **phương pháp biểu diễn đồ thị mới theo đơn vị cấu trúc** — gán mỗi tấm con, web gân và flange gân thành một đỉnh duy nhất, thay vì gán mỗi phần tử hữu hạn thành một đỉnh như cách truyền thống. Cách tiếp cận này phản ánh chính xác bản chất rời rạc của kết cấu vỏ mỏng: mỗi đơn vị cấu trúc là một thành phần được sản xuất riêng và hàn lại với nhau, do đó tự thân là một đối tượng có ý nghĩa vật lý độc lập. Việc tôn trọng cấu trúc này trong biểu diễn đồ thị mang lại lợi ích kép: số đỉnh giảm vài trăm lần, kéo theo bộ nhớ GPU giảm khoảng 98% và thời gian huấn luyện giảm khoảng 96% so với cách truyền thống, mà độ chính xác vẫn duy trì ở mức trung bình 92.3% cho dự đoán ứng suất cực đại.

Phương pháp được kiểm chứng qua sáu trường hợp nghiên cứu phủ đầy đủ các loại biến thiết kế quan trọng: điều kiện biên, vật liệu phi tuyến, hình học phức tạp, tải dạng patch, và panel cong với hệ gân hai chiều. Trong mọi trường hợp, mô hình bám sát kết quả FEM tại các vùng ứng suất cao — chính là vùng quyết định độ bền kết cấu — và đặc biệt mạnh trong dự đoán ứng suất cực đại nhờ lựa chọn hàm mất mát MSE. Mô hình có sai số tương đối lớn hơn ở các vùng ứng suất thấp, đây là hệ quả có thể chấp nhận của cùng lựa chọn thiết kế và không ảnh hưởng đến giá trị thực tế trong tối ưu kết cấu.

Trong bối cảnh rộng hơn của cơ học kết cấu, nghiên cứu này có ý nghĩa kép. Về mặt kỹ thuật, nó cung cấp một mô hình bậc rút gọn đủ nhanh và đủ chính xác để dùng trong vòng tối ưu hóa với hàng ngàn đến hàng vạn đánh giá — điều mà FEM thuần túy không thể đáp ứng do chi phí tính toán. Về mặt phương pháp luận, nó minh họa một nguyên tắc tổng quát: *mức độ trừu tượng đúng đắn luôn vượt trội so với độ phân giải cao*. Trong thiết kế kiến trúc mạng cho dữ liệu vật lý, việc xác định đúng "đơn vị có ý nghĩa" — và biểu diễn nó như một đỉnh đồ thị — quan trọng hơn việc nhồi nhét nhiều dữ liệu thô. Nguyên tắc này có thể được mở rộng cho các loại kết cấu khác — cánh máy bay với spar và rib, khung gầm xe, hệ thống ống chịu áp — và mở ra hướng nghiên cứu phong phú trong tương lai.

Một số hướng phát triển tự nhiên có thể nêu ra để hoàn thiện và mở rộng nghiên cứu. Thứ nhất, tích hợp ràng buộc vật lý (chẳng hạn cân bằng lực cục bộ) vào hàm mất mát có thể cải thiện độ chính xác và tính tin cậy của mô hình ở các vùng ngoại suy. Thứ hai, mở rộng sang các loại tiết diện gân khác (chữ L, chữ I, chữ U) đòi hỏi tổng quát hóa sơ đồ biểu diễn đỉnh và đòi hỏi nghiên cứu thêm về encoding cross-section topology. Thứ ba, tích hợp định lượng độ không chắc chắn (uncertainty quantification) — chẳng hạn qua Bayesian GNN hoặc ensemble — sẽ cho phép mô hình không chỉ dự đoán mà còn cảnh báo khi đầu vào ra khỏi miền tin cậy. Thứ tư, áp dụng học chủ động (active learning) có thể giảm số mẫu cần thiết bằng cách tập trung sinh dữ liệu tại những vùng mô hình ít chắc chắn. Cuối cùng, kết hợp với học chuyển giao (transfer learning) giữa các loại panel khác nhau có thể giảm chi phí huấn luyện cho các bài toán mới. Những hướng này không phủ định nghiên cứu hiện tại mà bổ sung cho nó, đưa GNN từ một công cụ nghiên cứu hứa hẹn thành một công cụ kỹ thuật trưởng thành cho cơ học kết cấu.

---

## Tài liệu tham khảo chính

Báo cáo này dựa trên paper gốc:

> Cai, Y., & Jelovica, J. (2024). Efficient graph representation in graph neural networks for stress predictions in stiffened panels. *Thin-Walled Structures*, 203, 112157. https://doi.org/10.1016/j.tws.2024.112157

Các tài liệu nền tảng quan trọng được tham chiếu bao gồm:

- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS — bài báo gốc về GraphSAGE.
- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How Powerful Are Graph Neural Networks?*. ICLR — phân tích lý thuyết về sức mạnh biểu diễn của các hàm gộp.
- Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. W. (2020). *Learning Mesh-Based Simulation with Graph Networks*. ICLR — đại diện của cách biểu diễn FE-vertex trong CFD.
- Avi, E., Lillemäe, I., Romanoff, J., & Niemelä, A. (2015). Equivalent shell element for ship structural design. *Ships and Offshore Structures*, 10(3), 239–255 — nghiên cứu được dùng để kiểm chứng mô hình FEM trong paper.

---

*Báo cáo này được biên soạn nhằm mục đích nghiên cứu và học tập, trình bày phương pháp luận đầy đủ để người đọc có thể mô tả lại nghiên cứu, thiết kế lại pipeline và triển khai lại bằng mã nguồn. Phần triển khai mã nguồn cụ thể, pseudo-code và phân tích thực tiễn được trình bày trong tài liệu hướng dẫn đi kèm "GNN_Stiffened_Panels_Mentor_Guide.md".*
