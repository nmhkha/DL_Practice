# Dự báo Giá Cổ phiếu (Stock Price Forecasting)
    Dự án này trình bày một mô hình học sâu để dự báo giá cổ phiếu bằng cách sử dụng Mạng thần kinh Hồi quy Dài-Ngắn hạn (LSTM).
    - Mục tiêu (Goal)
        Mục tiêu chính là dự đoán giá đóng cửa (Close) của cổ phiếu vào ngày hôm sau (next-day) dựa trên chuỗi dữ liệu giá trong 60 ngày trước đó.
    - Dữ liệu (Dataset)
        Nguồn: Dữ liệu được tải tự động từ Yahoo Finance.
        Mã cổ phiếu: GOOGL (Alphabet Inc.).
        Khung thời gian: 2 năm (period='2y').
        Đặc trưng (Feature): Chỉ sử dụng giá đóng cửa (Close) cho việc dự báo.
    - Mô hình (Model)
        Loại mô hình: LSTM (Long Short-Term Memory).
        Loại tác vụ: Hồi quy (Regression).
        Chi tiết: Mô hình sử dụng dữ liệu của 60 ngày liên tiếp (SEQ_LEN = 60) để dự đoán giá của ngày thứ 61.
# Thư viện sử dụng 
- torch (PyTorch) : dùng để xây dựn mô hình Deep Learning LSTM.
- yfinance (để tải dữ liệu): tải dữ liệu cổ phiếu.
- pandas: dùng nó để đọc, ghi, và thao tác với các DataFrame.
- numpy: giúp làm việc với mảng (array) và ma trận tốc độ cao.
- scikit-learn (sử dụng MinMaxScaler và các độ đo metrics): Dùng để đo lường hiệu suất, cụ thể là đo xem mô hình của bạn dự đoán "tệ" đến mức nào (tính toán lỗi).
- matplotlib (để vẽ biểu đồ): tạo ra các biểu đồ và đồ thị (như biểu đồ Loss và biểu đồ dự đoán).

# Tải dữ liệu
- Tai toan bo du lieu tu 5 nam tro lai
    dat = yf.download('GOOGL', period='2y')
- Luu vao file
    dat.to_csv('GOOGL_2y.csv')

Chức năng:
Sử dụng thư viện yfinance để tải dữ liệu lịch sử giá cổ phiếu GOOGL (Alphabet Inc.).
Tham số period='2y' nghĩa là lấy dữ liệu của 2 năm gần nhất
Dữ liệu trả về là một DataFrame của pandas, gồm các cột: Date, Open, High, Low, Close, Adj Close, Volume
Reproducility & Cấu hình
Tại sao phải đảm bảo Reproducibility?

Trong mô hình học máy (đặc biệt là LSTM, RNN), có nhiều thứ ngẫu nhiên:
Thành phần gây biến động kết quả
- Khởi tạo trọng số mô hình | trọng số random → mỗi lần học khác nhau
- Shuffle dữ liệu trong DataLoader | thay đổi cách mô hình học
- Các toán tử trên GPU | một số tính toán không cố định tuyệt đối
- Dropout / data augmentation | tạo random trong quá trình training

➡️ Nếu không cố định seed, bạn chạy hôm nay và chạy lại mai → kết quả khác nhau → không thể tin tưởng hay so sánh.

# Reproducibility: cố định seed để kết quả lặp lại
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

# Cấu hình
    CSV_PATH = "GOOGL_2y.csv"
    SEQ_LEN = 60 # Window length (60 ngay) dùng 60 ngày trước để dự đoán ngày kế tiếp
    BATCH_SIZE = 32
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    LR = 0.001
    EPOCHS = 200
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ",DEVICE)

# Chuẩn hoá và tạo sequence
- Chuẩn bị dữ liệu
    close = df[['Close']].copy()

- Lấy cột Close của giá cổ phiếu (hoặc giá tài sản) để dự báo.
- close là dataframe 1 cột, cần reshape sau này cho MinMaxScale
- Chia tỷ lệ 70/15/15 theo time series
        
        
        #Cutoff index theo ti le 70/15/15 tren time series
        n_total = len(close)
        #Tao sequences, tong samples = n_total - SEQ_LEN
        n_samples = n_total - SEQ_LEN
        train_samples = int(n_samples*0.70)
        val_samples = int(n_samples * 0.15)
        test_samples = n_samples - train_samples - val_samples
        n_total: tổng số ngày.
        n_samples: tổng số sequence bạn có thể tạo với SEQ_LEN.
        Bạn chia train/val/test trước khi tạo sequences, để tránh leakage.
        Fit MinMaxScaler chỉ trên train

        train_raw_end = SEQ_LEN + train_samples #exclusive index for raw array slicing
        print("train_raw end index (exclusive): ", train_raw_end)      


        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(close.values[:train_raw_end]) # fit only on train portion of raw close values
        scaled_all = scaler.transform(close.values) # transform whole series
    Quan trọng: fit scaler chỉ trên train portion, tức là close[:train_raw_end].
    train_raw_end = SEQ_LEN + train_samples vì sequence đầu tiên sử dụng SEQ_LEN ngày đầu tiên.
- Sau đó transform toàn bộ chuỗi, bao gồm val/test. Đây là cách chuẩn để tránh data leakage.

# Tạo sequences (sliding window)

    
    #tao sequences (sliding window)
    X_list, y_list = [], []
    for i in range(SEQ_LEN, len(scaled_all)):
        X_list.append(scaled_all[i-SEQ_LEN:i, 0]) # window of length SEQ_LEN
        y_list.append(scaled_all[i, 0]) # next-day target


    X = np.array(X_list)
    y = np.array(y_list)


    #reshape X -> (n_samples, seq_len, n_features)
    X = X.reshape(X.shape[0], X.shape[1],1)


    print("X shape:", X.shape, "y shape:", y.shape)

- X_list: từng window dài SEQ_LEN.
- y_list: giá kế tiếp (next day).
- Reshape X để có shape (samples, seq_len, n_features) cho LSTM/RNN.
- y là vector (samples,) cho target.

→ Sliding window giúp “cắt” chuỗi dài thành các mẫu nhỏ mà mô hình có thể học được mối quan hệ giữa lịch sử giá và giá ngày tiếp theo.

Output 

    Total days: 502, sequence samples: 442
    Train samples: 309, Val samples: 66, Test samples: 67
    train_raw end index (exclusive):  369
    X shape: (442, 60, 1) y shape: (442,)
- Bạn có 502 ngày giá.
- Với SEQ_LEN = 60, bạn tạo được 442 sequences.
- Chia train/val/test theo 70/15/15 → tương ứng 309/66/67 sequences.
- train_raw_end = 369 dùng để fit scaler mà không rò rỉ thông tin từ validation/test.
- X đã sẵn sàng cho LSTM: (samples, seq_len, features)
- y là target kế tiếp.
# Chia train/val/test
- Xác định cutoff cho train/val/test
    
        sample_cut_train = train_samples
        sample_cut_val = train_samples + val_samples
        sample_cut_train = 309 → số sample dùng cho train.
        sample_cut_val = 309 + 66 = 375 → số sample kết thúc validation.
        Dựa trên các sample đã tạo từ sliding window (X, y).

- Chia dữ liệu
    
        X_train = X[:sample_cut_train]
        y_train =y[:sample_cut_train]


        X_val = X[sample_cut_train:sample_cut_val]
        y_val = y[sample_cut_train:sample_cut_val]


        X_test = X[sample_cut_val:]
        y_test = y[sample_cut_val:]

    - Train: lấy từ sample 0 đến 308 (309 samples)
    - Validation: lấy từ sample 309 đến 374 (66 samples)
    - Test: từ sample 375 đến hết (67 samples)
Output 

    Final splits:
    
        X_train: (309, 60, 1) y_train: (309,)
        X_val:   (66, 60, 1) y_val: (66,)
        X_test:  (67, 60, 1) y_test: (67,)
- Dữ liệu đã được chia theo thứ tự thời gian (time series split), không shuffle.
- Mỗi tập có dạng (samples, sequence length, features) cho LSTM/RNN input.
- Mục đích:
    - X_train/y_train: huấn luyện mô hình.
    - X_val/y_val: điều chỉnh tham số, tránh overfitting.
    - X_test/y_test: đánh giá mô hình trên dữ liệu thực tế chưa thấy.

# Mô hình Deep Learning (LSTM)
Khởi tạo lớp LSTM

def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #Pytorch's built-in LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #Fully connected layer to map hidden state to ouput
        self.fc = nn.Linear(hidden_size, output_size)

→ Xây dựng kiến trúc mô hình (LSTM + fully connected)
input_size: số feature của input, ở đây là 1 (Close price).
hidden_size: số neuron trong LSTM hidden layer.
num_layers: số tầng LSTM xếp chồng (stacked LSTM).
output_size: số output mà mô hình dự đoán, ở đây là 1 (dự đoán giá tiếp theo).
batch_first=True: input có shape (batch_size, seq_len, input_size).

Forward pass
def forward(self, input_seq, hidden):
        # Input shape: (batch_size, seq_len, input_size)
        # Hidden shape: (num_layers, batch_size, hidden_size) - initialized outside
        # Cell state shape: (num_layers, batch_size, hidden_size) - initialized outside


        # out shape: (batch_size, seq_len, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell state shape: (num_layers, batch_size, hidden_size
        if hidden is None:
            batch_size = input_seq.size(0)
            hidden = self.init_hidden(batch_size, input_seq.device)
           
        out, hidden = self.lstm(input_seq, hidden)


        # We often want the output of the last time step for sequence classification
        # out[:, -1, :] has shape (batch_size, hidden_size)\
        output = self.fc(out[:, -1, :])
        return output, hidden

→ Xử lý input sequence, học mối quan hệ theo thời gian, dự đoán output
input_seq: (batch_size, seq_len, input_size) → chuỗi giá của nhiều batch.
hidden: tuple (h0, c0) của hidden state và cell state.
Nếu hidden=None, hàm sẽ khởi tạo bằng 0 cho batch hiện tại.
out: LSTM output cho toàn bộ sequence (batch_size, seq_len, hidden_size).
out[:, -1, :]: lấy output của time step cuối cùng (giá trị dự đoán dựa trên toàn bộ sequence).
self.fc(out[:, -1, :]): map hidden state cuối cùng sang giá trị output (ví dụ giá tiếp theo).
Trả về (output, hidden). Hidden state có thể dùng tiếp cho sequence tiếp theo (stateful LSTM).
Khởi tạo hidden state
def init_hidden(self, batch_size, device):
        # Initialize hidden and cell state with zeros
        # Shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


        return (h0, c0)

→ Khởi tạo trạng thái ban đầu (hidden + cell) để LSTM bắt đầu hoạt động
Hidden state (h0) và cell state (c0) đều khởi tạo bằng 0.
Shape: (num_layers, batch_size, hidden_size) → đúng chuẩn PyTorch LSTM.
device đảm bảo tensor nằm trên CPU/GPU tương thích.

Loss function
criterion = torch.nn.MSELoss()
MSELoss = Mean Squared Error (lỗi bình phương trung bình).
Trong bài toán dự báo giá, MSE thường được dùng để đo sai số giữa giá thực và giá dự đoán.
y_pred càng gần y_true → loss càng nhỏ → mô hình tốt hơn.
Mục đích: Cung cấp hàm mục tiêu để mô hình biết “lỗi của nó là bao nhiêu” và cần điều chỉnh.
Learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

Adam là thuật toán tối ưu hóa gradient hiện đại, hiệu quả cho neural network.
model.parameters() → các tham số của mô hình cần cập nhật (weights, bias).
lr=0.001 → learning rate, tốc độ cập nhật gradient.
Mục đích: Cập nhật tham số mô hình dựa trên gradient của loss function để giảm MSE trong quá trình huấn luyện.
Training loop
 Loop qua các epoch
for epoch in range(1, EPOCHS + 1):
Mỗi epoch = 1 lượt chạy qua toàn bộ dữ liệu train.
EPOCHS = số lần lặp tổng cộng để mô hình học dần.
Training phase
# Training
    model.train()
    running_train_loss = 0.0
    for Xb, yb in train_loader:
        batch_size = Xb.size(0)
        hidden = model.init_hidden(batch_size, DEVICE)


        optimizer.zero_grad()
        preds, _ = model(Xb, hidden)  # preds shape (batch, 1)
        preds = preds.squeeze(1)      # shape (batch,)
        loss = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # prevent exploding grad
        optimizer.step()


        running_train_loss += loss.item() * batch_size
   
    avg_train_loss = running_train_loss / len(train_ds)
    train_losses.append(avg_train_loss)

model.train() → bật chế độ training, cần thiết nếu có dropout/batchnorm.
train_loader → cung cấp batch dữ liệu train.
hidden = model.init_hidden(batch_size, DEVICE) → khởi tạo hidden + cell state cho batch.
optimizer.zero_grad() → xóa gradient cũ trước khi tính gradient mới.
preds, _ = model(Xb, hidden) → forward pass, dự đoán giá.
preds.squeeze(1) → đưa shape (batch, 1) → (batch,) để match yb.
loss = criterion(preds, yb) → tính MSE loss cho batch.
loss.backward() → backpropagation, tính gradient.
torch.nn.utils.clip_grad_norm_(...) → giới hạn gradient để tránh exploding gradient (LSTM dễ gặp).
optimizer.step() → cập nhật weights dựa trên gradient.
running_train_loss += loss.item() * batch_size → cộng loss từng batch.
avg_train_loss = running_train_loss / len(train_ds) → tính loss trung bình epoch.
train_losses.append(avg_train_loss) → lưu loss để plot learning curve.
Validation phase
   # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            batch_size = Xb.size(0)
            hidden = model.init_hidden(batch_size, DEVICE)
            preds, _ = model(Xb, hidden)
            preds = preds.squeeze(1)
            loss = criterion(preds, yb)
            running_val_loss += loss.item() * batch_size
    avg_val_loss = running_val_loss / len(val_ds)
    val_losses.append(avg_val_loss)

model.eval() → bật chế độ evaluation, tắt dropout, batchnorm không update.
with torch.no_grad(): → tắt tính gradient để tiết kiệm bộ nhớ.
Tương tự như training, nhưng không backward, không update weights.
Tính loss trung bình trên toàn bộ validation set → lưu vào val_losses để theo dõi.
Đánh giá trên test set
mae = mean_absolute_error(trues_price, preds_price) # Mức chênh lệch trung bình giữa dự đoán và giá thật
mse = mean_squared_error(trues_price, preds_price)  # Phạt sai số lớn mạnh hơn, đo độ ổn định của mô hình
rmse = np.sqrt(mse) # RMSE = √MSE → nên ta tự lấy căn.



MAE: độ chênh lệch trung bình tuyệt đối giữa dự đoán và thực tế.
MSE: trung bình bình phương sai số (phạt lỗi lớn hơn).
RMSE: căn bậc hai của MSE → đơn vị cùng với giá, dễ trực quan hơn.
Output:

Test MAE: 8.0702
MAE là độ chênh lệch trung bình tuyệt đối giữa dự đoán và giá thật.
Ở đây, đơn vị là đơn vị giá gốc (ví dụ USD, VND,… tùy dữ liệu).
Nghĩa là trung bình mỗi dự đoán lệch khoảng 8.07 so với giá thực tế.
MAE cho thấy lỗi dự đoán “trung bình” của model, không phạt quá nặng các outlier.

Test RMSE: 10.1410
RMSE là căn bậc hai của MSE, tức là trung bình bình phương sai số.
Vì bình phương nên các lỗi lớn bị phạt nặng hơn → RMSE thường ≥ MAE.
Ở đây, RMSE = 10.14, nghĩa là model mỗi dự đoán sai khoảng ±10 đơn vị về giá, và lỗi lớn hơn sẽ ảnh hưởng nhiều hơn đến RMSE.

Visualize
Loss


Đồ thị loss cho thấy cả Train Loss và Validation Loss đều giảm ổn định theo thời gian và tiến tới giá trị rất nhỏ (≈ 0.001 → 0.0002). Hai đường loss gần như song song và không tách rời nhau, cho thấy mô hình không gặp hiện tượng overfitting. Điều này chứng tỏ LSTM đã học được xu hướng biến động của giá cổ phiếu và có khả năng khái quát tốt trên dữ liệu chưa từng thấy.


Predicted vs Actual
Khả năng bám sát Xu hướng (Trend-Following): Mô hình thể hiện khả năng bám sát xu hướng chung của giá thực tế (Actual) một cách hiệu quả. Khi giá trị thực tăng hoặc giảm, giá trị dự đoán (Predicted) cũng di chuyển theo mô hình tương ứng.
Hiện tượng Trễ (Lag): Quan sát thấy có một độ trễ (lag) nhỏ, điển hình là 1 nhịp (step). Đường dự đoán có xu hướng phản ứng chậm hơn một chút so với sự thay đổi đột ngột của giá thực tế.
Biên độ Dao động (Volatility): Mô hình dự đoán có xu hướng "an toàn" hơn, tạo ra một đường cong mượt hơn. Nó dự đoán các đỉnh (peaks) thấp hơn và các đáy (troughs) cao hơn so với giá trị thực tế, cho thấy biên độ dao động của dự đoán bị giảm nhẹ.
Kết luận: Tổng quan, mô hình LSTM đã học thành công các mẫu (patterns) và xu hướng chính của dữ liệu, mang lại kết quả dự đoán tốt trên tập kiểm tra (test set).







