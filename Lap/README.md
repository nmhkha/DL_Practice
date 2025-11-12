Dự báo Giá Cổ phiếu (Stock Price Forecasting)
Dự án này trình bày một mô hình học sâu để dự báo giá cổ phiếu bằng cách sử dụng Mạng thần kinh Hồi quy Dài-Ngắn hạn (LSTM).

🎯 Mục tiêu (Goal)
Mục tiêu chính là dự đoán giá đóng cửa (Close) của cổ phiếu vào ngày hôm sau (next-day) dựa trên chuỗi dữ liệu giá trong 60 ngày trước đó.

📊 Dữ liệu (Dataset)
Nguồn: Dữ liệu được tải tự động từ Yahoo Finance.

Mã cổ phiếu: GOOGL (Alphabet Inc.).

Khung thời gian: 2 năm (period='2y').

Đặc trưng (Feature): Chỉ sử dụng giá đóng cửa (Close) cho việc dự báo.

🤖 Mô hình (Model)
Loại mô hình: LSTM (Long Short-Term Memory).

Loại tác vụ: Hồi quy (Regression).

Chi tiết: Mô hình sử dụng dữ liệu của 60 ngày liên tiếp (SEQ_LEN = 60) để dự đoán giá của ngày thứ 61.