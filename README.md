💤 Dự án Phân tích Dữ liệu Sleep-EDF
Dự án này sử dụng Python để đọc và trực quan hóa dữ liệu đa ký giấc ngủ (Polysomnography) từ bộ dữ liệu Sleep-EDF Database Expanded.

📁 Cấu trúc thư mục
sleep-edf-database-expanded-1.0.0/: Thư mục chứa dữ liệu gốc (.edf).

index.ipynb: File Jupyter Notebook chứa code xử lý và hiển thị sóng.

README.md: Hướng dẫn sử dụng.

🚀 Hướng dẫn cài đặt

1. Yêu cầu hệ thống:
   Máy tính đã cài sẵn Python (khuyên dùng phiên bản 3.8 trở lên).

2. Cài đặt các thư viện cần thiết:
   Mở terminal (hoặc Command Prompt) và chạy lệnh sau để cài đặt các thư viện xử lý tín hiệu y sinh và hiển thị:

- pip3 install mne matplotlib pandas yasa

🛠 Cách chạy dự án
Mở file index.ipynb bằng VS Code (đã cài extension Jupyter) hoặc chạy lệnh jupyter notebook trong thư mục dự án.

Đảm bảo đường dẫn trong code trỏ đúng vào thư mục dữ liệu:

Python

# Ví dụ đường dẫn đúng

raw = mne.io.read_raw_edf('sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001E0-PSG.edf')
Nhấn Run All hoặc chạy từng ô (cell) để xem kết quả.

📊 Giải thích các phím tắt khi xem biểu đồ (MNE Plot)
Khi cửa sổ biểu đồ hiện lên, bạn có thể tương tác:

Phím + / -: Tăng/Giảm tỉ lệ biên độ sóng (Scaling).

Phím a: Tự động căn chỉnh sóng cho vừa màn hình.

Phím mũi tên ← / →: Di chuyển tới/lui theo thời gian.

Chuột: Click vào tên kênh bên trái để ẩn/hiện kênh đó.
