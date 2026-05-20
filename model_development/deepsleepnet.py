import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class DeepSleepNet(nn.Module):
    def __init__(self, num_classes=5):
        super(DeepSleepNet, self).__init__()

        # ─── 1. CNN NHÁNH NHỎ (Học đặc trưng tần số - Fine filter) ───
        self.features_small = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, padding=25, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(0.5),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(0.5)
        )

        # ─── 2. CNN NHÁNH LỚN (Học đặc trưng thời gian - Coarse filter) ───
        self.features_large = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50,
                      padding=200, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(0.5),

            nn.Conv1d(64, 128, kernel_size=6, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=6,
                      stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.5)
        )

        # ─── 3. MẠNG CHUỖI BiLSTM VÀ RESIDUAL CONNECTION ───
        self.flatten_dim = 2560
        self.dropout_fc = nn.Dropout(0.5)

        # BiLSTM để học ngữ cảnh chuỗi giữa các epoch giấc ngủ
        self.bilstm = nn.LSTM(
            input_size=self.flatten_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        # Đường tắt kết nối thẳng đặc trưng CNN tới phân loại (Residual Connection)
        self.residual_fc = nn.Linear(self.flatten_dim, 1024)

        # Lớp phân loại cuối cùng (5 giai đoạn giấc ngủ)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x shape: (Batch, Sequence_Length, 1, 3000)
        batch_size, seq_len, channels, signal_len = x.size()

        # Phẳng hóa Batch và Sequence để đẩy qua CNN 1D
        x = x.view(batch_size * seq_len, channels, signal_len)

        # Đẩy qua 2 nhánh CNN song song
        out_small = self.features_small(x)
        out_large = self.features_large(x)

        # Nối đặc trưng của 2 nhánh lại với nhau
        out_small = out_small.view(out_small.size(0), -1)
        out_large = out_large.view(out_large.size(0), -1)
        cnn_features = torch.cat((out_small, out_large), dim=1)
        cnn_features = self.dropout_fc(cnn_features)

        # Khôi phục lại cấu trúc chuỗi thời gian: (Batch, Sequence_Length, Flatten_Dim)
        lstm_input = cnn_features.view(batch_size, seq_len, self.flatten_dim)

        # Đẩy qua BiLSTM -> Chiều ra: (Batch, Seq_Len, 1024)
        lstm_out, _ = self.bilstm(lstm_input)

        # Tính toán đường tắt Residual
        residual = self.residual_fc(cnn_features)
        residual = residual.view(batch_size, seq_len, 1024)

        # Cộng kết quả BiLSTM và Residual
        final_features = lstm_out + residual
        final_features = self.dropout_fc(final_features)

        # Dự đoán phân loại
        logits = self.classifier(final_features)  # (Batch, Seq_Len, 5)
        return logits

    def fit(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=16, device="cpu"):
        """
        Hàm huấn luyện mô hình (Đã đổi tên từ train thành fit để tránh trùng với self.train() của PyTorch)
        """
        # Kiểm tra và chèn thêm chiều channel nếu cần
        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, axis=2)
            X_val = np.expand_dims(X_val, axis=2)

        train_x_tensor = torch.tensor(X_train, dtype=torch.float32)
        train_y_tensor = torch.tensor(y_train, dtype=torch.long)
        val_x_tensor = torch.tensor(X_val, dtype=torch.float32)
        val_y_tensor = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        val_dataset = TensorDataset(val_x_tensor, val_y_tensor)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

        print(f"Bắt đầu huấn luyện trên thiết bị: {device}")

        for epoch in range(epochs):
            self.train()  # Bật chế độ Train của PyTorch (Dropout, BatchNorm hoạt động)
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)
                outputs = outputs.view(-1, 5)
                labels = labels.view(-1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            # Đánh giá trên tập Validation cuối mỗi Epoch sử dụng chính hàm evaluate nội bộ
            val_acc = self.evaluate(
                X_val, y_val, batch_size=batch_size, device=device)
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

        return self

    def evaluate(self, X_val, y_val, batch_size=16, device="cpu"):
        """
        Hàm đánh giá độ chính xác (Accuracy) trên tập dữ liệu bất kỳ
        """
        if len(X_val.shape) == 3:
            X_val = np.expand_dims(X_val, axis=2)

        val_x_tensor = torch.tensor(X_val, dtype=torch.float32)
        val_y_tensor = torch.tensor(y_val, dtype=torch.long)

        val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        self.eval()  # Bật chế độ Eval (Tắt Dropout)
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                outputs = outputs.view(-1, 5)
                labels = labels.view(-1)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        return val_acc
