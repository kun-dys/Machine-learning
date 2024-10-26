import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 定义模型
class nnModel(nn.Module):
    def __init__(self, input_n, output_n):
        super(nnModel, self).__init__()
        self.fc1 = nn.Linear(input_n, 128)  # 输入层到第一个隐藏层，64个节点
        self.fc2 = nn.Linear(128, 64)  # 第一个隐藏层到第二个隐藏层，32个节点
        self.fc3 = nn.Linear(64, output_n)  # 第二个隐藏层到输出层，3个节点

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train(X_train, y_train, num_epochs, lr=0.01, file_name='best_model.pth', lossFile_path='loss.png'):
        # 实例化模型
        input_n = len(X_train[0])
        output_n = len(y_train[0])
        model = nnModel(input_n, output_n)
        # 定义损失函数和优化器
        criterion = nn.MSELoss()  # 回归损失函数
        optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用Adam优化器

        # 初始化最小损失值
        best_loss = float('inf')
        best_model_state_dict = None

        # 训练模型
        num_epochs = num_epochs
        loss_history = []  # 记录每次迭代的损失值

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            # 保存最佳模型
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state_dict = model.state_dict()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 保存最佳模型
        file_name = file_name
        if best_model_state_dict is not None:
            torch.save(best_model_state_dict, file_name)

        lossFile_path = lossFile_path
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.savefig(lossFile_path)

    def val(X_test, y_test, file_name='best_model.pth'):
        input_n = len(X_test[0])
        output_n = len(y_test[0])
        model = nnModel(input_n, output_n)
        model.load_state_dict(torch.load(file_name))
        # 假设 `y_test` 是真实值，`y_pred` 是模型的预测值
        y_pred = model(X_test)
        y_mean = torch.mean(y_test)
        # 计算均方误差（MSE）
        mse = torch.mean((y_test - y_pred) ** 2)
        mse_value = mse.item()

        # 计算均方根误差（RMSE）
        rmse = torch.sqrt(mse)
        rmse_value = rmse.item()

        # 计算平均绝对误差（MAE）
        mae = torch.mean(torch.abs(y_test - y_pred))
        mae_value = mae.item()

        # R2 值
        sst = torch.sum((y_test - y_mean)**2)
        ssr = torch.sum((y_test - y_pred)**2)
        r_squared = 1 - (ssr / sst)
        r_squared_value = r_squared.item()

        print(f'MSE: {mse_value:.4f}')
        print(f'RMSE: {rmse_value:.4f}')
        print(f'MAE: {mae_value:.4f}')
        print(f'R²: {r_squared_value:.4f}')

