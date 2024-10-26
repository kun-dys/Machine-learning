# Machine-learning

## model文件夹包含一些机器学习算法代码，可直接调用

### nn(神经网络)方法调用说明
```python
# 导入
from model.nn import nnModel

# 将数据转换为PyTorch张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)  

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 训练函数
nnModel.train(X_train, y_train, num_epochs, lr, file_name, lossFile_path)
# X_train: 训练集特征
# y_train: 训练集标签
# num_epochs: 迭代次数（训练轮数）
# lr: 学习率
# file_name: 模型文件保存路径
# lossFile_path: 损失曲线保存路径

# 验证函数
nnModel.val(X_test, y_test, file_name)
# X_test: 测试集特征
# y_test: 测试集标签
# file_name: 模型文件路径
```
