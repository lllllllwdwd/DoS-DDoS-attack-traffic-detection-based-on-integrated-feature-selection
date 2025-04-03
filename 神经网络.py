import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, cohen_kappa_score

# ============== 1. 读取数据 ==============
file_path = r"filtered_features.csv"  # 请替换为你的 CSV 文件路径
data = pd.read_csv(file_path)

# ============== 2. 预处理数据 ==============
# 移除不必要的列
X = data.drop(columns=["Label"])  # 去掉 ID 和 ZIP.Code
y = data["Label"]  # 目标变量

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# ============== 3. 构建神经网络模型 ==============
class LoanApprovalNN(nn.Module):
    def __init__(self, input_dim):
        super(LoanApprovalNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),  # 输入层 -> 隐藏层1
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout 防止过拟合

            nn.Linear(64, 32),  # 隐藏层1 -> 隐藏层2
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1),  # 隐藏层2 -> 输出层
            nn.Sigmoid()  # 使用 Sigmoid 进行二分类
        )

    def forward(self, x):
        return self.model(x)


# 初始化模型
input_dim = X_train.shape[1]  # 特征维度
model = LoanApprovalNN(input_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类使用二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============== 4. 训练神经网络 ==============
# 训练神经网络
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = model(X_train_tensor).squeeze()  # 输出张量 shape -> [batch_size]

    # 目标张量的形状调整为 [batch_size]（去掉额外的维度）
    y_train_tensor = y_train_tensor.squeeze()  # 确保目标张量 shape -> [batch_size]

    # 计算损失
    loss = criterion(outputs, y_train_tensor)  # 计算损失

    # 反向传播
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# ============== 5. 评估模型 ==============
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_tensor).squeeze().numpy()
    y_pred = (y_pred_prob >= 0.5).astype(int)  # 0.5 为分类阈值

# 计算性能指标
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
cohen_kappa = cohen_kappa_score(y_test, y_pred)

print("\n模型评估结果:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Cohen Kappa Score: {cohen_kappa:.4f}")
print("Classification Report:\n", report)
print(f"AUC: {auc:.4f}")

# ============== 6. 绘制 AUC 曲线 ==============
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC Curve - Neural Network')
plt.legend(loc='lower right')
plt.savefig('output/神经网络_AUC.png')


def predict_fn(X):
    # Set the model to evaluation mode for inference
    model.eval()

    # Make predictions using the model
    with torch.no_grad():
        # Ensure X is a tensor, run through the model, and convert to numpy
        outputs = model(torch.tensor(X, dtype=torch.float32))

    # If it's a binary classification, we need the output to be probabilities (use sigmoid activation)
    return outputs.detach().numpy()


# 2. Initialize SHAP explainer with the predict function
explainer = shap.Explainer(predict_fn, X_train_tensor)  # Passing the prediction function

# 3. Compute SHAP values
shap_values = explainer(X_test_tensor)

# 4. Visualize SHAP values (feature importance)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=data.drop(columns=["ID", "Personal.Loan", "ZIP.Code"]).columns)
plt.savefig('output/神经网络_SHAP解释性.png')
'''
Accuracy: 0.9370
Cohen Kappa Score: 0.5687
Classification Report:
               precision    recall  f1-score   support

           0       0.94      0.99      0.97       895
           1       0.90      0.45      0.60       105

    accuracy                           0.94      1000
   macro avg       0.92      0.72      0.78      1000
weighted avg       0.94      0.94      0.93      1000

AUC: 0.9675
'''
