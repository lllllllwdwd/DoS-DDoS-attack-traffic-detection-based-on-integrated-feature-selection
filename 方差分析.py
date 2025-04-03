import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif

# 读取数据
df = pd.read_csv(r"C:\Users\月亮姐姐\Downloads\03-11\Portmap.csv", low_memory=False)

# 检查哪些列是非数值型
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
print("\n⚠️ 发现以下非数值列：", non_numeric_cols)

# 将所有非数值列转换为 NaN
for col in non_numeric_cols:
    if col != "Label":  # 跳过目标变量
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 检查 NaN 值
print("\n🔍 NaN 值统计：")
print(df.isnull().sum())

# 填充 NaN（使用中位数填充）
df.fillna(df.median(), inplace=True)

# 再次检查 NaN 是否被清理
print("\n✅ NaN 值清理完成，剩余 NaN 数量：")
print(df.isnull().sum().sum())

# 重新分离 X 和 y
X = df.drop(columns=["Label"])
y = df["Label"]

# 计算 ANOVA F 值和 p 值
F_values, p_values = f_classif(X, y)

# 创建 DataFrame 并排序
anova_results = pd.DataFrame({
    "特征": X.columns,
    "F值": F_values,
    "p值": p_values
}).sort_values(by="p值")

# 显示 ANOVA 结果
print("\n🔹 ANOVA 结果（按 p 值排序）：")
print(anova_results)

# 设定显著性水平（p > 0.05 的特征被移除）
significant_features = anova_results[anova_results["p值"] < 0.05]["特征"].values
X_filtered = X[significant_features]

# 保存结果
anova_results.to_csv("anova_results.csv", index=False)
X_filtered.to_csv("filtered_features.csv", index=False)

print("\n✅ ANOVA 结果已保存：anova_results.csv")
print("✅ 筛选后的特征数据已保存：filtered_features.csv")
