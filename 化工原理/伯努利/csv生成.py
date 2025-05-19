import pandas as pd

# 表一数据 (流量: 580, 500, 400 l/h)
data_set1 = {
    'Group': ['Group1', 'Group2', 'Group3'],
    'Flow(l/h)': [580, 500, 400],
    'P1(mmH2O)': [31.35, 34.7, 38.4],
    'P2(mmH2O)': [35.60, 37.8, 40.3],
    'P3(mmH2O)': [31.60, 34.5, 39.1],
    'P4(mmH2O)': [30.25, 33.7, 37.6],
    'P5(mmH2O)': [25.71, 30.5, 35.8],
    'P6(mmH2O)': [12.20, 21.3, 29.5],
    'P7(mmH2O)': [21.35, 27, 33.5],
    'P8(mmH2O)': [24.00, 19.5, 35],
    'P9(mmH2O)': [25.00, 30, 35.5],
    'P10(mmH2O)': [25.00, 3, 18.5],
    'P11(mmH2O)': [-12.50, 4, 19.5],
    'P12(mmH2O)': [-11.00, -0.5, 16.5],
    'P13(mmH2O)': [-16.70, 3, 18.5],
    'P14(mmH2O)': [-12.50, -4, 14.5],
    'P15(mmH2O)': [-21.50, -4, 14.5]
}

# 表三数据 (流量: 300, 200, 100 l/h)
data_set2 = {
    'Group': ['Group4', 'Group5', 'Group6'],
    'Flow(l/h)': [300, 200, 100],
    'P1(mmH2O)': [42, 43, 44],
    'P2(mmH2O)': [42, 43.5, 44.1],
    'P3(mmH2O)': [40.5, 42.9, 44],
    'P4(mmH2O)': [39.5, 42.8, 43.9],
    'P5(mmH2O)': [36.2, 42.3, 43.6],
    'P6(mmH2O)': [388, 40.8, 43.6],
    'P7(mmH2O)': [39.5, 41.5, 43.6],
    'P8(mmH2O)': [30, 41.9, 43.9],
    'P9(mmH2O)': [30.5, 42.3, 43.8],
    'P10(mmH2O)': [30, 38.5, 43.3],
    'P11(mmH2O)': [30.5, 38.7, 43.1],
    'P12(mmH2O)': [28.9, 38.1, 431],  # 注意：可能有输入错误
    'P13(mmH2O)': [30.1, 38.2, 42.9],
    'P14(mmH2O)': [27.8, 37.2, 42.9],
    'P15(mmH2O)': [27.8, 37.3, 42.9]
}

# 转换为DataFrame
df_set1 = pd.DataFrame(data_set1)
df_set2 = pd.DataFrame(data_set2)

# 合并两个DataFrame
df_combined = pd.concat([df_set1, df_set2], ignore_index=True)

# 保存为CSV文件
df_combined.to_csv('experiment_data_combined.csv', index=False)
print("文件 'experiment_data_combined.csv' 已生成！")
