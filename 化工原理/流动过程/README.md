# 流动过程综合实验工具

`main.py` 从 `experiment_data.txt` 中读取离心泵、管路以及摩擦损失三张表的数据，自动计算扬程、轴功率、泵效率、管路特性以及摩擦阻力系数，最后生成文字报告与特性曲线。

## 数据准备

`experiment_data.txt` 需要包含：

- **Table 1: Centrifugal Pump Data**：频率、流量、真空/压力表读数、功率等 6 列。
- **Table 2: Pipeline Characteristic Data**：变频器频率、泵频率、流量、真空/压力表读数等 6 列。
- **Table 3: Friction Loss Data**：流量、压差等 3 列。
- **Parameters**：在文末以 `名称: 数值` 的形式给出管径、长度、流体黏度等参数。

示例文件已放在目录中，可作为格式参考。

## 使用方法

```bash
cd 化工原理/流动过程
python main.py
```

脚本会自动：

1. 解析文本文件并转换单位。
2. 计算离心泵特性曲线 (H-Q、N-Q、η-Q)。
3. 计算管路特性及直管摩擦系数/雷诺数。
4. 生成 `experiment_report.txt` 并绘制 3 张图（泵特性曲线、管路特性、λ-Re）。

如需调整效率或流体性质，可修改文件顶部的 `eta_motor`、`rho`、`g` 等常量。
