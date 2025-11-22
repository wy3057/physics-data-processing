

# 📘 填料吸收塔实验数据自动处理脚本

本项目用于 **自动处理化工原理课程中的填料吸收塔实验数据**，包括：

* 干塔 & 湿塔流体力学
* CO₂–空气–水体系传质实验
* 自动计算各种关键参数（u、ΔP/Z、G、L、A、NOL、Kxa、吸收率）
* 自动生成图像与输出标准化表格

你只需要准备一份 Excel 表格，即可一键完成所有计算和绘图。

---

## 📂 文件结构

```
experiment_data.xlsx        # 实验数据（需自行填写）
absorption_analysis_excel.py  # 数据处理主程序
dp_u_curve.png              # 输出：干/湿塔 (ΔP/Z)-u 曲线
Kxa_vs_L.png                # 输出：Kxa-L 曲线
```

---

## 📝 Excel 数据格式说明（experiment_data.xlsx）

你需要在同一个 Excel 文件中建立三个工作表：

### 1. `dry`（干塔流体力学）

| 列名           | 意义           |
| ------------ | ------------ |
| DeltaP_mmH2O | 压降（mmH₂O）    |
| V_m3_h       | 气体体积流量（m³/h） |

### 2. `wet`（湿塔流体力学）

| 列名           | 意义           |
| ------------ | ------------ |
| DeltaP_mmH2O | 压降（mmH₂O）    |
| V_m3_h       | 气体体积流量（m³/h） |
| L_L_h        | 液体流量（L/h）    |
| Phenomenon   | 现象（正常/持液/液泛） |

### 3. `mass_transfer`（传质实验）

| 列名          | 意义            |
| ----------- | ------------- |
| V_CO2_L_min | CO₂ 流量（L/min） |
| V_air_L_min | 空气流量（L/min）   |
| L_L_h       | 液体流量（L/h）     |
| T_liq_out_C | 液体出口温度（℃）     |
| y_in        | 气相入口 CO₂ 摩尔分率 |
| y_out       | 气相出口 CO₂ 摩尔分率 |
| T_gas_C     | 气体温度（可不填）     |

> ⚠️ 列名必须完全一致（大小写也要一致）。

---

## ▶️ 使用方法

1. 将实验数据填入 `experiment_data.xlsx`
2. 运行脚本：

```bash
python absorption_analysis_excel.py
```

3. 程序会自动输出：

* 控制台：处理后的三组数据表
* 生成图像文件：

  * `dp_u_curve.png`
  * `Kxa_vs_L.png`

可用于实验报告或课程分析。

---

## 📊 输出结果示例

* **干/湿塔压降曲线**：`dp_u_curve.png`
* **传质系数 Kxa 随液量变化曲线**：`Kxa_vs_L.png`

---

