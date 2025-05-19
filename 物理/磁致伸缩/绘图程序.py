import matplotlib.pyplot as plt


def plot_graph(H_values, delta_L1_L1_values):
    """
    绘制以 ΔL1/L1 × 10⁻⁶ 为纵坐标，H × 10³ (A/m) 为横坐标的图

    参数:
    H_values: list，场强 H 的值 (单位：A/m)
    delta_L1_L1_values: list，对应的 ΔL1/L1 的值 (无量纲)

    返回:
    None
    """
    # 转换 H 为 H × 10³
    H_scaled = [H * 1e-3 for H in H_values]

    # 转换 ΔL1/L1 为 ΔL1/L1 × 10⁻⁶
    delta_L1_L1_scaled = [value * 1e6 for value in delta_L1_L1_values]

    # 绘制折线图（确保顺序绘制）
    plt.figure(figsize=(8, 6))
    plt.plot(H_scaled, delta_L1_L1_scaled, marker='o', linestyle='-', label='ΔL1/L1 vs H')

    # 设置坐标轴标签和标题
    plt.xlabel("场强 H × 10³ (A/m)", fontsize=12)
    plt.ylabel("ΔL1/L1 × 10⁻⁶", fontsize=12)
    plt.title("磁致伸缩特性曲线", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()


def user_input():
    """
    从用户输入中获取 H 和 ΔL1/L1 的值
    """
    print("请输入场强 H 和 ΔL1/L1 数据，以逗号分隔，每组数据一行，输入 'done' 结束：")
    H_values = []
    delta_L1_L1_values = []

    while True:
        user_input = input("输入一组数据 (H, ΔL1/L1)：")
        if user_input.lower() == 'done':
            break
        try:
            # 用户输入拆分
            H, delta_L1_L1 = map(float, user_input.split(','))
            H_values.append(H)
            delta_L1_L1_values.append(delta_L1_L1)
        except ValueError:
            print("输入格式错误，请重新输入，例如：1000, 0.00001")

    return H_values, delta_L1_L1_values


# 主程序
if __name__ == "__main__":
    print("磁致伸缩特性曲线绘制程序")
    H_values, delta_L1_L1_values = user_input()
    if len(H_values) > 0 and len(delta_L1_L1_values) > 0:
        plot_graph(H_values, delta_L1_L1_values)
    else:
        print("未输入任何有效数据，程序结束。")
