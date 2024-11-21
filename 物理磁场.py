def nanometers_to_meters(nanometers):
    """
    将纳米值转换为米值

    参数:
    nanometers: float，输入的纳米值

    返回:
    float，转换后的米值
    """
    meters = nanometers * 1e-9
    return meters
def calculate_magnetic_field(N, L, I):
    """
    计算螺线管内部的磁场强度

    参数:
    N: int，总匝数
    L: float，线圈长度（米）
    I: float，电流强度（安培）

    返回:
    float，磁场强度（特斯拉）
    """
    # 真空磁导率


    # 单位长度的匝数
    n = N / L  # 单位 匝/m

    # 磁场计算
    B = n * I  # 单位 T

    return B


# 用户输入
N = int(1200)
L = float(0.07)
I = float(input("请输入电流 I (单位: 安培): "))
n = int(input("请输入n"))
l1 = float(0.11)
# 计算磁场
B = calculate_magnetic_field(N, L, I)
meters = nanometers_to_meters(float(632.8))/2
# 输出结果
print(f"线圈内部的磁场强度为: {B:.6f} 特斯拉")
DL=n*meters
ddd=DL/l1
print(DL,ddd)