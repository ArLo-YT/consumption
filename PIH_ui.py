import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置绘图样式
from matplotlib import font_manager, rcParams
import os
from matplotlib.ticker import FuncFormatter, MaxNLocator

# 尝试使用系统已有的中文字体
found_font = None

if found_font:
    rcParams['font.family'] = found_font
else:
    # 如果系统没中文字体，则加载项目自带字体
    font_path = os.path.join("simhei.ttf")  # 确保你把 SimHei.ttf 放在项目的 fonts/ 目录下
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

# 确保负号正常显示
rcParams['axes.unicode_minus'] = False


plt.rcParams.update({'font.size':14})

# 侧边栏展示 README.md（只显示到 "## 功能特点" 前）
st.sidebar.title("项目说明")
try:
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()
    cut_index = content.find("## 环境依赖")
    if cut_index != -1:
        content = content[:cut_index]
    st.sidebar.markdown(content, unsafe_allow_html=True)
except FileNotFoundError:
    st.sidebar.info("未找到 README.md（将 README.md 放在项目根目录即可在此显示）")




# —— 美化辅助：千分位格式、轴/网格统一风格 ——
def _thousands(x, pos):
    # 依需要改成保留小数：f"{x:,.2f}"
    return f"{x:,.0f}"

def _beautify_axes(ax, y_ticks=6):
    # 仅保留左/下脊线，其他去除；让图更“干净”
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    # 网格：仅 y 轴，虚线，淡一点
    ax.grid(True, axis='y', linestyle='--', linewidth=0.8, alpha=0.35)
    ax.grid(False, axis='x')
    # 刻度样式
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.tick_params(axis='both', labelsize=13)
    # 让线条别贴边
    ax.margins(x=0.02, y=0.08)

def _annotate_last(ax, xs, ys):
    # 标注最后一个点的数值
    if len(xs) == 0:
        return
    x_last, y_last = xs[-1], ys[-1]
    ax.scatter([x_last], [y_last], s=28, zorder=3)
    ax.annotate(
        f"{y_last:,.0f}",
        xy=(x_last, y_last),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#cccccc", alpha=0.8)
    )


# 定义函数
def coeff(r, s):
    if np.abs(r)> 0.0001:
        coefficient = r / (1 + r) / (1 - 1 / np.power(1 + r, s))
    else:
        coefficient = 1/s
    return coefficient

def consump(r, s, A_t, Y_t, r_e=None, f_t=0,r_ins=0):
    if r_e is None:
        r_e = r
    c_t = coeff(r_e-r_ins, s) * ((A_t - f_t / (1 + r) ** s) * (1 + r) + Y_t)
    return c_t

def pv(y_t, r):
    total = 0
    for i in range(len(y_t)):
        total += y_t[i] / (1 + r) ** i
    return total

def income(wage, grow_rate, years):
    inc = np.zeros(years)
    for i in range(years):
        inc[i] = wage * (1 + grow_rate) ** i
    return inc

def parse_custom_income(input_str, years):
    """
    解析用户输入的自定义收入，支持中文冒号和逗号，减少引号的影响
    """
    income_dict = {}
    if input_str.strip() == "":
        return income_dict
    
    # 去除左右引号
    input_str = input_str.strip().strip('\'"')
    # 替换中文符号为英文符号
    input_str = input_str.replace('：', ':').replace('，', ',')
    
    items = input_str.split(",")
    for item in items:
        try:
            index_str, value_str = item.split(":")
            year_num = int(index_str.strip())
            value = float(value_str.strip())
            index = year_num - 1
            if 0 <= index < years and value >= 0:
                income_dict[index] = value
        except:
            continue
    return income_dict

def simulate_and_output(years, wage, A_t_init, r_c, l, grow_rate, final_wealth, r_ins, custom_income_str):
    y_t = income(wage, grow_rate, years)

    # 解析自定义收入
    custom_income = parse_custom_income(custom_income_str, years)
    for year_idx in custom_income:
        y_t[year_idx] = custom_income[year_idx]

    # —— 变更点：不再提前建 DataFrame，而是先用列表收集 —— 
    rows = []

    A_t = A_t_init
    c_t_list = []
    A_t_list = []

    # 用于购买力折算
    c_t_inflation_adjusted = []

    dr = 0
    r = r_c

    A_t_min = 0
    A_t_max = 0
    A_t_warn = 0

    for i in range(years):
        r_e = r - l
        year_label = i + 1
        initial_A = A_t
        s = years - i
        Y_t = pv(y_t[i:years], r + dr)
        c_t = consump(r, s, A_t, Y_t, r_e, final_wealth, r_ins + dr)

        A_t = A_t * (1 + r) + y_t[i] - c_t

        # 计算每年消费的通胀折算（购买力）
        consumption_inflation_adjusted = c_t / ((1 + l) ** i)
        c_t_inflation_adjusted.append(consumption_inflation_adjusted)

        # —— 变更点：把一行结果收集到 rows 列表 —— 
        rows.append({
            "年份": year_label,
            "年初资产": round(initial_A, 2),
            "年内收入": round(y_t[i], 2),
            "全年消费": round(c_t, 2),
            "年末资产": round(A_t, 2)
        })

        c_t_list.append(c_t)
        A_t_list.append(A_t)

        # 警告逻辑（保持不变）
        if c_t < 0:
            st.warning("很遗憾，当前收入难以支持目标资产，请努力增加收入或调整目标")
            st.stop()

        if A_t < A_t_min:
            A_t_min = A_t

        if A_t > A_t_max:
            A_t_max = A_t

        if A_t_max > 0.0001 and A_t_min < -0.0001:
            A_t_warn = 1

    if A_t_warn:
        st.warning("当前周期内同时出现净资产和净负债，相应的，也应同时存在存款和贷款利率，因此结果可能不准确，仅供参考")

    # —— 变更点：循环结束后一次性创建 DataFrame —— 
    results = pd.DataFrame(rows, columns=["年份", "年初资产", "年内收入", "全年消费", "年末资产"])

    # 原有图表
    fig, axs = plt.subplots(2, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    time = list(range(1, years + 1))
    
    # 上图：收入 & 消费（两条线）
    axs[0].plot(time, y_t, linewidth=2.2, marker='o', markersize=4, label='全年收入')
    axs[0].plot(time, c_t_list, linewidth=2.2, marker='o', markersize=4, label='全年消费')
    axs[0].set_title('未来收入与消费', fontsize=18, pad=12)
    axs[0].set_ylabel('金额', fontsize=14)
    _beautify_axes(axs[0])
    _annotate_last(axs[0], time, y_t)
    _annotate_last(axs[0], time, c_t_list)
    axs[0].legend(frameon=False, fontsize=13, ncols=2, loc='upper left')
    
    # 下图：年末资产
    axs[1].plot(time, A_t_list, linewidth=2.2, marker='o', markersize=4)
    axs[1].set_title('年末资产', fontsize=18, pad=12)
    axs[1].set_xlabel('年份', fontsize=14)
    axs[1].set_ylabel('金额', fontsize=14)
    axs[1].axhline(0, linewidth=1.0, linestyle='--', alpha=0.4)
    _beautify_axes(axs[1])
    _annotate_last(axs[1], time, A_t_list)

    # 新增图表：每年消费的购买力变化
    fig_inflation, ax_inflation = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax_inflation.plot(time, c_t_inflation_adjusted, linewidth=2.2, marker='o', markersize=4)
    ax_inflation.set_title("通胀修正后消费购买力（以第1年物价为基准）", fontsize=18, pad=10)
    ax_inflation.set_xlabel("年份", fontsize=14)
    ax_inflation.set_ylabel("金额", fontsize=14)
    _beautify_axes(ax_inflation, y_ticks=5)
    _annotate_last(ax_inflation, time, c_t_inflation_adjusted)

    c_t_inflation_adjusted_total = sum(c_t_inflation_adjusted)
    return fig, results, fig_inflation, c_t_inflation_adjusted_total

# Streamlit界面
st.title("消费规划模拟工具")

# 用户参数
years = st.number_input("周期（最小5年，最大80年）", min_value=5, max_value=80, value=30)
A_t_init = st.number_input("当前资产（可为负数）", value=0.0)
final_wealth = st.number_input("最终目标资产（大于等于0）", min_value=0, value=0)
wage = st.number_input("当前年薪（大于0）", min_value=0.01, value=100000.0)
grow_rate = st.slider("预期薪水增长率（-5% 到20%）", min_value=-5.0, max_value=20.0, value=1., step=0.1) / 100
r_c = st.slider("年利率（最大20%，当前周期内负债较多请用贷款利率）", min_value=0.0, max_value=20.0, value=3., step=0.1) / 100
# r_d = st.slider("贷款利率（最大20%）", min_value=0.0, max_value=20.0, value=3.0, step=0.1) / 100

submit_enabled = True
# if r_c <= r_d:
#     submit_enabled = True
# else:
#     submit_enabled = False
#     st.write("贷款利率不能少于存款利率。")

l = st.slider("通胀率（-5% 到20%）", min_value=-5.0, max_value=20.0, value=2.0, step=0.1) / 100
r_ins = st.slider("消费偏好修正（-5% 到5%，+表示偏好未来消费，-表示偏好当下消费）", min_value=-5.0, max_value=5.0, value=0.0, step=0.1) / 100


st.subheader("自定义某些年份的收入（编号从1开始，例如： '1： 100000，5:200000' 表示第一年收入为100000，第5年收入为200000）")

custom_income_input = st.text_area("输入自定义收入", height=100)

if st.button("开始计算",disabled=not submit_enabled):
    fig, df_results, fig_inflation,c_t_total = simulate_and_output(
        years, wage, A_t_init, r_c, l, grow_rate, final_wealth, r_ins,custom_income_input
    )
    st.subheader("每年收入、消费和资产记录")
    st.dataframe(df_results)
    st.pyplot(fig, use_container_width=True)
    st.subheader("每年消费的购买力变化")
    st.pyplot(fig_inflation, use_container_width=True)
    # st.subheader("通胀修正后的购买力总和（按照第一年物价）")
    # st.subheader(round(c_t_total,2))
