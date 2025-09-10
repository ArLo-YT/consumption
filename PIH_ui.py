import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置绘图样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']

plt.rcParams.update({'font.size':14})

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
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    time = list(range(1, years + 1))
    axs[0].plot(time, y_t, color='green', label='全年收入')
    axs[0].plot(time, c_t_list, color='orange', label='全年消费')
    axs[0].set_title('未来收入、消费')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].legend()

    axs[1].plot(time, A_t_list)
    axs[1].set_title('年末资产')
    axs[1].set_xlabel('年份')
    axs[1].set_ylabel('')

    axs[0].grid(True, axis='y', linestyle='--', alpha=0.7)
    axs[1].grid(True, axis='y', linestyle='--', alpha=0.7)

    # 新增图表：每年消费的购买力变化
    fig_inflation, ax_inflation = plt.subplots(figsize=(10, 4))
    ax_inflation.plot(time, c_t_inflation_adjusted, color='blue')
    ax_inflation.set_title("通胀修正后消费购买力变化（按照第一年物价）")
    ax_inflation.set_xlabel("年份")
    ax_inflation.set_ylabel("")
    ax_inflation.grid(True, axis='y', linestyle='--', alpha=0.7)

    c_t_inflation_adjusted_total = sum(c_t_inflation_adjusted)
    return fig, results, fig_inflation, c_t_inflation_adjusted_total

# Streamlit界面
st.title("未来消费规划")


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
    st.pyplot(fig)
    st.subheader("每年消费的购买力变化")
    st.pyplot(fig_inflation)
    st.subheader("通胀修正后的购买力总和（按照第一年物价）")
    st.subheader(round(c_t_total,2))
