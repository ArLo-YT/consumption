import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 绘图与字体 ======
from matplotlib import font_manager, rcParams
import os
from matplotlib.ticker import FuncFormatter, MaxNLocator
from io import BytesIO

# ---- 统一外观：更大的默认字号 ----
plt.rcParams.update({'font.size': 16})

# ---- 中文字体兜底 ----
found_font = None
if found_font:
    rcParams['font.family'] = found_font
else:
    font_path = os.path.join("simhei.ttf")  # 放在项目根目录或同级
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
rcParams['axes.unicode_minus'] = False

# ---- 颜色规范 ----
COLOR_INCOME = "#1f77b4"   # 蓝
COLOR_CONS   = "#f2c200"   # 黄
COLOR_ASSET  = "#000000"   # 黑
COLOR_MARKER = "#333333"

# ====== Sidebar：项目说明（裁掉“## 环境依赖”之后的内容） ======
st.sidebar.title("项目说明")
@st.cache_data(show_spinner=False)
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
        cut_index = content.find("## 环境依赖")
        return content if cut_index == -1 else content[:cut_index]
    except FileNotFoundError:
        return None

_readme = read_readme()
if _readme:
    st.sidebar.markdown(_readme, unsafe_allow_html=True)
else:
    st.sidebar.info("未找到 README.md（将 README.md 放在项目根目录即可在此显示）")

# ====== 小工具：格式化、坐标轴美化、末点标注 ======
def _thousands(x, pos):
    return f"{x:,.0f}"

def _beautify_axes(ax, y_ticks=6):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.9, alpha=0.35)
    ax.grid(False, axis='x')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.tick_params(axis='both', labelsize=14)
    ax.margins(x=0.02, y=0.08)

def _annotate_last(ax, xs, ys):
    if len(xs) == 0:
        return
    x_last, y_last = xs[-1], ys[-1]
    ax.scatter([x_last], [y_last], s=36, zorder=3, color=COLOR_MARKER)
    ax.annotate(
        f"{y_last:,.0f}",
        xy=(x_last, y_last),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#cccccc", alpha=0.85)
    )

# ====== 计算核心 ======
def coeff(r, s):
    if np.abs(r) > 1e-4:
        return r / (1 + r) / (1 - 1 / np.power(1 + r, s))
    else:
        return 1 / s

def consump(r, s, A_t, Y_t, r_e=None, f_t=0, r_ins=0):
    if r_e is None:
        r_e = r
    c_t = coeff(r_e - r_ins, s) * ((A_t - f_t / (1 + r) ** s) * (1 + r) + Y_t)
    return c_t

def pv(y_t, r):
    total = 0.0
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
    支持中文冒号/逗号，自动去引号。格式示例：
    1：100000，5:200000
    """
    income_dict = {}
    if input_str is None:
        return income_dict
    s = input_str.strip().strip('\'"')
    if s == "":
        return income_dict
    s = s.replace('：', ':').replace('，', ',')
    items = s.split(",")
    for item in items:
        try:
            index_str, value_str = item.split(":")
            year_num = int(index_str.strip())
            value = float(value_str.strip())
            index = year_num - 1
            if 0 <= index < years and value >= 0:
                income_dict[index] = value
        except Exception:
            continue
    return income_dict

def simulate_and_output(years, wage, A_t_init, r_c, l, grow_rate, final_wealth, r_ins, custom_income_str):
    y_t = income(wage, grow_rate, years)
    # 覆盖自定义收入
    custom_income = parse_custom_income(custom_income_str, years)
    for year_idx in custom_income:
        y_t[year_idx] = custom_income[year_idx]

    rows = []
    A_t = A_t_init
    c_t_list, A_t_list, c_t_inflation_adjusted = [], [], []

    dr = 0.0
    r = r_c

    A_t_min = 0.0
    A_t_max = 0.0
    A_t_warn = 0

    for i in range(years):
        r_e = r - l
        year_label = i + 1
        initial_A = A_t
        s = years - i
        Y_t = pv(y_t[i:years], r + dr)
        c_t = consump(r, s, A_t, Y_t, r_e, final_wealth, r_ins + dr)

        A_t = A_t * (1 + r) + y_t[i] - c_t

        # 购买力（以第 1 年物价为基准）
        c_t_inflation_adjusted.append(c_t / ((1 + l) ** i))

        rows.append({
            "年份": year_label,
            "年初资产": round(initial_A, 2),
            "年内收入": round(y_t[i], 2),
            "全年消费": round(c_t, 2),
            "年末资产": round(A_t, 2)
        })

        c_t_list.append(c_t)
        A_t_list.append(A_t)

        # 风险提示
        if c_t < 0:
            st.warning("很遗憾，当前收入难以支持目标资产，请增加收入/降低目标/调整偏好后重试。")
            st.stop()

        A_t_min = min(A_t_min, A_t)
        A_t_max = max(A_t_max, A_t)
        if A_t_max > 1e-4 and A_t_min < -1e-4:
            A_t_warn = 1

    if A_t_warn:
        st.warning("本周期内同时出现净资产与净负债，现实中应区分存款/贷款利率，本结果仅供参考。")

    results = pd.DataFrame(rows, columns=["年份", "年初资产", "年内收入", "全年消费", "年末资产"])

    # ====== 图1：收入 vs 消费 ======
    fig, axs = plt.subplots(2, 1, figsize=(10.8, 12.0), sharex=True, constrained_layout=True)
    time = list(range(1, years + 1))

    axs[0].plot(time, y_t, color=COLOR_INCOME, linewidth=2.4, marker='o', markersize=4.5, label='全年收入')
    axs[0].plot(time, c_t_list, color=COLOR_CONS, linewidth=2.4, marker='o', markersize=4.5, label='全年消费')
    axs[0].set_title('未来收入与消费', fontsize=20, pad=12)
    axs[0].set_ylabel('金额', fontsize=16)
    _beautify_axes(axs[0])
    _annotate_last(axs[0], time, y_t)
    _annotate_last(axs[0], time, c_t_list)
    axs[0].legend(frameon=False, fontsize=14, ncols=2, loc='upper left')

    # 年末资产
    axs[1].plot(time, A_t_list, color=COLOR_ASSET, linewidth=2.4, marker='o', markersize=4.5)
    axs[1].set_title('年末资产', fontsize=20, pad=12)
    axs[1].set_xlabel('年份', fontsize=16)
    axs[1].set_ylabel('金额', fontsize=16)
    axs[1].axhline(0, linewidth=1.0, linestyle='--', alpha=0.4, color="#666666")
    _beautify_axes(axs[1])
    _annotate_last(axs[1], time, A_t_list)

    # ====== 图2：通胀修正后消费购买力 ======
    fig_inflation, ax_inflation = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    ax_inflation.plot(time, c_t_inflation_adjusted, color=COLOR_CONS, linewidth=2.4, marker='o', markersize=4.5)
    ax_inflation.set_title("通胀修正后消费购买力（以第1年物价为基准）", fontsize=20, pad=10)
    ax_inflation.set_xlabel("年份", fontsize=16)
    ax_inflation.set_ylabel("购买力", fontsize=16)
    _beautify_axes(ax_inflation, y_ticks=5)
    _annotate_last(ax_inflation, time, c_t_inflation_adjusted)

    c_t_inflation_adjusted_total = float(np.sum(c_t_inflation_adjusted))
    return fig, results, fig_inflation, c_t_inflation_adjusted_total

# ====== 预设场景 ======
PRESETS = {
    "— 请选择 / 不使用预设 —": {},
    "基础稳定（30年，起薪10万，增速3%，通胀2%，年利率3%）": dict(
        years=30, A_t_init=0.0, final_wealth=0.0,
        wage=100000.0, grow_rate=0.03, r_c=0.03, l=0.02, r_ins=0.00,
        custom_income="1:100000, 5:120000"
    ),
    "房贷压力（40年，初始-50万，利率5%，增速2%）": dict(
        years=40, A_t_init=-500000.0, final_wealth=0.0,
        wage=120000.0, grow_rate=0.02, r_c=0.05, l=0.02, r_ins=-0.01,
        custom_income="1:120000, 10:180000"
    ),
    "高成长（25年，起薪8万，增速10%，通胀3%）": dict(
        years=25, A_t_init=0.0, final_wealth=200000.0,
        wage=80000.0, grow_rate=0.10, r_c=0.04, l=0.03, r_ins=0.00,
        custom_income=""
    ),
    "提前退休（20年，初始200万，目标300万，偏好未来+2%）": dict(
        years=20, A_t_init=2000000.0, final_wealth=3000000.0,
        wage=150000.0, grow_rate=0.02, r_c=0.04, l=0.02, r_ins=0.02,
        custom_income="3:500000, 5:800000"
    ),
}

st.sidebar.markdown("### 预设场景")
chosen_preset = st.sidebar.selectbox("一键载入常见场景", list(PRESETS.keys()), index=0)
if st.sidebar.button("载入该场景参数", use_container_width=True, type="primary"):
    preset = PRESETS.get(chosen_preset, {})
    for k, v in preset.items():
        st.session_state[k] = v
    st.sidebar.success(f"已载入：{chosen_preset}（可在主界面查看与修改）")

# ====== 主界面 ======
st.title("消费规划模拟工具（统一配色：收入=蓝｜消费=黄｜资产=黑）")

# 从 session_state 读默认（若已加载预设则覆盖控件默认值）
def _sv(key, default):  # safe value
    return st.session_state.get(key, default)

years       = st.number_input("周期（最小5年，最大80年）", min_value=5, max_value=80,
                              value=_sv("years", 30), key="years")
A_t_init    = st.number_input("当前资产（可为负数）", value=_sv("A_t_init", 0.0), key="A_t_init")
final_wealth= st.number_input("最终目标资产（大于等于0）", min_value=0,
                              value=int(_sv("final_wealth", 0.0)), key="final_wealth")
wage        = st.number_input("当前年薪（大于0）", min_value=0.01, value=_sv("wage", 100000.0), key="wage")
grow_rate   = st.slider("预期薪水增长率（-5% 到 20%）", min_value=-5.0, max_value=20.0,
                        value=float(_sv("grow_rate", 1.0)*100), step=0.1, key="gr_slider") / 100
r_c         = st.slider("年利率（最大20%，负债多时可理解为贷款利率）", min_value=0.0, max_value=20.0,
                        value=float(_sv("r_c", 3.0)*100), step=0.1, key="rc_slider") / 100
l           = st.slider("通胀率（-5% 到 20%）", min_value=-5.0, max_value=20.0,
                        value=float(_sv("l", 2.0)*100), step=0.1, key="l_slider") / 100
r_ins       = st.slider("消费偏好修正（-5% 到 5%，+偏好未来，-偏好当下）", min_value=-5.0, max_value=5.0,
                        value=float(_sv("r_ins", 0.0)*100), step=0.1, key="rins_slider") / 100

st.subheader("自定义某些年份的收入（编号从1开始）")
st.caption("示例：`1：100000，5:200000` 表示第1年=100000，第5年=200000；支持中文冒号/逗号。")
custom_income_input = st.text_area(
    "输入自定义收入",
    height=100,
    value=_sv("custom_income", ""),
    key="custom_income"
)

submit_enabled = True

# ====== 计算 & 展示 ======
if st.button("开始计算", disabled=not submit_enabled, type="primary"):
    fig, df_results, fig_inflation, c_t_total = simulate_and_output(
        years=int(years),
        wage=float(wage),
        A_t_init=float(A_t_init),
        r_c=float(r_c),
        l=float(l),
        grow_rate=float(grow_rate),
        final_wealth=float(final_wealth),
        r_ins=float(r_ins),
        custom_income_str=custom_income_input
    )

    # 结果表格
    st.subheader("每年收入、消费和资产记录")
    st.dataframe(df_results, use_container_width=True)

    # 图展示
    st.pyplot(fig, use_container_width=True)
    st.subheader("每年消费的购买力变化")
    st.pyplot(fig_inflation, use_container_width=True)

    # ====== 导出区 ======
    st.subheader("下载导出")
    # CSV
    csv_bytes = df_results.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="下载数据（CSV）",
        data=csv_bytes,
        file_name="消费规划结果.csv",
        mime="text/csv",
        use_container_width=True
    )
    # Excel
    xbuf = BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="results", index=False)
        # 也可在此写入汇总指标等
    st.download_button(
        label="下载数据（Excel）",
        data=xbuf.getvalue(),
        file_name="消费规划结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    # 图 PNG（两张）
    def _fig_to_png_bytes(f):
        buf = BytesIO()
        f.savefig(buf, format="png", dpi=180, bbox_inches="tight")
        buf.seek(0)
        return buf

    st.download_button(
        label="下载图像（收入-消费-资产 PNG）",
        data=_fig_to_png_bytes(fig).getvalue(),
        file_name="income_consumption_assets.png",
        mime="image/png",
        use_container_width=True
    )
    st.download_button(
        label="下载图像（通胀修正后购买力 PNG）",
        data=_fig_to_png_bytes(fig_inflation).getvalue(),
        file_name="inflation_adjusted_consumption.png",
        mime="image/png",
        use_container_width=True
    )

    # 简要 KPI
    st.subheader("小结（购买力口径）")
    st.write(f"通胀修正后的消费购买力总和：**{c_t_total:,.0f}**")
