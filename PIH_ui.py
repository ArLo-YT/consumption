import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置绘图样式
from matplotlib import font_manager, rcParams
import os
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ==========================
# 字体与全局样式
# ==========================
found_font = None
if found_font:
    rcParams['font.family'] = found_font
else:
    # 如果系统没中文字体，则加载项目自带字体
    font_path = os.path.join("simhei.ttf")  # 确保你把 SimHei.ttf 放在项目根目录
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

# 确保负号正常显示
rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 14})

# 侧边栏展示 README.md（只显示到 "## 环境依赖" 前）
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

# ==========================
# 工具函数（美化坐标轴等）
# ==========================

def _thousands(x, pos):
    return f"{x:,.0f}"


def _beautify_axes(ax, y_ticks=6):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.8, alpha=0.35)
    ax.grid(False, axis='x')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.tick_params(axis='both', labelsize=13)
    ax.margins(x=0.02, y=0.08)


def _annotate_last(ax, xs, ys):
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

# ==========================
# 经济学计算函数
# ==========================

def coeff(r, s):
    if np.abs(r) > 0.0001:
        coefficient = r / (1 + r) / (1 - 1 / np.power(1 + r, s))
    else:
        coefficient = 1 / s
    return coefficient


def consump(r, s, A_t, Y_t, r_e=None, f_t=0, r_ins=0):
    if r_e is None:
        r_e = r
    c_t = coeff(r_e - r_ins, s) * ((A_t - f_t / (1 + r) ** s) * (1 + r) + Y_t)
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
    形如： 1:100000, 5:200000
    """
    income_dict = {}
    if input_str is None or str(input_str).strip() == "":
        return income_dict

    input_str = str(input_str).strip().strip('\'\"')
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
        except Exception:
            continue
    return income_dict


# ==========================
# 核心模拟（不直接在函数里使用 st.*，返回消息，避免警告被刷新）
# ==========================

def simulate_and_output(years, wage, A_t_init, r_c, l, grow_rate, final_wealth, r_ins, custom_income_str):
    y_t = income(wage, grow_rate, years)

    # 自定义收入覆盖
    custom_income = parse_custom_income(custom_income_str, years)
    for year_idx in custom_income:
        y_t[year_idx] = custom_income[year_idx]

    rows = []
    A_t = A_t_init
    c_t_list, A_t_list = [], []
    c_t_inflation_adjusted = []  # 购买力折算

    dr = 0
    r = r_c

    A_t_min = 0
    A_t_max = 0
    A_t_warn = 0

    fatal_msg = None  # 如果消费为负，返回致命信息

    for i in range(years):
        r_e = r - l
        year_label = i + 1
        initial_A = A_t
        s = years - i
        Y_t = pv(y_t[i:years], r + dr)
        c_t = consump(r, s, A_t, Y_t, r_e, final_wealth, r_ins + dr)

        # 若消费为负，直接返回提示，不再继续计算
        if c_t < 0:
            fatal_msg = "很遗憾，当前收入难以支持目标资产，请增加收入或下调最终目标/偏好。"
            break

        A_t = A_t * (1 + r) + y_t[i] - c_t

        # 通胀折算（购买力）
        consumption_inflation_adjusted = c_t / ((1 + l) ** i)
        c_t_inflation_adjusted.append(consumption_inflation_adjusted)

        rows.append({
            "年份": year_label,
            "年初资产": round(initial_A, 2),
            "年内收入": round(y_t[i], 2),
            "全年消费": round(c_t, 2),
            "年末资产": round(A_t, 2)
        })

        c_t_list.append(c_t)
        A_t_list.append(A_t)

        # 资产正负穿越提示
        if A_t < A_t_min:
            A_t_min = A_t
        if A_t > A_t_max:
            A_t_max = A_t
        if A_t_max > 0.0001 and A_t_min < -0.0001:
            A_t_warn = 1

    # 若致命错误，提前返回
    if fatal_msg is not None:
        return None, None, None, None, {"fatal": fatal_msg, "warn_net_assets": False}

    # 汇总
    results = pd.DataFrame(rows, columns=["年份", "年初资产", "年内收入", "全年消费", "年末资产"])

    # 图 1：收入&消费 + 年末资产
    fig, axs = plt.subplots(2, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    time = list(range(1, years + 1))

    axs[0].plot(time, y_t, linewidth=2.2, marker='o', markersize=4, label='全年收入')
    axs[0].plot(time, c_t_list, linewidth=2.2, marker='o', markersize=4, label='全年消费')
    axs[0].set_title('未来收入与消费', fontsize=18, pad=12)
    axs[0].set_ylabel('金额', fontsize=14)
    _beautify_axes(axs[0])
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    _annotate_last(axs[0], time, y_t)
    _annotate_last(axs[0], time, c_t_list)
    axs[0].legend(frameon=False, fontsize=13, ncols=2, loc='upper left')

    axs[1].plot(time, A_t_list, linewidth=2.2, marker='o', markersize=4)
    axs[1].set_title('年末资产', fontsize=18, pad=12)
    axs[1].set_xlabel('年份', fontsize=14)
    axs[1].set_ylabel('金额', fontsize=14)
    axs[1].axhline(0, linewidth=1.0, linestyle='--', alpha=0.4)
    _beautify_axes(axs[1])
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    _annotate_last(axs[1], time, A_t_list)

    # 图 2：通胀修正后消费购买力
    fig_inflation, ax_inflation = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax_inflation.plot(time, c_t_inflation_adjusted, linewidth=2.2, marker='o', markersize=4)
    ax_inflation.set_title("通胀修正后消费购买力（以第1年物价为基准）", fontsize=18, pad=10)
    ax_inflation.set_xlabel("年份", fontsize=14)
    ax_inflation.set_ylabel("购买力", fontsize=14)
    _beautify_axes(ax_inflation, y_ticks=5)
    ax_inflation.xaxis.set_major_locator(MaxNLocator(integer=True))
    _annotate_last(ax_inflation, time, c_t_inflation_adjusted)

    c_t_inflation_adjusted_total = float(sum(c_t_inflation_adjusted))

    return fig, results, fig_inflation, c_t_inflation_adjusted_total, {
        "fatal": None,
        "warn_net_assets": bool(A_t_warn)
    }


# ==========================
# 预设场景
# ==========================
PRESETS = {
    "应届毕业生起步": {
        "years": 15,
        "A_t_init": 20000.0,
        "final_wealth": 500000.0,
        "wage": 80000.0,
        "grow_rate": 0.06,   # 6%
        "r_c": 0.02,         # 2%
        "l": 0.01,           # 1%
        "r_ins": 0.00,       # 0%
        "custom_income_str": ""
    },
    "房贷+育儿期": {
        "years": 20,
        "A_t_init": -1000000.0,
        "final_wealth": 0,
        "wage": 180000.0,
        "grow_rate": 0.03,
        "r_c": 0.05,
        "l": 0.025,
        "r_ins": -0.005,      # 偏好当下消费
        "custom_income_str": ""
    },
    "临近退休": {
        "years": 20,
        "A_t_init": 800000.0,
        "final_wealth": 500000.0,
        "wage": 150000.0,
        "grow_rate": 0.01,
        "r_c": 0.03,
        "l": 0.015,
        "r_ins": 0.005,      # 偏好未来消费
        "custom_income_str": ""
    },
    "天降横财": {
        "years": 30,
        "A_t_init": 100000.0,
        "final_wealth": 350000.0,
        "wage": 80000.0,
        "grow_rate": 0.00,
        "r_c": 0.02,
        "l": 0.01,
        "r_ins": 0.005,
        # 第 7 年获得 100 万的一次性横财（注意：年份从 1 开始）
        "custom_income_str": "7:1080000"
    },
    "坐吃山空": {
        "years": 40,
        "A_t_init": 5000000.0,
        "final_wealth": 1000000.0,
        "wage": 0.01,
        "grow_rate": 0.00,
        "r_c": 0.03,
        "l": 0.02,
        "r_ins": 0.005,
        "custom_income_str": ""
    },
}


def _load_preset_to_state(preset: dict):
    for k, v in preset.items():
        st.session_state[k] = v

# 首次默认值
for k, v in {
    "years": 30, "A_t_init": 0.0, "final_wealth": 0.0, "wage": 100000.0,
    "grow_rate": 0.01, "r_c": 0.03, "l": 0.02, "r_ins": 0.0,
    "custom_income_str": ""
}.items():
    st.session_state.setdefault(k, v)

# 结果缓存位（避免 st.rerun 导致消息丢失）
st.session_state.setdefault("show_results", False)
st.session_state.setdefault("out_df", None)
st.session_state.setdefault("out_png_main", None)
st.session_state.setdefault("out_png_infl", None)
st.session_state.setdefault("out_csv", None)
st.session_state.setdefault("warn_net_assets", False)
st.session_state.setdefault("fatal_msg", None)

# ==========================
# 页面布局
# ==========================

st.title("消费规划模拟工具")

# 顶部操作条：开始计算按钮上移 + 载入预设
op_col1, op_col2 = st.columns([1, 1])

with op_col1:
    st.markdown("<div class='op-card'>", unsafe_allow_html=True)
    st.subheader("默认例子")
    preset_name = st.selectbox("选择一个场景", list(PRESETS.keys()), index=0)
    if st.button("一键填充该场景", use_container_width=True):
        _load_preset_to_state(PRESETS[preset_name])
        st.success(f"已载入预设：{preset_name}")
        st.caption("提示：载入后你仍可在下方继续调整所有参数。")
    st.markdown("</div>", unsafe_allow_html=True)

with op_col2:
    st.markdown("<div class='op-card'>", unsafe_allow_html=True)
    st.subheader("操作")
    run_clicked = st.button("▶️ 开始计算", type="primary", use_container_width=True)
    if st.button("🔄 清空结果", use_container_width=True):
        st.session_state.update({
            "show_results": False,
            "out_df": None,
            "out_png_main": None,
            "out_png_infl": None,
            "out_csv": None,
            "warn_net_assets": False,
            "fatal_msg": None,
        })
    # 可选：一个小的占位，确保内容贴顶且视觉平衡


st.divider()  # —— 输入与结果的清晰分界线 ——

# ========== 参数输入区 ==========
with st.container():
    st.header("🧮 参数设置", anchor=False)

    years = st.number_input("周期（最小5年，最大80年）", min_value=5, max_value=80,
                            value=int(st.session_state["years"]), key="years")
    A_t_init = st.number_input("当前资产（可为负数）", value=float(st.session_state["A_t_init"]), key="A_t_init")
    final_wealth = st.number_input("最终目标资产（大于等于0）", min_value=0,
                                   value=int(st.session_state["final_wealth"]), key="final_wealth")
    wage = st.number_input("当前年薪（大于0）", min_value=0.01,
                           value=float(st.session_state["wage"]), key="wage")

    grow_rate = st.slider("预期薪水增长率（-5% 到20%）",
                          min_value=-5.0, max_value=20.0,
                          value=float(st.session_state["grow_rate"]*100), step=0.1, key="grow_rate_pct") / 100
    st.session_state["grow_rate"] = grow_rate

    r_c = st.slider("年利率（最大20%，当前周期内负债较多请用贷款利率）",
                    min_value=0.0, max_value=20.0,
                    value=float(st.session_state["r_c"]*100), step=0.1, key="r_c_pct") / 100
    st.session_state["r_c"] = r_c

    l = st.slider("通胀率（-5% 到20%）",
                  min_value=-5.0, max_value=20.0,
                  value=float(st.session_state["l"]*100), step=0.1, key="l_pct") / 100
    st.session_state["l"] = l

    r_ins = st.slider("消费偏好修正（-5% 到5%，+表示偏好未来消费，-表示偏好当下消费）",
                      min_value=-5.0, max_value=5.0,
                      value=float(st.session_state["r_ins"]*100), step=0.1, key="r_ins_pct") / 100
    st.session_state["r_ins"] = r_ins

    st.subheader("自定义某些年份的收入（编号从1开始，例如： '1:100000, 5:200000'）")
    custom_income_input = st.text_area("输入自定义收入", height=100,
                                       value=st.session_state["custom_income_str"], key="custom_income_str")

# 触发计算（按钮在顶部 op_col2 中），这里读取 run_clicked 状态
if 'run_clicked' not in st.session_state:
    st.session_state['run_clicked'] = False

if run_clicked:    
    # 点击“开始计算”时，先清空旧结果，确保不会展示上一次的输出
    st.session_state.update({
        "show_results": False,
        "out_df": None,
        "out_png_main": None,
        "out_png_infl": None,
        "out_csv": None,
        "warn_net_assets": False,
        "fatal_msg": None,
    })
    fig, df_results, fig_inflation, c_t_total, msgs = simulate_and_output(
        years, wage, A_t_init, r_c, l, grow_rate, final_wealth, r_ins, custom_income_input
    )

    # 发生致命错误（如消费为负）：仅记录消息，不清空之前结果
    st.session_state["fatal_msg"] = msgs.get("fatal")
    st.session_state["warn_net_assets"] = msgs.get("warn_net_assets", False)

    if fig is not None:
        from io import BytesIO

        def _fig_to_png_bytes(f):
            buf = BytesIO()
            f.savefig(buf, format="png", dpi=180, bbox_inches="tight")
            buf.seek(0)
            return buf.getvalue()

        st.session_state["out_df"] = df_results
        st.session_state["out_png_main"] = _fig_to_png_bytes(fig)
        st.session_state["out_png_infl"] = _fig_to_png_bytes(fig_inflation)
        st.session_state["out_csv"] = df_results.to_csv(index=False).encode("utf-8-sig")
        st.session_state["show_results"] = True

# 清晰分界：结果区
st.divider()

with st.container():
    st.header("📊 模拟结果", anchor=False)

    # 致命提示（例如消费为负）
    if st.session_state.get("fatal_msg"):
        st.error(st.session_state["fatal_msg"])

    if st.session_state.get("show_results") and st.session_state.get("out_df") is not None:
        st.subheader("每年收入、消费和资产记录")
        st.dataframe(st.session_state["out_df"], use_container_width=True)

        st.subheader("图表")
        if st.session_state.get("out_png_main") is not None:
            st.image(st.session_state["out_png_main"], use_container_width=True, caption="收入-消费-资产")
        if st.session_state.get("out_png_infl") is not None:
            st.image(st.session_state["out_png_infl"], use_container_width=True, caption="通胀修正后的购买力")

        st.subheader("下载导出")
        st.download_button(
            label="下载数据（CSV）",
            data=st.session_state["out_csv"],
            file_name="消费规划结果.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_csv_cached"
        )
        st.download_button(
            label="下载图像（收入-消费-资产 PNG）",
            data=st.session_state["out_png_main"],
            file_name="income_consumption_assets.png",
            mime="image/png",
            use_container_width=True,
            key="dl_png_main_cached"
        )
        st.download_button(
            label="下载图像（通胀修正后购买力 PNG）",
            data=st.session_state["out_png_infl"],
            file_name="inflation_adjusted_consumption.png",
            mime="image/png",
            use_container_width=True,
            key="dl_png_infl_cached"
        )

    # 非致命警告：资产正负穿越
    if st.session_state.get("warn_net_assets", False):
        st.markdown(
            """
<div style="
    padding: 12px 14px;
    border-left: 6px solid #f59e0b;
    background: rgba(255, 215, 0, 0.12);
    border-radius: 6px;">
  <strong>⚠️ 提示：</strong> 当前周期内同时出现净资产和净负债，相应的，也应同时存在存款和贷款利率，
  因此结果会略有偏差，仅供参考。
</div>
""",
            unsafe_allow_html=True,
        )

