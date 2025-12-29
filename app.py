import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 页面配置 ---
st.set_page_config(page_title="多因子量化选股平台", layout="wide")

# --- 侧边栏：策略配置 ---
st.sidebar.header("🧠 策略中控台")

# 1. 策略选择 (新增功能)
strategy_type = st.sidebar.radio(
    "1. 请选择你的战术🎯",
    ("进攻型：动量策略 (Momentum)", "防守型：低波策略 (Low Volatility)")
)

# 2. 股票池
default_pool = """AAPL, MSFT, NVDA, TSLA, GOOG, AMZN, META, NFLX, AMD, 
XLK, XLV, XLF, XLE, GLD, VOO, QQQ, SMH, JNJ, PG, KO, PEP, MCD, V, MA"""
st.sidebar.subheader("2. 股票池")
tickers_input = st.sidebar.text_area("输入代码 (逗号分隔)", default_pool, height=100)

# 3. 参数
st.sidebar.subheader("3. 参数设置")
lookback_days = st.sidebar.selectbox("回测周期", [30, 90, 180, 365], index=2, format_func=lambda x: f"过去 {x} 天")
top_n = st.sidebar.slider("选出几只股票?", 1, 10, 5)
initial_capital = st.sidebar.number_input("虚拟本金", value=100000)

# --- 核心逻辑 ---
@st.cache_data
def get_data(tickers):
    data = yf.download(tickers, period="2y", progress=False)['Close']
    return data

try:
    pool = [x.strip().upper() for x in tickers_input.split(',') if x.strip()]
    pool = list(set(pool))
    
    with st.spinner('正在量化计算中...'):
        df = get_data(pool)
    
    if df is None or df.empty:
        st.error("数据获取失败")
        st.stop()
        
    df = df.dropna(axis=1, how='all').ffill()
    
    # 确定计算的起止时间
    start_idx = -1 * lookback_days
    if abs(start_idx) > len(df): start_idx = 0
    
    subset = df.iloc[start_idx:]
    
    # --- 策略分流核心代码 ---
    
    if "进攻型" in strategy_type:
        # 策略 A：动量 (计算区间涨幅)
        metric_name = "区间涨幅"
        start_price = subset.iloc[0]
        end_price = subset.iloc[-1]
        scores = (end_price - start_price) / start_price
        # 涨幅越大越好 -> 降序排列
        top_picks = scores.sort_values(ascending=False).head(top_n)
        st.success(f"🚀 当前策略逻辑：寻找过去 {lookback_days} 天涨势最猛的 {top_n} 只股票 (强者恒强)")
        
    else:
        # 策略 B：低波动 (计算标准差/波动率)
        metric_name = "波动率 (越低越稳)"
        # 计算日收益率
        daily_returns = subset.pct_change().dropna()
        # 计算波动率 (标准差)
        scores = daily_returns.std()
        # 波动越小越好 -> 升序排列
        top_picks = scores.sort_values(ascending=True).head(top_n)
        st.info(f"🛡️ 当前策略逻辑：寻找过去 {lookback_days} 天震荡最小的 {top_n} 只股票 (避险抗跌)")

    # --- 展示结果 ---
    winner_tickers = top_picks.index.tolist()
    
    # 图表数据准备
    winner_data = subset[winner_tickers]
    normalized = winner_data / winner_data.iloc[0]
    portfolio_curve = normalized.mean(axis=1)

    # 布局
    st.title(f"策略分析报告：{strategy_type.split('：')[0]}")
    
    col_table, col_chart = st.columns([1, 2])
    
    with col_table:
        st.subheader("选股结果")
        display_df = pd.DataFrame({'代码': top_picks.index, metric_name: top_picks.values})
        # 格式化数字
        if "进攻" in strategy_type:
            display_df[metric_name] = display_df[metric_name].apply(lambda x: f"+{x:.2%}")
        else:
            display_df[metric_name] = display_df[metric_name].apply(lambda x: f"{x:.4f}")
            
        st.table(display_df)
        
    with col_chart:
        fig = go.Figure()
        for t in winner_tickers:
            fig.add_trace(go.Scatter(x=normalized.index, y=normalized[t], mode='lines', name=t, opacity=0.3))
        fig.add_trace(go.Scatter(x=portfolio_curve.index, y=portfolio_curve, mode='lines', name='组合净值', line=dict(color='red', width=3)))
        fig.update_layout(title="组合回测走势 (归一化)", yaxis_title="净值", height=450)
        st.plotly_chart(fig, use_container_width=True)

    # 调仓建议
    st.markdown("---")
    st.subheader("💰 调仓指令")
    money_per = initial_capital / top_n
    latest_p = df[winner_tickers].iloc[-1]
    
    buy_list = []
    for t in winner_tickers:
        buy_list.append({
            '标的': t,
            '最新价': f"${latest_p[t]:.2f}",
            '建议买入': f"{money_per/latest_p[t]:.2f} 股"
        })
    st.dataframe(pd.DataFrame(buy_list), use_container_width=True)

except Exception as e:
    st.error(f"出错: {e}")
