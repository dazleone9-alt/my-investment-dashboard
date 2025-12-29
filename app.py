import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 页面配置 ---
st.set_page_config(page_title="全能多因子量化工作台", layout="wide")

# --- 侧边栏：因子工厂 ---
st.sidebar.header("🎛️ 因子选股工厂")

# 1. 策略选择
strategy_map = {
    "🔥 动量因子 (Momentum)": "寻找强势股 (强者恒强)",
    "🛡️ 低波因子 (Low Volatility)": "寻找稳健股 (抗跌防御)",
    "🎯 高贝塔因子 (High Beta)": "寻找高弹性股 (牛市急先锋)",
    "🐢 低贝塔因子 (Low Beta)": "寻找低相关股 (熊市避风港)",
    "💰 流动性因子 (Liquidity)": "寻找资金拥挤股 (热门成交)",
    "🎣 反转因子 (RSI Reversal)": "寻找超卖股 (短线抄底)"
}

selected_strategy = st.sidebar.selectbox("1. 选择选股因子", list(strategy_map.keys()))
st.sidebar.info(f"策略逻辑：{strategy_map[selected_strategy]}")

# 2. 股票池
default_pool = """AAPL, MSFT, NVDA, TSLA, GOOG, AMZN, META, NFLX, AMD, INTC,
XLK, XLV, XLF, XLE, GLD, VOO, QQQ, SMH, ARKK, COIN,
JPM, BAC, WMT, COST, KO, PEP, JNJ, PFE, XOM, CVX"""

st.sidebar.subheader("2. 股票池配置")
tickers_input = st.sidebar.text_area("输入股票池 (逗号分隔)", default_pool, height=120)

# 3. 参数设置
st.sidebar.subheader("3. 回测参数")
lookback_days = st.sidebar.slider("计算周期 (天)", 30, 365, 90)
top_n = st.sidebar.slider("优选数量", 1, 10, 5)
initial_capital = st.sidebar.number_input("虚拟本金", value=100000)

# --- 核心计算函数 ---
@st.cache_data
def get_data(tickers):
    # 多下载一些数据用于计算技术指标，并必须包含 SPY 作为市场基准
    all_tickers = list(set(tickers + ['SPY']))
    data = yf.download(all_tickers, period="2y", group_by='ticker', progress=False)
    return data

def calculate_beta(stock_returns, market_returns):
    # 计算 Beta: Cov(s, m) / Var(m)
    covariance = np.cov(stock_returns, market_returns)[0][1]
    variance = np.var(market_returns)
    return covariance / variance if variance != 0 else 0

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 主逻辑 ---
try:
    pool = [x.strip().upper() for x in tickers_input.split(',') if x.strip()]
    
    if len(pool) < top_n:
        st.error("股票池数量太少，无法选股！")
        st.stop()
        
    with st.spinner('正在进行多因子量化运算...'):
        # 获取 OHLCV 数据
        raw_data = get_data(pool)
    
    if raw_data is None or raw_data.empty:
        st.error("数据获取失败。")
        st.stop()

    # 提取 Close 和 Volume
    # yfinance multi-level columns 处理
    close_df = pd.DataFrame()
    volume_df = pd.DataFrame()
    
    for t in raw_data.columns.levels[0]:
        if 'Close' in raw_data[t]:
            close_df[t] = raw_data[t]['Close']
        if 'Volume' in raw_data[t]:
            volume_df[t] = raw_data[t]['Volume']
            
    close_df = close_df.ffill().dropna(axis=1, how='all')
    
    # 截取回测时间段
    start_idx = -1 * lookback_days
    if abs(start_idx) > len(close_df): start_idx = 0
    subset = close_df.iloc[start_idx:]
    
    # 准备基准数据 (SPY)
    spy_returns = subset['SPY'].pct_change().dropna() if 'SPY' in subset else None
    
    # --- 因子计算引擎 ---
    scores = {}
    
    # 排除 SPY 自身参与排名
    ranking_pool = [t for t in pool if t in subset.columns and t != 'SPY']
    
    for ticker in ranking_pool:
        series = subset[ticker]
        daily_ret = series.pct_change().dropna()
        
        if "动量" in selected_strategy:
            # 动量：区间涨幅
            scores[ticker] = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
            ascending_order = False # 越大越好
            col_name = "区间涨幅"
            fmt = "{:.2%}"
            
        elif "低波" in selected_strategy:
            # 波动率：标准差
            scores[ticker] = daily_ret.std() * np.sqrt(252)
            ascending_order = True # 越小越好
            col_name = "年化波动率"
            fmt = "{:.2%}"
            
        elif "贝塔" in selected_strategy:
            # Beta 计算
            if spy_returns is not None:
                # 对齐数据长度
                common_idx = daily_ret.index.intersection(spy_returns.index)
                beta = calculate_beta(daily_ret.loc[common_idx], spy_returns.loc[common_idx])
                scores[ticker] = beta
            else:
                scores[ticker] = 0
            
            if "高贝塔" in selected_strategy:
                ascending_order = False # 越大越弹性
                col_name = "Beta系数"
            else:
                ascending_order = True # 越小越独立
                col_name = "Beta系数"
            fmt = "{:.2f}"
            
        elif "流动性" in selected_strategy:
            # 流动性：平均成交金额 (Close * Volume)
            vol_series = volume_df[ticker].iloc[start_idx:]
            avg_turnover = (series * vol_series).mean()
            scores[ticker] = avg_turnover
            ascending_order = False # 越大越活跃
            col_name = "日均成交额($)"
            fmt = "${:,.0f}"

        elif "反转" in selected_strategy:
            # RSI 反转：寻找 RSI 低于 30 的或者最低的
            rsi = calculate_rsi(series).iloc[-1]
            scores[ticker] = rsi
            ascending_order = True # RSI越低越超卖
            col_name = "当前RSI(14)"
            fmt = "{:.2f}"

    # --- 排名与筛选 ---
    scores_series = pd.Series(scores)
    top_picks = scores_series.sort_values(ascending=ascending_order).head(top_n)
    winner_tickers = top_picks.index.tolist()
    
    # --- 仪表盘展示 ---
    st.title(f"🔍 量化选股报告：{selected_strategy.split(' ')[1]}")
    
    # 1. 选股结果表
    st.subheader(f"🏆 因子选股 Top {top_n}")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        res_df = pd.DataFrame({'代码': top_picks.index, col_name: top_picks.values})
        res_df[col_name] = res_df[col_name].apply(lambda x: fmt.format(x))
        st.table(res_df)
        
        # 投资建议
        total_weight = 1.0
        money = initial_capital * total_weight / top_n
        st.info(f"💡 建议操作：将 ${initial_capital:,.0f} 平均分配，每只股票买入约 ${money:,.0f}。")

    # 2. 模拟走势图
    with c2:
        # 归一化对比
        winner_data = subset[winner_tickers]
        normalized = winner_data / winner_data.iloc[0]
        
        # 组合曲线
        portfolio_curve = normalized.mean(axis=1)
        
        fig = go.Figure()
        # 个股轻色线
        for t in winner_tickers:
            fig.add_trace(go.Scatter(x=normalized.index, y=normalized[t], mode='lines', name=t, opacity=0.3))
        
        # 组合重色线
        fig.add_trace(go.Scatter(x=portfolio_curve.index, y=portfolio_curve, mode='lines', name='优选组合', line=dict(color='#FF4B4B', width=3)))
        
        # SPY 基准线
        if 'SPY' in subset:
            spy_norm = subset['SPY'] / subset['SPY'].iloc[0]
            fig.add_trace(go.Scatter(x=spy_norm.index, y=spy_norm, mode='lines', name='标普500 (基准)', line=dict(color='gray', dash='dot')))

        fig.update_layout(title="优选组合 vs 市场基准 (同期走势)", yaxis_title="净值 (起点=1)", height=450)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"程序运行出错: {e}")
    st.markdown("建议：可能是网络问题或股票代码输入有误，请刷新页面重试。")
