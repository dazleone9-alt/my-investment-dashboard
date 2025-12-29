import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 页面配置 ---
st.set_page_config(page_title="我的私人投资仪表盘", layout="wide")

# --- 侧边栏：输入投资组合 ---
st.sidebar.header("⚙️ 投资组合配置")

# 默认持仓
default_tickers = "AAPL, MSFT, NVDA, TSLA, VOO"
default_weights = "0.2, 0.2, 0.2, 0.2, 0.2"
default_amount = 100000 

user_tickers = st.sidebar.text_input("股票代码 (用逗号分隔)", default_tickers)
user_weights = st.sidebar.text_input("对应仓位权重 (小数，用逗号分隔)", default_weights)
initial_capital = st.sidebar.number_input("总投入金额 ($)", value=default_amount)
lookback_period = st.sidebar.selectbox("回测/数据时间范围", ["1y", "3y", "5y", "ytd", "max"], index=0)

# --- 核心函数 ---
@st.cache_data
def get_data(tickers, benchmark_tickers, period):
    all_tickers = tickers + benchmark_tickers
    # 尝试下载数据
    try:
        data = yf.download(all_tickers, period=period, progress=False)['Close']
        return data
    except Exception as e:
        return None

def calculate_metrics(daily_returns):
    if daily_returns.empty:
        return 0, 0, 0, 0
    cagr = (1 + daily_returns.mean()) ** 252 - 1
    volatility = daily_returns.std() * np.sqrt(252)
    rf = 0.04
    sharpe = (cagr - rf) / volatility if volatility != 0 else 0
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    return cagr, volatility, sharpe, max_drawdown

# --- 主逻辑 ---
try:
    tickers_list = [x.strip().upper() for x in user_tickers.split(',')]
    weights_list = [float(x.strip()) for x in user_weights.split(',')]
    
    if len(tickers_list) != len(weights_list):
        st.error(f"⚠️ 错误：股票数量({len(tickers_list)}) 与 权重数量({len(weights_list)}) 不一致！")
        st.stop()
        
    benchmarks = ['^GSPC', '^NDX'] 
    
    # 1. 获取数据
    with st.spinner('正在从华尔街抓取数据...'):
        df = get_data(tickers_list, benchmarks, lookback_period)
    
    # 2. 数据有效性检查 (关键修复步骤)
    if df is None or df.empty:
        st.error("❌ 无法获取数据。可能原因：1.股票代码错误 2.网络超时 3.Yahoo数据源暂时不可用。请尝试刷新页面。")
        st.stop()

    # 修复：先填充空缺数据(ffill)，再去除由于刚上市等原因导致的真正空值
    df = df.ffill().dropna()

    if df.empty:
        st.error("❌ 数据清洗后为空。这通常是因为某个股票在选定时间段内没有数据。建议检查代码或缩短时间范围。")
        st.stop()

    # 3. 计算收益率
    returns = df.pct_change().dropna()
    
    # 确保所有代码都在数据列中
    available_tickers = [t for t in tickers_list if t in returns.columns]
    if len(available_tickers) != len(tickers_list):
        missing = set(tickers_list) - set(available_tickers)
        st.warning(f"⚠️ 以下股票数据缺失，已自动忽略: {missing}")
        # 重新调整权重 (归一化)
        valid_indices = [i for i, t in enumerate(tickers_list) if t in available_tickers]
        available_tickers = [tickers_list[i] for i in valid_indices]
        valid_weights = [weights_list[i] for i in valid_indices]
        total_weight = sum(valid_weights)
        if total_weight == 0:
            st.error("剩余有效资产权重为0")
            st.stop()
        weights_list = [w/total_weight for w in valid_weights]
        tickers_list = available_tickers

    portfolio_returns = returns[tickers_list].dot(weights_list)
    
    # 4. 指标计算
    p_cagr, p_vol, p_sharpe, p_mdd = calculate_metrics(portfolio_returns)
    sp500_cagr, sp500_vol, sp500_sharpe, sp500_mdd = calculate_metrics(returns['^GSPC']) if '^GSPC' in returns else (0,0,0,0)
    
    # --- 仪表盘展示 ---
    st.title(f"🚀 个人投资策略分析 ({lookback_period})")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("年化收益率", f"{p_cagr:.2%}", delta=f"{p_cagr-sp500_cagr:.2%}")
    col2.metric("夏普比率", f"{p_sharpe:.2f}", delta=f"{p_sharpe-sp500_sharpe:.2f}")
    col3.metric("最大回撤", f"{p_mdd:.2%}")
    col4.metric("波动率", f"{p_vol:.2%}")

    st.markdown("---")
    
    st.subheader("📈 净值走势")
    cum_returns = (1 + returns).cumprod()
    cum_portfolio = (1 + portfolio_returns).cumprod()
    
    fig_chart = go.Figure()
    fig_chart.add_trace(go.Scatter(x=cum_portfolio.index, y=cum_portfolio, mode='lines', name='我的组合', line=dict(color='#00CC96', width=3)))
    if '^GSPC' in cum_returns:
        fig_chart.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns['^GSPC'], mode='lines', name='S&P 500', line=dict(color='gray', dash='dot')))
    if '^NDX' in cum_returns:
        fig_chart.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns['^NDX'], mode='lines', name='Nasdaq 100', line=dict(color='blue', dash='dot')))
    
    fig_chart.update_layout(height=500, xaxis_title="", yaxis_title="净值 (起点=1)")
    st.plotly_chart(fig_chart, use_container_width=True)

    # 持仓分布
    st.subheader("💰 当前持仓估值")
    current_prices = df.iloc[-1]
    start_prices = df.iloc[0]
    price_ratio = current_prices / start_prices
    
    asset_values = []
    for ticker, weight in zip(tickers_list, weights_list):
        if ticker in price_ratio:
            val = initial_capital * weight * price_ratio[ticker]
            asset_values.append({'Ticker': ticker, 'Value': val})
            
    assets_df = pd.DataFrame(asset_values)
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(px.pie(assets_df, values='Value', names='Ticker', hole=0.4), use_container_width=True)
    with c2:
        st.dataframe(assets_df.style.format({'Value': "${:,.2f}"}), use_container_width=True)

except Exception as e:
    st.error(f"程序运行出错: {e}")
