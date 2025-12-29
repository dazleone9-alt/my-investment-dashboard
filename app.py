import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 页面配置 ---
st.set_page_config(page_title="我的私人投资仪表盘", layout="wide")

# --- 侧边栏：输入投资组合 ---
st.sidebar.header("⚙️ 投资组合配置")

# 默认持仓 (用户可以在网页上修改)
default_tickers = "AAPL, MSFT, NVDA, TSLA, VOO"
default_weights = "0.2, 0.2, 0.2, 0.2, 0.2"
default_amount = 100000  # 初始本金

user_tickers = st.sidebar.text_input("股票代码 (用逗号分隔)", default_tickers)
user_weights = st.sidebar.text_input("对应仓位权重 (小数，用逗号分隔)", default_weights)
initial_capital = st.sidebar.number_input("总投入金额 ($)", value=default_amount)

lookback_period = st.sidebar.selectbox("回测/数据时间范围", ["1y", "3y", "5y", "ytd", "max"], index=0)

# --- 核心函数：获取数据并计算 ---
@st.cache_data # 缓存数据，避免重复下载
def get_data(tickers, benchmark_tickers, period):
    all_tickers = tickers + benchmark_tickers
    data = yf.download(all_tickers, period=period, progress=False)['Close']
    return data

def calculate_metrics(daily_returns):
    # 年化收益率 (假设252个交易日)
    cagr = (1 + daily_returns.mean()) ** 252 - 1
    # 波动率
    volatility = daily_returns.std() * np.sqrt(252)
    # 夏普比率 (假设无风险利率为 4%)
    rf = 0.04
    sharpe = (cagr - rf) / volatility
    # 最大回撤
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    return cagr, volatility, sharpe, max_drawdown

# --- 主逻辑 ---
try:
    # 1. 数据处理
    tickers_list = [x.strip().upper() for x in user_tickers.split(',')]
    weights_list = [float(x.strip()) for x in user_weights.split(',')]
    
    if len(tickers_list) != len(weights_list):
        st.error("错误：股票数量与权重数量不一致！")
        st.stop()
        
    benchmarks = ['^GSPC', '^NDX'] # 标普500 和 纳指100
    df = get_data(tickers_list, benchmarks, lookback_period)
    
    # 清洗数据
    df = df.dropna()

    # 2. 构建投资组合净值曲线
    # 计算个股日收益率
    returns = df.pct_change().dropna()
    
    # 计算组合的加权日收益率
    portfolio_returns = returns[tickers_list].dot(weights_list)
    
    # 3. 计算各个指标
    p_cagr, p_vol, p_sharpe, p_mdd = calculate_metrics(portfolio_returns)
    sp500_cagr, sp500_vol, sp500_sharpe, sp500_mdd = calculate_metrics(returns['^GSPC'])
    ndx_cagr, ndx_vol, ndx_sharpe, ndx_mdd = calculate_metrics(returns['^NDX'])

    # --- 仪表盘展示 ---
    
    st.title(f"🚀 个人投资策略分析 ({lookback_period})")
    
    # 第一行：核心指标卡片
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("年化收益率 (CAGR)", f"{p_cagr:.2%}", delta=f"{p_cagr-sp500_cagr:.2%} vs SP500")
    col2.metric("夏普比率 (Sharpe)", f"{p_sharpe:.2f}", delta=f"{p_sharpe-sp500_sharpe:.2f} vs SP500")
    col3.metric("最大回撤 (Max Drawdown)", f"{p_mdd:.2%}")
    col4.metric("波动率 (Volatility)", f"{p_vol:.2%}")

    st.markdown("---")

    # 第二行：主要图表 - 收益率走势对比
    st.subheader("📈 累计收益率对比：组合 vs 标普500 vs 纳指100")
    
    # 计算累计收益
    cum_returns = (1 + returns).cumprod()
    cum_portfolio = (1 + portfolio_returns).cumprod()
    
    # 绘图
    fig_chart = go.Figure()
    fig_chart.add_trace(go.Scatter(x=cum_portfolio.index, y=cum_portfolio, mode='lines', name='我的组合', line=dict(color='green', width=3)))
    fig_chart.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns['^GSPC'], mode='lines', name='S&P 500', line=dict(color='gray', dash='dot')))
    fig_chart.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns['^NDX'], mode='lines', name='Nasdaq 100', line=dict(color='blue', dash='dot')))
    
    fig_chart.update_layout(height=500, xaxis_title="日期", yaxis_title="净值 (起点=1)")
    st.plotly_chart(fig_chart, use_container_width=True)

    # 第三行：持仓分布
    st.subheader("💰 持仓分布与金额")
    
    # 计算当前各资产价值 (基于初始本金 + 累计涨幅)
    # 注意：这里简化处理，假设一直持有不动，实际价值需考虑再平衡
    current_prices = df.iloc[-1]
    start_prices = df.iloc[0]
    price_change_ratio = current_prices / start_prices
    
    # 估算当前各仓位金额
    asset_values = []
    for ticker, weight in zip(tickers_list, weights_list):
        val = initial_capital * weight * price_change_ratio[ticker]
        asset_values.append({'Ticker': ticker, 'Value': val})
    
    assets_df = pd.DataFrame(asset_values)
    
    col_pie, col_table = st.columns([1, 1])
    
    with col_pie:
        fig_pie = px.pie(assets_df, values='Value', names='Ticker', title='当前持仓占比')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_table:
        st.dataframe(assets_df.style.format({'Value': "${:,.2f}"}), use_container_width=True)
        st.caption(f"当前组合总市值预估: ${assets_df['Value'].sum():,.2f}")

except Exception as e:
    st.error(f"发生错误，请检查股票代码是否正确或网络连接。错误信息: {e}")