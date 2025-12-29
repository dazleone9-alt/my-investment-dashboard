import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 页面配置 ---
st.set_page_config(page_title="量化动量选股神器", layout="wide")

# --- 侧边栏：策略配置 ---
st.sidebar.header("🧠 量化策略配置")

# 1. 定义股票池 (这里预设了一些热门科技股和行业ETF，你可以随意修改)
default_pool = """AAPL, MSFT, NVDA, TSLA, GOOG, AMZN, META, NFLX, AMD, INTC, 
XLK, XLV, XLF, XLE, GLD, VOO, QQQ, SMH, ARKK, COIN"""

st.sidebar.subheader("1. 候选股票池 (Ticker Pool)")
tickers_input = st.sidebar.text_area("输入备选代码 (逗号分隔)", default_pool, height=150)

# 2. 策略参数
st.sidebar.subheader("2. 选股逻辑")
lookback_days = st.sidebar.selectbox("按过去多久的收益率排名?", 
                                     options=[30, 90, 180, 365, 730], 
                                     index=2, 
                                     format_func=lambda x: f"过去 {x} 天")

top_n = st.sidebar.slider("只持有排名前几名?", 1, 10, 5)

initial_capital = st.sidebar.number_input("虚拟本金 ($)", value=100000)

# --- 核心函数 ---
@st.cache_data
def get_data(tickers):
    # 下载足够长的数据以计算动量
    data = yf.download(tickers, period="2y", progress=False)['Close']
    return data

# --- 主逻辑 ---
try:
    # 1. 清洗输入
    pool = [x.strip().upper() for x in tickers_input.split(',') if x.strip() != '']
    pool = list(set(pool)) # 去重
    
    if len(pool) < top_n:
        st.error(f"股票池里的数量 ({len(pool)}) 少于你要选的数量 ({top_n})，请多加点股票！")
        st.stop()

    # 2. 获取数据
    with st.spinner('正在扫描市场数据，寻找最强王者...'):
        df = get_data(pool)
    
    if df is None or df.empty:
        st.error("无法获取数据，请检查代码或网络。")
        st.stop()
        
    # 清洗：去掉全是空值的列，并向前填充
    df = df.dropna(axis=1, how='all').ffill()
    
    # 3. 计算动量 (Momentum Ranking)
    # 计算“回测周期”前的价格。如果数据不够长，就取第一天。
    start_date_idx = -1 * lookback_days
    if abs(start_date_idx) > len(df):
        start_date_idx = 0
        
    current_prices = df.iloc[-1]
    past_prices = df.iloc[start_date_idx]
    
    # 计算区间收益率
    momentum_returns = (current_prices - past_prices) / past_prices
    
    # 4. 排序并选出 Top N
    # ascending=False 表示从高到低排
    ranked_assets = momentum_returns.sort_values(ascending=False)
    top_picks = ranked_assets.head(top_n)
    
    # 获取赢家的代码
    winner_tickers = top_picks.index.tolist()

    # --- 仪表盘展示 ---
    
    st.title(f"🏆 动量优选策略 (基于过去 {lookback_days} 天表现)")
    
    # 展示排名表格
    st.subheader(f"📊 表现最强的 {top_n} 只标的")
    
    # 美化表格显示
    display_df = pd.DataFrame({'代码': top_picks.index, '区间涨幅': top_picks.values})
    display_df['区间涨幅'] = display_df['区间涨幅'].apply(lambda x: f"{x:.2%}")
    
    # 颜色高亮
    col_rank, col_chart = st.columns([1, 2])
    
    with col_rank:
        st.table(display_df)
        st.success(f"系统建议当前持有：{', '.join(winner_tickers)}")

    # 5. 模拟组合表现 (假设在过去N天持有这几只最好的)
    # 注意：这是一个“事后诸葛亮”视角，展示的是这些赢家是怎么跑出来的
    winner_data = df[winner_tickers].iloc[start_date_idx:]
    
    # 归一化处理：假设起点都是 1
    normalized_growth = winner_data / winner_data.iloc[0]
    
    # 计算组合平均走势 (等权重持有)
    portfolio_curve = normalized_growth.mean(axis=1)
    
    with col_chart:
        fig = go.Figure()
        # 画个股的细线
        for ticker in winner_tickers:
            fig.add_trace(go.Scatter(x=normalized_growth.index, y=normalized_growth[ticker], 
                                     mode='lines', name=ticker, opacity=0.5, line=dict(width=1)))
        
        # 画组合的粗线
        fig.add_trace(go.Scatter(x=portfolio_curve.index, y=portfolio_curve, 
                                 mode='lines', name='优选组合 (平均)', 
                                 line=dict(color='red', width=4)))
        
        fig.update_layout(title="赢家组合走势回顾", yaxis_title="净值增长 (1 = 起点)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # 6. 具体持仓建议
    st.markdown("---")
    st.subheader("💰 建议调仓方案")
    
    # 假设等权重买入
    weight_per_stock = 1.0 / top_n
    money_per_stock = initial_capital * weight_per_stock
    
    suggested_shares = []
    latest_prices = df[winner_tickers].iloc[-1]
    
    for ticker in winner_tickers:
        price = latest_prices[ticker]
        shares = money_per_stock / price
        suggested_shares.append({
            '代码': ticker,
            '当前价格': f"${price:.2f}",
            '分配金额': f"${money_per_stock:,.0f}",
            '建议买入股数': f"{shares:.2f} 股"
        })
        
    st.dataframe(pd.DataFrame(suggested_shares))

except Exception as e:
    st.error(f"发生错误: {e}")
    st.info("提示：如果股票池太大，可能会导致Yahoo API超时，请尝试减少一些备选股票。")
