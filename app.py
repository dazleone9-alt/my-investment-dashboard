import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 页面配置 ---
st.set_page_config(page_title="AlphaCopilot v6.1", layout="wide", page_icon="📈")

# --- 全局函数 ---
@st.cache_data
def get_data(tickers, period="2y"):
    if not tickers: return None
    # 自动去重并大写
    tickers = list(set([t.upper().strip() for t in tickers]))
    # 必须加入基准
    if 'SPY' not in tickers: tickers.append('SPY')
    if 'QQQ' not in tickers: tickers.append('QQQ')
    
    try:
        data = yf.download(tickers, period=period, group_by='ticker', progress=False)
        return data
    except Exception as e:
        return None

def calculate_metrics(daily_returns):
    if daily_returns.empty: return 0,0,0,0
    cagr = (1 + daily_returns.mean()) ** 252 - 1
    vol = daily_returns.std() * np.sqrt(252)
    rf = 0.04
    sharpe = (cagr - rf) / vol if vol != 0 else 0
    cum_ret = (1 + daily_returns).cumprod()
    peak = cum_ret.expanding(min_periods=1).max()
    max_dd = ((cum_ret / peak) - 1).min()
    return cagr, vol, sharpe, max_dd

# --- 主界面 ---
st.title("📈 AlphaCopilot 个人量化指挥舱")

# 使用 Tab 分隔不同功能区
tab1, tab2 = st.tabs(["💼 我的持仓 (Portfolio)", "🔍 市场扫描 (Scanner)"])

# ==========================================
# TAB 1: 我的持仓管理
# ==========================================
with tab1:
    st.sidebar.header("💼 持仓配置")
    
    # 1. 持仓输入
    default_pos = "NVDA:30, AAPL:20, MSFT:20, TSLA:15, COIN:15"
    pos_input = st.sidebar.text_area("输入持仓 (格式: 代码:比例)", default_pos, height=100)
    capital = st.sidebar.number_input("总资金 ($)", 100000, key="cap1")
    
    # 解析持仓
    try:
        portfolio_dict = {}
        valid_input = True
        if not pos_input.strip():
            valid_input = False
        else:
            for item in pos_input.split(','):
                if ':' in item:
                    k, v = item.split(':')
                    portfolio_dict[k.strip().upper()] = float(v)
                else:
                    valid_input = False
        
        if valid_input and portfolio_dict:
            # 归一化权重
            total_w = sum(portfolio_dict.values())
            weights = {k: v/total_w for k, v in portfolio_dict.items()}
            tickers = list(weights.keys())
            
            # 获取数据
            raw_data = get_data(tickers)
            
            if raw_data is not None and not raw_data.empty:
                # 提取收盘价
                close_df = pd.DataFrame()
                for t in raw_data.columns.levels[0]:
                    if 'Close' in raw_data[t]:
                        close_df[t] = raw_data[t]['Close']
                
                # 数据清洗
                close_df = close_df.ffill().dropna()
                
                if not close_df.empty:
                    # 计算收益
                    returns = close_df.pct_change().dropna()
                    
                    # 确保权重里的 key 都在数据里
                    valid_tickers = [t for t in tickers if t in returns.columns]
                    valid_weights = [weights[t] for t in valid_tickers]
                    
                    # 重新归一化
                    if sum(valid_weights) > 0:
                        valid_weights = [w/sum(valid_weights) for w in valid_weights]
                        
                        # 组合收益流
                        port_ret = returns[valid_tickers].dot(valid_weights)
                        
                        # --- 核心指标卡片 ---
                        p_cagr, p_vol, p_sharpe, p_mdd = calculate_metrics(port_ret)
                        
                        # 获取SPY数据 (如果存在)
                        if 'SPY' in returns.columns:
                            sp500_cagr, _, _, _ = calculate_metrics(returns['SPY'])
                            delta_val = f"{p_cagr-sp500_cagr:.2%} vs SPY"
                        else:
                            sp500_cagr = 0
                            delta_val = "无基准数据"

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("年化收益率", f"{p_cagr:.2%}", delta=delta_val)
                        c2.metric("夏普比率", f"{p_sharpe:.2f}")
                        c3.metric("最大回撤", f"{p_mdd:.2%}")
                        c4.metric("波动
