import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 页面配置 ---
st.set_page_config(page_title="AlphaCopilot v6.0", layout="wide", page_icon="📈")

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
    sharpe = (cagr - 0.04) / vol if vol != 0 else 0
    cum_ret = (1 + daily_returns).cumprod()
    max_dd = ((cum_ret / cum_ret.expanding().max()) - 1).min()
    return cagr, vol, sharpe, max_dd

# --- 主界面 ---
st.title("📈 AlphaCopilot 个人量化指挥舱")

# 使用 Tab 分隔不同功能区
tab1, tab2 = st.tabs(["💼 我的持仓 (Portfolio)", "🔍 市场扫描 (Scanner)"])

# ==========================================
# TAB 1: 我的持仓管理 (满足需求 2,3,4,5,6,7)
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
        for item in pos_input.split(','):
            k, v = item.split(':')
            portfolio_dict[k.strip().upper()] = float(v)
        
        # 归一化权重
        total_w = sum(portfolio_dict.values())
        weights = {k: v/total_w for k, v in portfolio_dict.items()}
        tickers = list(weights.keys())
        
        # 获取数据
        raw_data = get_data(tickers)
        
        if raw_data is not None:
            # 提取收盘价
            close_df = pd.DataFrame()
            for t in raw_data.columns.levels[0]:
                if 'Close' in raw_data[t]:
                    close_df[t] = raw_data[t]['Close']
            close_df = close_df.ffill().dropna()
            
            # 计算收益
            returns = close_df.pct_change().dropna()
            
            # 组合收益流
            port_ret = returns[tickers].dot(list(weights.values()))
            
            # --- 核心指标卡片 ---
            p_cagr, p_vol, p_sharpe, p_mdd = calculate_metrics(port_ret)
            sp500_cagr, _, _, _ = calculate_metrics(returns['SPY'])
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("年化收益率", f"{p_cagr:.2%}", delta=f"{p_cagr-sp500_cagr:.2%} vs SPY")
            c2.metric("夏普比率", f"{p_sharpe:.2f}")
            c3.metric("最大回撤", f"{p_mdd:.2%}")
            c4.metric("波动率", f"{p_vol:.2%}")
            
            st.divider()
            
            # --- 图表区 ---
            col_chart, col_alloc = st.columns([2, 1])
            
            with col_chart:
                st.subheader("📈 收益率走势 (含基准对比)")
                
                # 净值计算
                cum_port = (1 + port_ret).cumprod()
                cum_spy = (1 + returns['SPY']).cumprod()
                cum_qqq = (1 + returns['QQQ']).cumprod()
                
                fig = go.Figure()
                
                # 定义画线函数，包含【需求7：尾端显示数字】
                def add_line(fig, series, name, color, width=2, dash=None):
                    fig.add_trace(go.Scatter(
                        x=series.index, y=series, mode='lines', name=name,
                        line=dict(color=color, width=width, dash=dash)
                    ))
                    # 添加尾端具体的数字 Annotation
                    last_val = series.iloc[-1]
                    fig.add_annotation(
                        x=series.index[-1], y=last_val,
                        text=f"{last_val:.2f}",
                        showarrow=True, arrowhead=0, ax=30, ay=0,
                        font=dict(color=color, size=12, style="bold")
                    )

                add_line(fig, cum_port, "我的组合", "#00CC96", 3)
                add_line(fig, cum_spy, "S&P 500", "gray", 1, "dot")
                add_line(fig, cum_qqq, "Nasdaq 100", "#636EFA", 1, "dot")
                
                fig.update_layout(
                    hovermode="x unified", 
                    margin=dict(r=50), #以此留出空间给右侧数字
                    height=450,
                    yaxis_title="净值 (起点=1)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col_alloc:
                st.subheader("💰 资产分布")
                # 计算当前市值
                latest_prices = close_df.iloc[-1]
                # 估算相对市值（这里简化为权重*资金，不考虑再平衡的复杂历史）
                current_vals = {t: capital * w for t, w in weights.items()}
                
                df_alloc = pd.DataFrame(list(current_vals.items()), columns=['Ticker', 'Value'])
                df_alloc['Weight'] = df_alloc['Value'] / df_alloc['Value'].sum()
                
                fig_pie = px.pie(df_alloc, values='Value', names='Ticker', hole=0.4)
                fig_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.dataframe(df_alloc.style.format({'Value': "${:,.2f}", 'Weight': "{:.2%}"}), use_container_width=True)

            # --- [自动补全] 风险相关性分析 ---
            st.subheader("🔥 风险雷达：持仓相关性矩阵 (Correlation Matrix)")
            st.caption("颜色越红代表两个资产走势越同步。如果全部是深红色，说明你的分散化做得不够。")
            corr_matrix = returns[tickers].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"请检查输入格式。错误详情: {e}")

# ==========================================
# TAB 2: 量化扫描工厂 (复刻 v5.0 功能)
# ==========================================
with tab2:
    st.header("🧬 策略实验室")
    
    c1, c2 = st.columns([1, 3])
    
    with c1:
        st.info("从这里发掘下一个潜力股，添加到 Tab 1 的持仓中。")
        factor = st.selectbox("选择因子", ["🔥 动量 (涨幅)", "🛡️ 低波 (抗跌)", "💰 流动性 (热度)"])
        scan_pool_str = st.text_area("扫描池", "AAPL, MSFT, NVDA, TSLA, AMD, GOOG, META, AMZN, NFLX, COIN, MSTR, PLTR, ARM, SMH, SOXL", height=150)
        lookback = st.slider("回测天数", 30, 365, 90)
        top_k = st.slider("选出 Top N", 3, 10, 5)
        
    with c2:
        if st.button("开始扫描", key="scan_btn"):
            scan_tickers = [x.strip().upper() for x in scan_pool_str.split(',') if x.strip()]
            with st.spinner("正在计算因子..."):
                s_data = get_data(scan_tickers, period="2y")
                
            if s_data is not None:
                # 数据清洗
                cls = pd.DataFrame()
                vol = pd.DataFrame()
                for t in s_data.columns.levels[0]:
                    if 'Close' in s_data[t]: cls[t] = s_data[t]['Close']
                    if 'Volume' in s_data[t]: vol[t] = s_data[t]['Volume']
                
                cls = cls.ffill().dropna()
                
                # 切片
                start_idx = -1 * lookback
                if abs(start_idx) > len(cls): start_idx = 0
                sub_cls = cls.iloc[start_idx:]
                
                scores = {}
                for t in sub_cls.columns:
                    if t in ['SPY', 'QQQ']: continue
                    
                    if "动量" in factor:
                        scores[t] = (sub_cls[t].iloc[-1] - sub_cls[t].iloc[0]) / sub_cls[t].iloc[0]
                        asc = False
                        col_name = "区间涨幅"
                    elif "低波" in factor:
                        scores[t] = sub_cls[t].pct_change().std()
                        asc = True
                        col_name = "波动率"
                    elif "流动性" in factor:
                        scores[t] = (sub_cls[t] * vol[t].iloc[start_idx:]).mean()
                        asc = False
                        col_name = "日均成交额"
                
                # 排序
                res = pd.Series(scores).sort_values(ascending=asc).head(top_k)
                
                st.success(f"✅ 筛选完成！以下是表现最好的 {top_k} 只股票：")
                
                # 结果可视化
                r_c1, r_c2 = st.columns([1, 2])
                with r_c1:
                    df_res = pd.DataFrame({col_name: res.values}, index=res.index)
                    if "成交" not in col_name:
                        df_res[col_name] = df_res[col_name].apply(lambda x: f"{x:.2%}")
                    else:
                        df_res[col_name] = df_res[col_name].apply(lambda x: f"${x:,.0f}")
                    st.table(df_res)
                    
                with r_c2:
                    norm = sub_cls[res.index] / sub_cls[res.index].iloc[0]
                    fig_scan = go.Figure()
                    for t in res.index:
                        fig_scan.add_trace(go.Scatter(x=norm.index, y=norm[t], name=t))
                    fig_scan.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0), yaxis_title="归一化走势")
                    st.plotly_chart(fig_scan, use_container_width=True)
