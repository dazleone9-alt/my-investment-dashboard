import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 页面配置 ---
st.set_page_config(page_title="AlphaCopilot v7.0 实盘版", layout="wide", page_icon="💰")

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
        # group_by='ticker' 确保多级索引结构
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
st.title("💰 AlphaCopilot 实盘账本")

tab1, tab2 = st.tabs(["💼 我的实盘 (My Portfolio)", "🔍 机会扫描 (Scanner)"])

# ==========================================
# TAB 1: 实盘管理 (核心升级)
# ==========================================
with tab1:
    st.sidebar.header("💼 实盘录入")
    st.sidebar.info("格式：代码:股数:成本价\n(用逗号或换行分隔)")
    
    # 1. 新的输入格式
    default_pos = """NVDA:50:85.5
AAPL:100:180
MSFT:20:350
TSLA:30:210
COIN:40:150"""
    pos_input = st.sidebar.text_area("持仓明细", default_pos, height=150)
    
    # 解析逻辑
    portfolio_data = []
    tickers_query = []
    
    try:
        # 处理换行和逗号
        raw_items = pos_input.replace('\n', ',').split(',')
        for item in raw_items:
            item = item.strip()
            if not item: continue
            
            parts = item.split(':')
            if len(parts) == 3:
                t = parts[0].strip().upper()
                s = float(parts[1])
                c = float(parts[2])
                portfolio_data.append({'Ticker': t, 'Shares': s, 'Avg Cost': c})
                tickers_query.append(t)
            else:
                st.sidebar.error(f"格式错误忽略: {item}")
        
        if not portfolio_data:
            st.warning("请在左侧输入持仓信息，格式：代码:股数:成本")
            st.stop()

        # 获取数据
        with st.spinner("正在同步最新行情..."):
            raw_data = get_data(tickers_query)
            
        if raw_data is not None:
            # 数据清洗
            close_df = pd.DataFrame()
            for t in raw_data.columns.levels[0]:
                if 'Close' in raw_data[t]:
                    close_df[t] = raw_data[t]['Close']
            close_df = close_df.ffill().dropna()
            
            # 获取最新价格
            current_prices = close_df.iloc[-1]
            
            # --- 构建详细持仓表 ---
            df_port = pd.DataFrame(portfolio_data)
            
            # 匹配最新价格
            df_port['Current Price'] = df_port['Ticker'].apply(lambda x: current_prices.get(x, 0))
            
            # 计算核心数据
            df_port['Market Value'] = df_port['Shares'] * df_port['Current Price']
            df_port['Total Cost'] = df_port['Shares'] * df_port['Avg Cost']
            df_port['P&L ($)'] = df_port['Market Value'] - df_port['Total Cost']
            df_port['P&L (%)'] = (df_port['P&L ($)'] / df_port['Total Cost'])
            df_port['Allocation'] = df_port['Market Value'] / df_port['Market Value'].sum()
            
            # 汇总数据
            total_invested = df_port['Total Cost'].sum()
            total_value = df_port['Market Value'].sum()
            total_pl = total_value - total_invested
            total_pl_pct = total_pl / total_invested if total_invested != 0 else 0
            
            # --- 顶部大盘点 (Summary) ---
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总资产 (Total Equity)", f"${total_value:,.2f}")
            k2.metric("总投入 (Total Cost)", f"${total_invested:,.2f}")
            k3.metric("总盈亏 (Total P&L)", f"${total_pl:+,.2f}", f"{total_pl_pct:+.2%}")
            
            # 计算当日盈亏 (Day P&L) - 近似计算
            last_day_ret = close_df.pct_change().iloc[-1]
            day_pl = 0
            for _, row in df_port.iterrows():
                if row['Ticker'] in last_day_ret:
                    day_pl += row['Market Value'] * last_day_ret[row['Ticker']]
            k4.metric("今日预估波动", f"${day_pl:+,.2f}", f"{day_pl/total_value:+.2%}")
            
            st.divider()

            # --- 详细表格展示 ---
            st.subheader("📋 持仓详情")
            
            # 格式化表格用于展示
            display_df = df_port.copy()
            display_df = display_df.set_index('Ticker')
            
            # 样式优化
            st.dataframe(
                display_df[['Shares', 'Avg Cost', 'Current Price', 'Total Cost', 'Market Value', 'P&L ($)', 'P&L (%)', 'Allocation']].style
                .format({
                    'Shares': '{:,.2f}',
                    'Avg Cost': '${:,.2f}',
                    'Current Price': '${:,.2f}',
                    'Total Cost': '${:,.2f}',
                    'Market Value': '${:,.2f}',
                    'P&L ($)': '${:+,.2f}',
                    'P&L (%)': '{:+.2%}',
                    'Allocation': '{:.2%}'
                })
                .background_gradient(subset=['P&L (%)'], cmap='RdYlGn', vmin=-0.5, vmax=0.5),
                use_container_width=True
            )
            
            st.divider()
            
            # --- 图表分析区 ---
            c_chart, c_pie = st.columns([2, 1])
            
            with c_chart:
                st.subheader("📈 组合净值走势 (假设当前持仓一直持有)")
                
                # 计算历史每日净值 (Shares * Historical Price)
                # 这是一个"当前持仓回溯"，能告诉你如果两年前你就拿着这些票，现在是多少钱
                hist_value = pd.DataFrame()
                for _, row in df_port.iterrows():
                    t = row['Ticker']
                    if t in close_df.columns:
                        hist_value[t] = close_df[t] * row['Shares']
                
                total_hist_val = hist_value.sum(axis=1)
                # 归一化用于对比
                normalized_port = total_hist_val / total_hist_val.iloc[0]
                
                # 获取基准数据
                returns = close_df.pct_change().dropna()
                
                fig = go.Figure()
                
                # 画组合线
                fig.add_trace(go.Scatter(
                    x=total_hist_val.index, 
                    y=normalized_port, 
                    mode='lines', 
                    name='我的持仓',
                    line=dict(color='#00CC96', width=3)
                ))
                
                # 添加最新金额标签
                last_val = normalized_port.iloc[-1]
                fig.add_annotation(
                    x=total_hist_val.index[-1], y=last_val,
                    text=f"<b>{last_val:.2f}x</b>",
                    showarrow=True, arrowhead=0, ax=30, ay=0,
                    font=dict(color="#00CC96", size=12)
                )

                # 画基准线
                if 'SPY' in close_df.columns:
                    spy_cum = (1 + returns['SPY']).cumprod()
                    fig.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum, mode='lines', name='S&P 500', line=dict(color='gray', dash='dot')))
                
                if 'QQQ' in close_df.columns:
                    qqq_cum = (1 + returns['QQQ']).cumprod()
                    fig.add_trace(go.Scatter(x=qqq_cum.index, y=qqq_cum, mode='lines', name='Nasdaq 100', line=dict(color='#636EFA', dash='dot')))

                fig.update_layout(height=400, margin=dict(r=50), yaxis_title="净值增长 (1 = 起点)")
                st.plotly_chart(fig, use_container_width=True)
                
            with c_pie:
                st.subheader("💰 资产分布")
                fig_pie = px.pie(df_port, values='Market Value', names='Ticker', hole=0.4)
                fig_pie.update_traces(textinfo='label+percent')
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- 风险矩阵 ---
            st.subheader("🔥 风险相关性矩阵")
            # 只计算持仓股票，不含基准
            valid_tickers = [t for t in tickers_query if t in returns.columns]
            corr_matrix = returns[valid_tickers].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"处理数据时出错: {e}")

# ==========================================
# TAB 2: 量化扫描 (保持不变)
# ==========================================
with tab2:
    st.header("🧬 策略实验室")
    
    c1, c2 = st.columns([1, 3])
    
    with c1:
        st.info("从这里发掘下一个潜力股")
        factor = st.selectbox("选择因子", ["🔥 动量 (涨幅)", "🛡️ 低波 (抗跌)", "💰 流动性 (热度)"])
        scan_pool_str = st.text_area("扫描池", "AAPL, MSFT, NVDA, TSLA, AMD, GOOG, META, AMZN, NFLX, COIN, MSTR, PLTR, ARM, SMH, SOXL, KO, PEP, JNJ", height=150)
        lookback = st.slider("回测天数", 30, 365, 90)
        top_k = st.slider("选出 Top N", 3, 10, 5)
        
    with c2:
        if st.button("开始扫描", key="scan_btn"):
            scan_tickers = [x.strip().upper() for x in scan_pool_str.split(',') if x.strip()]
            with st.spinner("正在计算因子..."):
                s_data = get_data(scan_tickers, period="2y")
                
            if s_data is not None:
                cls = pd.DataFrame()
                vol = pd.DataFrame()
                for t in s_data.columns.levels[0]:
                    if 'Close' in s_data[t]: cls[t] = s_data[t]['Close']
                    if 'Volume' in s_data[t]: vol[t] = s_data[t]['Volume']
                
                cls = cls.ffill().dropna()
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
                
                res = pd.Series(scores).sort_values(ascending=asc).head(top_k)
                st.success(f"✅ 筛选完成！")
                
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
