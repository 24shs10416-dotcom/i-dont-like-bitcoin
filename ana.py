import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# 페이지 설정 (레이아웃을 넓게 설정)
st.set_page_config(page_title="Bitcoin Analysis Dashboard", layout="wide")

@st.cache_data
def load_data():
    """
    데이터 로드 및 전처리 함수
    같은 폴더의 coin.csv 파일을 우선적으로 읽어옵니다.
    """
    # 분석할 파일명 설정
    file_path = 'coin.csv'
    
    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        # 만약 확장자가 중복된 coin.csv.csv로 되어 있을 경우를 대비
        if os.path.exists('coin.csv.csv'):
            file_path = 'coin.csv.csv'
        else:
            raise FileNotFoundError(f"'{file_path}' 파일을 찾을 수 없습니다. 파이썬 파일과 같은 폴더에 저장되어 있는지 확인해주세요.")

    # CSV 데이터 로드 (구분자 ';')
    df = pd.read_csv(file_path, sep=';')
    
    # 날짜 데이터 전처리: 따옴표 제거 및 datetime 객체 변환
    date_cols = ['timeOpen', 'timeClose', 'timestamp']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str).str.replace('"', ''))
    
    # 분석을 위해 시간 순서대로 정렬
    df = df.sort_values('timeOpen')
    
    return df

def predict_tomorrow(df):
    """
    선형 회귀 모델을 사용하여 내일의 가격을 예측합니다.
    """
    # 학습을 위한 데이터 준비 (최근 60일 데이터 사용)
    # 날짜를 숫자로 변환하여 피처로 사용
    df_pred = df.copy().tail(60)
    df_pred['days_index'] = np.arange(len(df_pred))
    
    X = df_pred[['days_index']]
    y = df_pred['close']
    
    # 모델 생성 및 학습
    model = LinearRegression()
    model.fit(X, y)
    
    # 내일 날짜 인덱스로 예측
    tomorrow_index = np.array([[len(df_pred)]])
    predicted_price = model.predict(tomorrow_index)[0]
    
    # 결정 계수(R-squared)로 모델 신뢰도 측정
    r_squared = model.score(X, y)
    
    return predicted_price, r_squared

def main():
    st.title("📊 비트코인(BTC) 가격 분석 및 내일 예측")
    st.markdown("`coin.csv` 파일 데이터를 기반으로 한 AI 예측 대시보드입니다.")

    try:
        df = load_data()
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        st.info("💡 해결 방법: 분석하려는 CSV 파일 이름을 'coin.csv'로 변경하여 이 프로그램(.py)과 같은 폴더에 넣어주세요.")
        return

    # 사이드바: 필터링 옵션
    st.sidebar.header("🔍 조회 설정")
    
    # 기간 선택 필터
    min_date = df['timeOpen'].min().date()
    max_date = df['timeOpen'].max().date()
    
    date_range = st.sidebar.date_input(
        "날짜 범위 선택",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # 선택된 기간으로 데이터 필터링
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['timeOpen'].dt.date >= start_date) & (df['timeOpen'].dt.date <= end_date)
        filtered_df = df.loc[mask].copy()
    else:
        filtered_df = df.copy()

    if filtered_df.empty:
        st.warning("선택한 날짜 범위에 해당하는 데이터가 없습니다.")
        return

    # 1. 지표 섹션 (KPI)
    st.subheader("📌 주요 요약 지표")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    latest = filtered_df.iloc[-1]
    prev_close = filtered_df['close'].iloc[-2] if len(filtered_df) > 1 else latest['close']
    delta = ((latest['close'] - prev_close) / prev_close) * 100

    kpi1.metric("현재 종가", f"{latest['close']:,.0f} KRW", f"{delta:.2f}%")
    kpi2.metric("기간 최고가", f"{filtered_df['high'].max():,.0f} KRW")
    kpi3.metric("기간 최저가", f"{filtered_df['low'].min():,.0f} KRW")
    kpi4.metric("평균 거래량", f"{filtered_df['volume'].mean():,.0f}")

    # 2. 인공지능 가격 예측 섹션
    st.divider()
    st.subheader("🤖 AI 기반 내일 가격 예측")
    
    # 예측 실행
    predicted_price, confidence = predict_tomorrow(df)
    current_price = df['close'].iloc[-1]
    diff = predicted_price - current_price
    direction = "상승 📈" if diff > 0 else "하락 📉"
    
    col_pred1, col_pred2 = st.columns([1, 2])
    
    with col_pred1:
        st.write(f"**예측 결과:** 내일은 오늘보다 **{direction}**할 것으로 예상됩니다.")
        st.metric("내일 예상가", f"{predicted_price:,.0f} KRW", f"{diff:,.0f} KRW ({ (diff/current_price)*100:.2f}%)")
        st.info(f"모델 신뢰도 (R²): {confidence:.2f} (1.0에 가까울수록 과거 추세와 일치)")

    with col_pred2:
        # 예측 근거 시각화
        recent_df = df.tail(30).copy()
        pred_date = recent_df['timeOpen'].iloc[-1] + timedelta(days=1)
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=recent_df['timeOpen'], y=recent_df['close'], name='과거 가격'))
        fig_pred.add_trace(go.Scatter(x=[recent_df['timeOpen'].iloc[-1], pred_date], 
                                     y=[current_price, predicted_price], 
                                     name='예측 지점', line=dict(dash='dash', color='red')))
        
        fig_pred.update_layout(title="최근 30일 추세 및 내일 예측 점선", template="plotly_dark", height=300)
        st.plotly_chart(fig_pred, use_container_width=True)

    # 3. 가격 차트 섹션
    st.divider()
    st.subheader("📈 가격 변동 추이 (Candlestick)")
    
    use_ma = st.toggle("20일 이동평균선(MA20) 보기", value=True)
    
    fig = go.Figure(data=[go.Candlestick(
        x=filtered_df['timeOpen'],
        open=filtered_df['open'],
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'],
        name='Bitcoin'
    )])

    if use_ma and len(filtered_df) >= 20:
        ma20 = filtered_df['close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=filtered_df['timeOpen'], y=ma20, name='MA 20', line=dict(color='orange', width=1.5)))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. 보조 차트 섹션
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 일별 거래량")
        vol_fig = px.bar(filtered_df, x='timeOpen', y='volume', color='volume', color_continuous_scale='Viridis')
        vol_fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(vol_fig, use_container_width=True)

    with col_b:
        st.subheader("📉 수익률 분포")
        filtered_df['daily_return'] = filtered_df['close'].pct_change() * 100
        ret_fig = px.histogram(filtered_df, x='daily_return', nbins=30, marginal="box", title="Daily Return %")
        ret_fig.update_layout(template="plotly_dark")
        st.plotly_chart(ret_fig, use_container_width=True)

    # 5. 데이터 테이블
    with st.expander("📂 전체 데이터 보기"):
        st.dataframe(filtered_df.sort_values('timeOpen', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
