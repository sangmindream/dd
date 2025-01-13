import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 페이지 설정
st.set_page_config(page_title="대학원 진학 성공률 분석", layout="wide")

# 제목
st.title("🎓 대학원 진학 성공률 분석")

# 샘플 데이터 생성 (실제 데이터로 대체 가능)
@st.cache_data
def load_data():
    data = {
        'GRE': np.random.randint(290, 340, 500),
        'TOEFL': np.random.randint(90, 120, 500),
        'GPA': np.random.uniform(3.0, 4.0, 500),
        '연구경험': np.random.randint(0, 3, 500),
        '합격여부': np.random.randint(0, 2, 500)
    }
    return pd.DataFrame(data)

df = load_data()

# 사이드바 - 사용자 입력
st.sidebar.header("📊 본인 정보 입력")
user_gre = st.sidebar.slider("GRE 점수", 290, 340, 315)
user_toefl = st.sidebar.slider("TOEFL 점수", 90, 120, 100)
user_gpa = st.sidebar.slider("GPA", 3.0, 4.0, 3.5)
user_research = st.sidebar.selectbox("연구 경험(년)", [0, 1, 2, 3])

# 메인 페이지 레이아웃
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 합격률 통계")
    
    # 전체 합격률
    total_acceptance = (df['합격여부'].mean() * 100).round(2)
    st.metric("전체 합격률", f"{total_acceptance}%")
    
    # GRE 점수별 합격률 차트
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='합격여부', y='GRE')
    st.pyplot(fig)

with col2:
    st.subheader("🎯 합격 가능성 예측")
    
    # 간단한 예측 모델 (로지스틱 회귀)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    X = df[['GRE', 'TOEFL', 'GPA', '연구경험']]
    y = df['합격여부']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    # 사용자 입력 데이터로 예측
    user_data = np.array([[user_gre, user_toefl, user_gpa, user_research]])
    user_data_scaled = scaler.transform(user_data)
    prediction_prob = model.predict_proba(user_data_scaled)[0][1]
    
    st.metric("예상 합격 확률", f"{(prediction_prob * 100).round(2)}%")
    
    # 각 요소별 중요도
    importance_df = pd.DataFrame({
        '요소': ['GRE', 'TOEFL', 'GPA', '연구경험'],
        '중요도': abs(model.coef_[0])
    })
    
    fig2, ax2 = plt.subplots()
    sns.barplot(data=importance_df, x='요소', y='중요도')
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# 데이터 분포 확인
st.subheader("📊 전체 데이터 분포")
st.dataframe(df.describe())
