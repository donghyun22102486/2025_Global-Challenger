import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 예시 광고 데이터 생성
data = {
    'age': [25, 34, 45, 23, 35, 52, 46, 51, 23, 40],
    'gender': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],  # 0: 남성, 1: 여성
    'ad_click': [1, 0, 0, 1, 1, 0, 0, 1, 1, 0],  # 1: 클릭, 0: 미클릭
    'time_spent': [5, 2, 1, 6, 7, 1, 2, 5, 6, 1],  # 광고에 머문 시간(초)
    'clicked': [1, 0, 0, 1, 1, 0, 0, 1, 1, 0]  # 타겟 변수
}
df = pd.DataFrame(data)

# 특성과 타겟 분리
X = df.drop('clicked', axis=1)
y = df['clicked']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

