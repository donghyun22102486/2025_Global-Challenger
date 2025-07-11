import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

def train_and_save_model(data_path='Faults.NNA'):
    """
    UCI Steel Plate Faults 데이터를 로드하여 XGBoost 모델을 학습하고,
    학습된 모델과 관련 정보를 파일로 저장합니다.
    """
    # 1. 데이터 로드 및 컬럼명 정의
    # 원본 데이터는 컬럼명이 없으므로 직접 정의해줍니다.
    feature_names = [
        'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
        'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
        'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
        'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index',
        'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index',
        'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index',
        'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas'
    ]
    target_names = [
        'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
        'Dirtiness', 'Bumps', 'Other_Faults'
    ]
    all_columns = feature_names + target_names
    
    try:
        df = pd.read_csv(data_path, header=None, names=all_columns, sep='\t')
        print("데이터 파일 로드 성공.")
    except FileNotFoundError:
        print(f"오류: '{data_path}' 파일을 찾을 수 없습니다. UCI 웹사이트에서 다운로드하여 이 스크립트와 같은 폴더에 위치시켜 주세요.")
        return

    # 2. 데이터 전처리
    # 입력(X)과 목표(y) 분리
    X = df[feature_names]
    y_one_hot = df[target_names]
    
    # One-hot 인코딩된 목표 변수를 단일 라벨로 변환 (0~6)
    # 예: [0,0,1,0,0,0,0] -> 2
    y_labels = np.argmax(y_one_hot.values, axis=1)

    # 라벨 인코더를 사용해 숫자 라벨을 실제 결함 이름으로 매핑 준비
    le = LabelEncoder()
    le.fit(target_names)
    
    # 3. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

    # 4. XGBoost 모델 학습
    print("XGBoost 모델 학습 시작...")
    model = xgb.XGBClassifier(
        objective='multi:softmax', # 다중 클래스 분류
        num_class=7,               # 클래스 개수
        use_label_encoder=False,   # DeprecationWarning 방지
        eval_metric='mlogloss'     # 평가 지표
    )
    model.fit(X_train, y_train)
    print("모델 학습 완료.")

    # 5. 모델 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"모델 정확도 (Accuracy): {accuracy:.4f}")

    # 6. 모델 및 라벨 인코더 저장 (가장 중요!)
    # 서버에서 예측값(숫자)을 다시 결함 이름(문자열)으로 변환하려면 라벨 인코더도 함께 저장해야 합니다.
    with open('fault_prediction_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'label_encoder': le, 'feature_names': feature_names}, f)
    print("학습된 모델과 관련 정보가 'fault_prediction_model.pkl' 파일로 저장되었습니다.")

if __name__ == '__main__':
    train_and_save_model()

