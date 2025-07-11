# Projects 폴더

1. 모든 데이터 'datasets' 폴더에 보관
2. 학습 완료한 모든 모델 'models' 폴더에 보관
3. 코드 작성 시 모든 경로 해당 파일 기준으로 상대경로로 작성

```
경로 사용 예시

(python310) Global Challenger\Projects\donghyun\model1\backend> python model1.py

<model1.py>
MODEL_PATH = "../../../models/fault_prediction_model.pkl"


(python310) Global Challenger\Projects\junseop\backend> python train_model.py

<train_model.py>
data_path="../../datasets/Faults.NNA"
with open("../../models/fault_prediction_model.pkl", "wb")

```
