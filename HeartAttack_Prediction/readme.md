# 대회 링크 : https://dacon.io/competitions/official/235848/overview/description

# data description
1. train.csv : 학습 데이터
- id: 데이터 고유 id
- age: 나이
- sex: 성별 (여자 = 0, 남자 = 1)
- cp: 가슴 통증(chest pain) 종류 
  - 0: asymptomatic 무증상
  - 1: atypical angina 일반적이지 않은 협심증
  - 2: non-anginal pain 협심증이 아닌 통증
  - 3: typical angina 일반적인 협심증
- trestbps: (resting blood pressure) 휴식 중 혈압(mmHg)
- chol: (serum cholestoral) 혈중 콜레스테롤 (mg/dl)
- fbs: (fasting blood sugar) 공복 중 혈당
    - 120 mg/dl 이하일 시 = 0
    - 120 mg/dl 초과일 시 = 1
- restecg: (resting electrocardiographic) 휴식 중 심전도 결과 
  - 0: showing probable or definite left ventricular hypertrophy by Estes' criteria
  - 1: 정상
  - 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
- thalach: (maximum heart rate achieved) 최대 심박수
- exang: (exercise induced angina) 활동으로 인한  협심증 여부
    - 없음 = 0
    - 있음 = 1
- oldpeak: (ST depression induced by exercise relative to rest) 휴식 대비 운동으로 인한 ST 하강
- slope: (the slope of the peak exercise ST segment) 활동 ST 분절 피크의 기울기
  - 0: downsloping 하강
  - 1: flat 평탄
  - 2: upsloping 상승
- ca: number of major vessels colored by flouroscopy 형광 투시로 확인된 주요 혈관 수 (0~3 개) 
  - Null 값은 숫자 4로 인코딩됨 
- thal: thalassemia 지중해빈혈 여부
  - 0 = Null 
  - 1 = normal 정상
  - 2 = fixed defect 고정 결함
  - 3 = reversable defect 가역 결함
- target: 심장 질환 진단 여부
  - 0: < 50% diameter narrowing
  - 1: > 50% diameter narrowing

2. test.csv : 테스트 데이터
1.의 데이터에서 target column 제외


3. sample_submissoin.csv : 제출 양식
- id: 데이터 고유 id
- target: 심장 질환 진단 여부

</br></br>
위 자료는 아래 데이터를 바탕으로 제작되었습니다.

https://archive.ics.uci.edu/ml/datasets/Heart+Disease

Creators:

1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.

2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.

3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.

4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.