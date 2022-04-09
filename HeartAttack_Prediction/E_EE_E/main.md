<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# 대회 링크 : https://dacon.io/competitions/official/235848/overview/description



주어진 dataset을 통해 심장질환을 예측하는 dacon basic 대회입니다.



# Data description

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


# Import packages and train dataset



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from keras.models import Sequential
from keras.layers.core import Dense, Activation

warnings.filterwarnings(action='ignore')

heart_data = pd.read_csv('./data/train.csv')
heart_data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>53</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>197</td>
      <td>1</td>
      <td>0</td>
      <td>152</td>
      <td>0</td>
      <td>1.2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>52</td>
      <td>1</td>
      <td>3</td>
      <td>152</td>
      <td>298</td>
      <td>1</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>1.2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>54</td>
      <td>1</td>
      <td>1</td>
      <td>192</td>
      <td>283</td>
      <td>0</td>
      <td>0</td>
      <td>195</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>138</td>
      <td>236</td>
      <td>0</td>
      <td>0</td>
      <td>152</td>
      <td>1</td>
      <td>0.2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>35</td>
      <td>1</td>
      <td>1</td>
      <td>122</td>
      <td>192</td>
      <td>0</td>
      <td>1</td>
      <td>174</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


# Missing data check



```python
def check_missing_col(dataframe):
    counted_missing_col = 0
    for i, col in enumerate(dataframe.columns):
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            counted_missing_col += 1
            print(f'결측치가 있는 컬럼은: {col}입니다')
            print(f'총 {missing_values}개의 결측치가 존재합니다.')

    if counted_missing_col == 0:
        print('결측치가 존재하지 않습니다')

check_missing_col(heart_data)
```

<pre>
결측치가 존재하지 않습니다
</pre>
# Labeled encoding data features

- sex

    - female = 0, male = 1

- cp: 가슴 통증(chest pain) 종류 

  - 0: asymptomatic 무증상

  - 1: atypical angina 일반적이지 않은 협심증

  - 2: non-anginal pain 협심증이 아닌 통증

  - 3: typical angina 일반적인 협심증

- fbs: (fasting blood sugar) 공복 중 혈당

    - 120 mg/dl 이하일 시 = 0

    - 120 mg/dl 초과일 시 = 1

- restecg: (resting electrocardiographic) 휴식 중 심전도 결과 

  - 0: showing probable or definite left ventricular hypertrophy by Estes' criteria

  - 1: 정상

  - 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)

- exang: (exercise induced angina) 활동으로 인한  협심증 여부

    - 없음 = 0

    - 있음 = 1

- slope: (the slope of the peak exercise ST segment) 활동 ST 분절 피크의 기울기

  - 0: downsloping 하강

  - 1: flat 평탄

  - 2: upsloping 상승

- ca: number of major vessels colored by flouroscopy 형광 투시로 확인된 주요 혈관 수 (0~3 개) 

  - Null 값은 숫자 4로 인코딩됨

- thal: thalassemia 지중해빈혈 여부

  - 0: Null 

  - 1: normal 정상

  - 2: fixed defect 고정 결함

  - 3: reversable defect 가역 결함

- target: 심장 질환 진단 여부

  - 0: < 50% diameter narrowing

  - 1: > 50% diameter narrowing


# Process label encoding features to one-hot encoding

pd.get_dummies()를 통해 자동으로 one-hot encoding으로 변경해 줄 수 있다.



'ca' column이 가질 수 있는 값의 범위는 0~4 이지만 train dataset에서 ca=4인 data가 없으므로 직접 'ca_4' column을 생성해준다.



```python
#train_x = train data except 'id', 'target'
train_x = heart_data.drop(['id', 'target'], axis=1)
train_x = pd.get_dummies(train_x, columns= ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'])
train_x = pd.get_dummies(train_x, columns= ['ca'])
ca_4 = []
for i in range(train_x.shape[0]):
    ca_4.append(0)
train_x['ca_4'] = ca_4
print(train_x.shape)
print(train_x.head())

#train_y = train data only 'target'
train_y = heart_data[['target']]
print(train_y.shape)
print(train_y.head())
```

<pre>
(151, 30)
   age  trestbps  chol  thalach  oldpeak  sex_0  sex_1  cp_0  cp_1  cp_2  ...  \
0   53       130   197      152      1.2      0      1     0     0     1  ...   
1   52       152   298      178      1.2      0      1     0     0     0  ...   
2   54       192   283      195      0.0      0      1     0     1     0  ...   
3   45       138   236      152      0.2      1      0     1     0     0  ...   
4   35       122   192      174      0.0      0      1     0     1     0  ...   

   slope_2  thal_0  thal_1  thal_2  thal_3  ca_0  ca_1  ca_2  ca_3  ca_4  
0        0       0       0       1       0     1     0     0     0     0  
1        0       0       0       0       1     1     0     0     0     0  
2        1       0       0       0       1     0     1     0     0     0  
3        0       0       0       1       0     1     0     0     0     0  
4        1       0       0       1       0     1     0     0     0     0  

[5 rows x 30 columns]
(151, 1)
   target
0       1
1       1
2       0
3       1
4       1
</pre>
# Creat deep learning model

순차 layer는 3개의 deep learning 층으로 2개 모두 64개의 cell을 가지도록 임의로 지정, activation function은 relu로 하였습니다.



마지막 output layer는 activation function을 sigmoid로 하였습니다.



optimizer는 Nadam, loss는 binary_crossentropy로 하여 batch_size=64, 500회 반복학습시켰습니다.



```python
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(train_x, train_y, epochs=500, batch_size=64)
```

<pre>
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_19 (Dense)             (None, 64)                1984      
_________________________________________________________________
dense_20 (Dense)             (None, 64)                4160      
_________________________________________________________________
dense_21 (Dense)             (None, 64)                4160      
_________________________________________________________________
dense_22 (Dense)             (None, 1)                 65        
=================================================================
Total params: 10,369
Trainable params: 10,369
Non-trainable params: 0
_________________________________________________________________
Epoch 1/500
3/3 [==============================] - 1s 16ms/step - loss: 1.6036 - accuracy: 0.5116
Epoch 2/500
3/3 [==============================] - 0s 16ms/step - loss: 0.7361 - accuracy: 0.5725
Epoch 3/500
3/3 [==============================] - 0s 22ms/step - loss: 1.0292 - accuracy: 0.4968
...
Epoch 499/500
3/3 [==============================] - 0s 7ms/step - loss: 0.0447 - accuracy: 1.0000
Epoch 500/500
3/3 [==============================] - 0s 3ms/step - loss: 0.0569 - accuracy: 0.9908
</pre>
# Loss and accuracy against epoch plot 



```python
%matplotlib inline
import matplotlib.pyplot as plt
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 1.0])
acc_ax.set_ylim([0.0, 1.0])

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
```

<img src="data:image/png;base64,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"/>

# Model evaluation



```python
loss_and_metrics = model.evaluate(train_x, train_y, batch_size=32)
print('loss_and_metrics : ' + str(loss_and_metrics))
```

<pre>
5/5 [==============================] - 0s 5ms/step - loss: 0.0479 - accuracy: 0.9868
loss_and_metrics : [0.047855038195848465, 0.9867549538612366]
</pre>
# Prediction with test dataset 



```python
test_data = pd.read_csv('./data/test.csv')
result = pd.DataFrame()
result['id'] = test_data['id']
test_data = test_data.drop(['id'], axis=1)
test_data = pd.get_dummies(test_data, columns= ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
target = model.predict(np.array(test_data))
#0.5 보다 큰 추정값은 심장질환이 있을 것으로 예측 -> 1,  0.5보다 작은 값은 심장질환 없을 것으로 예측 -> 0
for i in range(target.shape[0]):
    if target[i] > 0.5:
        target[i] = 1
    else :
        target[i] = 0
result['target'] = target
print(result.head())
```

<pre>
   id  target
0   1     0.0
1   2     0.0
2   3     1.0
3   4     0.0
4   5     0.0
</pre>
# Make output file by prediction result



```python
result.to_csv('output.csv', index=False)
```
