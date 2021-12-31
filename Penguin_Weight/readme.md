# 대회링크 : https://dacon.io/competitions/official/235862/data

# Data description
## 1. train.csv : 학습 데이터
    - id : 샘플 아이디
    - Species : 펭귄의 종을 나타내는 문자열
    - Island : 샘플들이 수집된 Palmer Station 근처 섬 이름
    - Clutch Completion : 관찰된 펭귄 둥지의 알이 2개인 경우 Full Clutch이며 Yes로 표기
    - Culmen Length (mm) : 펭귄 옆모습 기준 부리의 가로 길이
    - Culmen Depth (mm) : 펭귄 옆모습 기준 부리의 세로 길이
    - Flipper Length (mm) : 펭귄의 팔(날개) 길이
    - Sex : 펭귄의 성별
    - Delta 15 N (o/oo) : 토양에 따라 변화하는 안정 동위원소 15N:14N의 비율
    - Delta 13 C (o/oo) : 먹이에 따라 변화하는 안정 동위원소 13C:12C의 비율
    - Body Mass (g): 펭귄의 몸무게를 나타내는 숫자 (g)


## 2. test.csv : 테스트 데이터
    - id : 샘플 아이디
    - Species : 펭귄의 종을 나타내는 문자열
    - Island : 샘플들이 수집된 Palmer Station 근처 섬 이름
    - Clutch Completion : 관찰된 펭귄 둥지의 알이 2개인 경우 Full Clutch이며 Yes로 표기
    - Culmen Length (mm) : 펭귄 옆모습 기준 부리의 가로 길이
    - Culmen Depth (mm) : 펭귄 옆모습 기준 부리의 세로 길이
    - Flipper Length (mm) : 펭귄의 팔(날개) 길이
    - Sex : 펭귄의 성별
    - Delta 15 N (o/oo) : 토양에 따라 변화하는 안정 동위원소 15N:14N의 비율
    - Delta 13 C (o/oo) : 먹이에 따라 변화하는 안정 동위원소 13C:12C의 비율


## 3. sample_submissoin.csv : 제출 양식
    - id : 샘플 아이디
    - Body Mass (g) : 펭귄의 몸무게를 나타내는 숫자 (g)

```python
# VB select 역할 = apply
data["ln"] = data["value"].apply(lambda x: math.log(x / data.iloc[-1]))
# 
# VB where 역할 = loc 으로 대체
# loc 인수는 boolean series 만 넣어주면 됨, 무명함수로 row 넣을 수 있음
#
# loc 문법 
 df.loc[행idx Series,열이름 Series]
# 행idx Series 의 자료형이 blean이면 True인 행만 뽑아줌, 
# 행idx Series 의 자료형이 정수형이면 해당 idx의 행만 뽑아줌
# 열이름이 자료형이 문자열이면, 해당 이름의 열만 뽑아줌 => Serise로 반환
# 열이름이 list이면 해당 목록의 열만 뽑아줌 => DataFrame으로 반환
```