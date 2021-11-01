# 김봉준, 박원빈 Kaggle 도전기

## 종목 : [Lux AI](https://www.kaggle.com/c/lux-ai-2021/overview)

#### 평가방식
- 팀별로 bot은 하루에 최대 5개까지 submit할 수 있다. 
- 등록한 bot은 rating이 비슷한 다른 bot과 게임을 하게 된다.
- 등록된 bot은 competition 끝날 때까지 새로 등록된 봇과 싸우게 된다.
- leaderboard는 해당 팀이 등록한 bot 중 rating이 가장 높은 봇 1개만 표시가 된다. 
- subission 페이지에 가면 등록한 모든 봇의 점수를 확인할 수 있다. 
- bot을 업로드하면 정상작동 여부 검증하고 pool에 추가되게 된다.
- 초기 점수는 600 점 (이기면 득점, 지면 실점, 비기면 평균에 가까워짐)
- deadline(2021.12.06)이 지나면 추가적인 제출이 불가하다.
- 제출한 봇 상위 2개가 평가에 반영될 수 있다.
- 최종 순위는 2021.12.21일 기준 leaderboard 순위다.

#### 참고자료
참고자료의 3,4는 대회에서 제공한 Template이다.  
Template 그대로 카피해서 조금씩 수정하여 제출해도 된다.
1. [게임 세부내용](https://www.kaggle.com/c/lux-ai-2021/overview/lux-ai-specifications)
2. [GitHub-게임다운](https://github.com/Lux-AI-Challenge/Lux-Design-2021)
3. [Kaggle-NoteBook-Tutorial](https://www.kaggle.com/stonet2000/lux-ai-season-1-jupyter-notebook-tutorial)
4. [Kaggle-NoteBook-QuickStart](https://www.kaggle.com/stonet2000/lux-ai-season-1-jupyter-notebook-quickstart)
5. [Lux - API Kit Python 문서](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits/python)
6. [Lux - API Kit Functions](https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/kits/README.md)
7. [Lux - YouTube](https://www.youtube.com/channel/UCK4aJwBPG6nME0yLNUi3qQQ/videos)

#### 환경설정
1. [NodeJS v12 다운로드](https://nodejs.org/en/download/)
2. [Docker 다운로드](https://docs.docker.com/get-docker/)
3. [Docker file](https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/Dockerfile)
4. [ReplayViewer-Web-Upload](https://2021vis.lux-ai.org/)
5. [ReplayViewer-Git-Download](https://github.com/Lux-AI-Challenge/Lux-Viewer-2021)
 
### ToDo List
- [x] 개발환경 설치 : VScode, Python, NodeJS
- [x] 게임 규칙 및 득점 기준 확인
- [x] LUX 실행방법 확인 
- [ ] LUX API 사용법 확인
- [ ] LUX 직접 실행해서 데이터 확인하기
- [ ] LUX 전략 및 프로세스 설계
- [ ] LUX 설계 내용 구현 시작
- [ ] LUX 직접 실행해서 업로드
- [ ] LUX leaderBoard 점수확인
- [ ] LUX 결과 피드백 및 개선사항 정의
- [ ] LUX 결과 개선 결과 확인 및 피드백 반복
