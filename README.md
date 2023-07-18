# goorm_nlp_project_2

구름 2차 프로젝트(기계번역 QA)

해당 프로젝트는 팀 별 자체 제작 모델로 KLUE 데이터로 구성 된 평가 데이터에 대한 '편집거리'를 측정해 캐글 스코어를 기록한다.
평가데이터가 포함될 수 있는 KLUE 데이터가 학습데이터에 포함된 모델은 사용을 금한다.
수단과 방법을 가리지 않고 모델의 성능을 개선하는 것을 목표로 한다.

메인 모델 구성
1.KoBigbird
2.KoBERT
3.KoEelctra

성능이 가장 높은 KoBigbird 모델을 기반으로 KoBERT모델과 KoElectra모델 결과의 Hard Votinig Ensemble을 적용.
최종 스코어 - 1.65779 (Kaggle 1등)

해당 깃허브 코드는 본인이 KoBigbird모델의 학습에 사용한 코드까지만 기록.(그 외 모델은 다른 동료들이 학습)
Best-Fit code 폴더에 저장.
