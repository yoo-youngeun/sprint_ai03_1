# 🏥 Sprint AI Project 1
이 프로젝트는 경구 복용 의약품(정제, 캡슐 등)의 이미지를 기반으로, 개별 약제를 자동으로 식별하고 위치를 탐지하는 객체 검출(Object Detection) 시스템을 구축하는 것을 목표로 합니다.  

약제의 외형, 색상, 각인, 형태 등의 시각적 특징을 활용하여 이미지 내 모든 약제 객체를 정확히 감지하고, YOLO 기반 모델을 통해 Bounding Box 좌표와 클래스(약제명)를 예측합니다.

## 📂 프로젝트 구조
```
sprint_ai03_1/
├── src/
│ ├── data_preprocessing.py
│ ├── augmentation.py
│ ├── utils_io.py
│ ├── utils_yolo.py
│ ├── utils_viz.py
│ ├── train.py
│ └── predict.py
├── notebooks/
│ └── data_preprocessing_yye.ipynb
├── requirements.txt
└── data.yaml
```

## 📁 파일 및 디렉터리 설명
`src/`  
YOLO 학습을 위한 데이터 전처리, 증강, 시각화, 유틸리티 코드가 들어있는 모듈 디렉터리
- data_preprocessing.py
    - COCO JSON 병합, YOLO 라벨 변환, 불완전 라벨 필터링, 학습/검증 데이터 분할 등의 함수 포함
    - 데이터셋 준비 전체 파이프라인의 핵심 처리 담당
- augmentation.py
    - Albumentations 기반 YOLO 데이터 증강 함수 모음
    - 희소 클래스(few-shot class) 증강, 잘라내기(crop) 증강, 클래스별 증강 횟수 제한 기능 포함
- utils_io.py
    - 경로 생성, 이미지/라벨 파일 검사, 클래스별 통계 계산, 파일 입출력 관련 유틸 함수
- utils_yolo.py
    - YOLO ↔ COCO 어노테이션 변환 함수, bbox 포맷 변환(xyxy ↔ xywh ↔ YOLO) 유틸리티
- utils_viz.py
    - 바운딩박스 오류 시각화, 데이터셋 분포 그래프, 증강 결과 이미지 확인 등의 시각화 함수
- train.py
    - YOLO 모델 학습 실행 스크립트
    - data.yaml을 읽어 학습 파라미터 설정 후 ultralytics.YOLO로 학습 수행
- predict.py
    - 학습된 YOLO 모델을 이용한 추론(예측) 스크립트
    - 추론 결과 저장 및 시각화

`notebooks/`
- data_preprocessing_yye.ipynb
    - 개인 실험/개발용 Jupyter Notebook
    - src/ 모듈을 불러와서 데이터 전처리와 증강 실행, 시각화 테스트 등 수행
`requirements.txt`
    - 프로젝트 실행에 필요한 Python 라이브러리 목록
    - `$ pip install -r requirements.txt` 설치 가능
