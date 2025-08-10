# -*- coding: utf-8 -*-
"""
YOLO 모델 학습 실행 스크립트
- data.yaml 기반으로 데이터셋 경로, 클래스 정보 불러오기
- ultralytics.YOLO로 학습 수행
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="data.yaml 경로")
    parser.add_argument("--weights", type=str, default="yolov8l.pt", help="사전 학습된 가중치 경로 또는 모델명")
    parser.add_argument("--epochs", type=int, default=10, help="학습 epoch 수")
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 크기")
    parser.add_argument("--project", type=str, default="./runs", help="결과 저장 폴더")
    parser.add_argument("--name", type=str, default="exp", help="실험 이름")
    parser.add_argument("--device", type=str, default="0", help="GPU 또는 CPU 선택 (예: '0' 또는 'cpu')")
    args = parser.parse_args()

    # YOLO 모델 로드
    model = YOLO(args.weights)

    # 학습 실행
    results = model.train(
        data=args.data,       # data.yaml
        epochs=args.epochs,   # 학습 횟수
        imgsz=args.imgsz,     # 입력 이미지 크기
        patience=20,          # 조기 종료 patience
        batch=16,             # 배치 크기
        device=args.device,   # 장치 선택
        lr0=0.01,              # 초기 learning rate
        lrf=0.01,              # 최종 learning rate
        mosaic=0.5,            # mosaic augmentation 비율
        mixup=0.2,             # mixup augmentation 비율
        close_mosaic=10,       # mosaic 종료 시점 (epoch)
        hsv_h=0.015,           # HSV 색상 변화
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.1,            # 상하 반전 확률
        fliplr=0.1,            # 좌우 반전 확률
        augment=True,          # 추가 증강 여부
        project=args.project,  # 결과 저장 경로
        name=args.name,        # 실험명
        exist_ok=True          # 기존 결과 덮어쓰기 허용
    )

    print(results)


if __name__ == "__main__":
    main()