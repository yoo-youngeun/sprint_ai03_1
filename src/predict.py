# -*- coding: utf-8 -*-
"""
YOLO 모델 예측 스크립트
- 지정한 이미지/폴더에 대해 예측 수행
- 결과 저장 및 간단 시각화
"""
import argparse
import os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="학습된 가중치 경로 or 모델명")
    parser.add_argument("--source", type=str, required=True, help="이미지 또는 폴더 경로")
    parser.add_argument("--project", type=str, default="./results/test", help="예측 결과 저장 경로")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 크기")
    parser.add_argument("--device", type=str, default="0", help="GPU/CPU 장치 지정")
    args = parser.parse_args()

    # 모델 로드
    model = YOLO(args.weights)

    # 예측 실행
    results = model.predict(
        source=args.source,   # 입력 이미지/폴더
        save=True,            # 결과 이미지 저장
        save_txt=True,        # 예측 라벨 txt 저장
        conf=args.conf,       # confidence threshold
        iou=args.iou,         # IOU threshold
        imgsz=args.imgsz,     # 입력 이미지 크기
        device=args.device,   # GPU/CPU
        project=args.project, # 결과 저장 경로
    )

    # 예측 결과 디렉터리 경로
    predict_dir = args.project
    labels_dir = os.path.join(predict_dir, 'labels')

    # 예측된 이미지 파일 목록
    images = [
        os.path.join(predict_dir, img)
        for img in os.listdir(predict_dir)
        if img.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    print(f"Found {len(images)} images.")

    # 상위 50개 이미지를 시각화
    for idx, img_path in enumerate(images[:50]):
        img_name = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, img_name.rsplit('.', 1)[0] + '.txt')

        # 라벨 개수 카운트
        label_count = 0
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                label_count = len(f.readlines())

        # 이미지 표시
        img = Image.open(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Index: {idx}   Label: {img_name}   라벨 수: {label_count}", fontsize=12)
        plt.show()

    return results


if __name__ == "__main__":
    main()
