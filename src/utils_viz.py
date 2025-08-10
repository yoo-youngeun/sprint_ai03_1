# -*- coding: utf-8 -*-
"""
utils_viz.py
- 데이터셋 시각화, 바운딩 박스 오류 검출, 클래스 분포 시각화 등
"""
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from PIL import Image
import numpy as np

from .utils_io import IMG_EXTS  # 프로젝트 전역에서 통일된 확장자 사용


# ----------------------------
#  BBox 좌표/IoU 계산 관련 함수
# ----------------------------
def compute_iou(box1, box2):
    """
    두 박스(x1,y1,x2,y2)의 IoU(Intersection over Union) 계산
    """
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area
    return 0.0 if union_area == 0 else inter_area / union_area


def xywh_to_xyxy(bbox):
    """
    COCO 형식 [x,y,w,h] → 좌상단-우하단 좌표 [x1,y1,x2,y2]
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


# ----------------------------
#  JSON 병합/맵 생성
# ----------------------------
def merge_all_jsons_recursive(json_folder):
    """
    json_folder 내 모든 JSON 파일을 재귀적으로 읽어,
    이미지/어노테이션/카테고리/차트 매핑 딕셔너리 생성
    
    Returns:
        images_map: {image_id: image_info}
        annotations_map: {image_id: [annotations]}
        categories_map: {category_id: category_info}
        chart_map: {image_id: [chart_info]}
    """
    images_map = defaultdict(lambda: None)
    annotations_map = defaultdict(list)
    categories_map = dict()
    chart_map = defaultdict(list)

    for root, _, files in os.walk(json_folder):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                img_info = data['images'][0]
                img_id = img_info['id']
                images_map[img_id] = img_info

                for ann in data['annotations']:
                    annotations_map[img_id].append(ann)

                for cat in data['categories']:
                    categories_map[cat['id']] = cat

                chart = img_info.get("chart", "")
                chart_map[img_id].append(chart)

    return images_map, annotations_map, categories_map, chart_map


# ----------------------------
#  IoU 기반 BBox 시각화
# ----------------------------
def visualize_overlapping_bboxes_with_all_labels(
    images_map, annotations_map, categories_map, chart_map, source_img_dir, iou_threshold=0.1
):
    """
    IoU 기준으로 겹치는 바운딩박스 시각화
    - IoU ≥ iou_threshold인 박스 쌍을 색상으로 표시
    """
    base_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown']

    for img_id in images_map:
        anns = annotations_map[img_id]
        if len(anns) <= 1:
            continue

        boxes = [xywh_to_xyxy(ann['bbox']) for ann in anns]
        overlapping_pairs, overlapping_indices = [], set()

        # IoU 계산
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if compute_iou(boxes[i], boxes[j]) >= iou_threshold:
                    overlapping_pairs.append((i, j))
        if not overlapping_pairs:
            continue
        for i, j in overlapping_pairs:
            overlapping_indices.update([i, j])

        img_file = images_map[img_id]['file_name']
        img_path = os.path.join(source_img_dir, img_file)

        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"⚠ 이미지 파일 없음: {img_file}, 건너뜀")
            continue

        print(f"Image: {img_file} - Overlapping bbox pairs (IoU≥{iou_threshold}): {overlapping_pairs}")

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        ax = plt.gca()

        for i, ann in enumerate(anns):
            x, y, w, h = ann['bbox']
            color = base_colors[i % len(base_colors)] if i in overlapping_indices else 'black'
            ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none'))

            chart_label = chart_map[img_id][i] if i < len(chart_map[img_id]) else "N/A"
            category_id = ann.get("category_id")
            category_name = categories_map.get(category_id, {}).get("name", "unknown")
            ax.text(x, max(y - 10, 5), f"[{i}] {category_name} - {chart_label}",
                    color=color, fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        plt.axis('off')
        plt.title(f"{img_file} - Overlapping BBoxes (IoU≥{iou_threshold})")
        plt.show()


# ----------------------------
#  YOLO 라벨 로드 및 변환
# ----------------------------
def load_yolo_labels(txt_path, img_w, img_h):
    """
    YOLO 라벨(.txt)을 읽어 픽셀 좌표(x1,y1,x2,y2)로 변환
    """
    boxes = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, w, h = map(float, parts)
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((int(cls_id), x1, y1, x2, y2))
    return boxes


def is_out_of_bounds(box, img_w, img_h):
    """
    박스가 이미지 범위를 벗어나는지 확인
    """
    _, x1, y1, x2, y2 = box
    return x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h


def compute_iou_yolo(box1, box2):
    """
    YOLO 변환 박스 좌표 간 IoU 계산
    """
    _, x1, y1, x2, y2 = box1
    _, x3, y3, x4, y4 = box2
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0


# ----------------------------
#  오류 이미지 시각화
# ----------------------------
def show_image_with_boxes(img, boxes, out_classes, overlap_classes, fname):
    """
    이미지와 바운딩박스를 표시, 오류 클래스 색상 강조
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    for cls_id, x1, y1, x2, y2 in boxes:
        color = 'blue'
        if cls_id in out_classes:
            color = 'red'       # 이미지 밖
        elif cls_id in overlap_classes:
            color = 'orange'    # 겹침
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=2, edgecolor=color, facecolor='none'))
        ax.text(x1, y1 - 4, f'{cls_id}', color=color, fontsize=10)
    ax.set_title(f"{fname}" + ("  [⚠️ 확인 필요]" if (out_classes or overlap_classes) else ""))
    ax.axis('off')
    plt.show()


def process_image(image_path, label_path, fname):
    """
    단일 이미지의 라벨 박스 오류(범위 밖/겹침) 검사 및 시각화
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] 이미지 로드 실패: {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    boxes = load_yolo_labels(label_path, w, h)
    if not boxes:
        return None

    out_classes, overlap_classes = set(), set()
    for i, b in enumerate(boxes):
        if is_out_of_bounds(b, w, h):
            out_classes.add(b[0])
        for j in range(i + 1, len(boxes)):
            if compute_iou_yolo(b, boxes[j]) > 0:
                overlap_classes.update([b[0], boxes[j][0]])

    if out_classes or overlap_classes:
        print(f"📂 {fname}")
        if out_classes:
            print(f"  🔴 이미지 밖 클래스: {sorted(out_classes)}")
        if overlap_classes:
            print(f"  🟠 겹치는 클래스: {sorted(overlap_classes)}\n")
        show_image_with_boxes(img, boxes, out_classes, overlap_classes, fname)

    return len(out_classes) > 0, len(overlap_classes) > 0


def process_folder(images_dir, labels_dir):
    """
    폴더 내 모든 이미지에 대해 오류 검사 수행
    """
    total = out_count = overlap_count = 0
    print("🔍 오류 있는 파일 목록:\n")
    for fname in sorted(os.listdir(images_dir)):
        if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
            continue
        base = os.path.splitext(fname)[0]
        img_p = os.path.join(images_dir, fname)
        lbl_p = os.path.join(labels_dir, base + '.txt')
        if not os.path.exists(lbl_p):
            continue
        r = process_image(img_p, lbl_p, fname)
        if r:
            total += 1
            o, ov = r
            if o: out_count += 1
            if ov: overlap_count += 1
    print("📊 통계 요약:")
    print(f"  전체 오류 이미지 수: {total}")
    print(f"  🔴 이미지 밖 박스 포함 수: {out_count}")
    print(f"  🟠 겹치는 박스 포함 수: {overlap_count}")


# ----------------------------
#  클래스 분포 변화 시각화
# ----------------------------
def plot_class_distribution(before_counts, after_counts):
    """
    증강 전후 클래스별 이미지 수 분포 비교 바 차트
    """
    sorted_classes = sorted(before_counts.keys(), key=lambda x: int(x))
    before_vals = [before_counts.get(cls, 0) for cls in sorted_classes]
    after_vals = [after_counts.get(cls, 0) for cls in sorted_classes]

    bar_width = 0.35
    index = np.arange(len(sorted_classes))

    plt.figure(figsize=(17, 6))
    plt.bar(index, before_vals, bar_width, label='Before Augmentation')
    plt.bar(index, after_vals, bar_width, label='After Augmentation', alpha=0.5)

    max_v = max(max(before_vals or [0]), max(after_vals or [0]))
    plt.ylim(0, max_v if max_v > 0 else 1)
    if max_v > 0:
        step = max(1, max_v // 10)
        for y in range(0, max_v + step, step):
            plt.axhline(y=y, linestyle='--', alpha=0.3)

    plt.ylabel('이미지수')
    plt.title('Class Distribution Before and After Augmentation')
    plt.xticks(index, sorted_classes, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()
