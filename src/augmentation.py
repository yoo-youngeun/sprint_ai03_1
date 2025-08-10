# -*- coding: utf-8 -*-
"""
augmentation.py

YOLO 학습용 데이터 증강 유틸리티.

핵심 포인트
- Albumentations 사용 시 반드시 bbox_params를 지정해야 bboxes가 정상 변환됨.
- 증강 결과 이미지 크기(aug_h, aug_w) 기준으로 YOLO 좌표를 재정규화.
- 원본 확장자를 유지하여 저장(.png/.jpg 혼재 지원).
- 소수/희소 클래스의 샘플 수를 target까지 끌어올리는 증강 루틴 제공.
- 과다 클래스는 균형을 맞추기 위해 완화 규칙(margin)으로 삭제.

주의
- bboxes 포맷: Albumentations엔 pascal_voc(xyxy, pixel), YOLO 저장 시 (cx,cy,w,h, normalized).
- 크롭 증강은 대상 객체가 화면 전체(1.0x1.0)를 차지하도록 단일 라벨을 부여.
"""

import os
import heapq
import random
import cv2
from collections import defaultdict, Counter

import albumentations as A

from .utils_yolo import yolo_to_xyxy, xyxy_to_yolo
from .utils_io import find_image_path, IMG_EXTS


def make_aug(fn):
    """
    단일 변환 `fn`을 Albumentations Compose로 감싸며
    bbox_params를 공통으로 강제 적용.

    Args:
        fn (A.BasicTransform): Albumentations 변환(예: A.HorizontalFlip(...))

    Returns:
        A.Compose: bbox_params가 설정된 변환 파이프라인
    """
    return A.Compose(
        [fn],
        bbox_params=A.BboxParams(
            format='pascal_voc',      # xyxy(pixel) 포맷
            label_fields=['category_ids'],
            min_visibility=0.001,     # 거의 0에 가까운 작은 박스만 허용 (완전 사라진 박스 제거)
            clip=True                 # 이미지 경계 밖으로 나간 박스는 잘라서 유지
        )
    )


# 🔧 기본 증강 파이프라인 모음
AUGMENTATIONS = [
    make_aug(A.HorizontalFlip(p=1.0)),
    make_aug(A.VerticalFlip(p=1.0)),
    make_aug(A.RandomBrightnessContrast(p=1.0)),
    make_aug(A.Rotate(limit=25, p=1.0, border_mode=0, value=0)),  # 회전 시 빈 영역은 0 채움
    make_aug(A.GaussianBlur(p=1.0)),
    make_aug(A.ColorJitter(p=1.0)),
    make_aug(A.RandomGamma(p=1.0)),
]


def copy_few_shot_images(
    yolo_img_dir,
    yolo_label_dir,
    cat_id_to_name,
    max_classes_threshold=75,
    top_n=10,
    padding_ratio=0
):
    """
    희소 클래스(few-shot)를 우선적으로 증강해 클래스 간 데이터 불균형을 완화.

    동작 개요:
      1) 현재 라벨(.txt)을 스캔해 클래스별 "이미지 수"를 계산
      2) 상위 빈도 클래스(top_n)는 증강 제외
      3) 목표치(max_classes_threshold)에 도달할 때까지:
         - 해당 클래스가 포함된 원본 이미지에서 대상 bbox만 크롭 증강
         - 또는 전체 이미지에 기하/포토메트릭 변환 적용 후 좌표 재계산

    Args:
        yolo_img_dir (str): 증강 대상/결과 이미지 폴더
        yolo_label_dir (str): YOLO 라벨(.txt) 폴더
        cat_id_to_name (dict): {class_id: class_name} (로그용)
        max_classes_threshold (int): 각 클래스별 목표 "이미지 수"
        top_n (int): 상위 빈도 클래스 제외 개수
        padding_ratio (float): (선택) 크롭 시 bbox 주변 패딩 비율 (현재는 미사용)

    Note:
        - 이미지 저장 시 원본 확장자를 그대로 사용합니다.
        - Albumentations에는 pascal_voc(xyxy) 박스를, YOLO 저장 시엔 정규화 좌표로 변환합니다.
    """
    os.makedirs(yolo_img_dir, exist_ok=True)
    os.makedirs(yolo_label_dir, exist_ok=True)

    # 클래스별 포함된 "이미지(베이스명) 집합" / 이미지별 포함 "클래스 집합"
    class_img_map = defaultdict(set)  # cls -> {base_names}
    img_class_map = defaultdict(set)  # base_name -> {cls_ids}

    # 1) 현재 라벨 스캔해 맵 생성
    for label_fname in os.listdir(yolo_label_dir):
        if not label_fname.endswith(".txt"):
            continue
        base_name = os.path.splitext(label_fname)[0]
        label_path = os.path.join(yolo_label_dir, label_fname)
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                class_img_map[cls_id].add(base_name)
                img_class_map[base_name].add(cls_id)

    class_counts = {cid: len(imgs) for cid, imgs in class_img_map.items()}
    top_n_classes = set(heapq.nlargest(top_n, class_counts, key=class_counts.get))

    # 2) 희소 클래스만 타깃으로 선정
    few_classes = {
        cid: list(imgs)
        for cid, imgs in class_img_map.items()
        if (len(imgs) <= max_classes_threshold and cid not in top_n_classes)
    }
    sorted_few_class_ids = sorted(few_classes.keys(), key=lambda x: class_counts.get(x, 0))

    # ---- 내부 함수: 단일 bbox 크롭 + 저장 ----
    def crop_and_save(image, bbox_xyxy, cls_id, base_name, count, out_ext):
        """
        단일 bbox를 크롭하여 새로운 학습 샘플을 생성.
        크롭 결과는 대상 객체가 화면을 100% 차지하므로 YOLO 라벨은 (0.5, 0.5, 1.0, 1.0)로 기록.

        Args:
            image (np.ndarray): 원본 이미지(BGR)
            bbox_xyxy (list[float]): [x1, y1, x2, y2] (pixel)
            cls_id (int): 클래스 ID
            base_name (str): 원본 베이스 파일명
            count (int): 증강 카운터 (파일명 suffix용)
            out_ext (str): 결과 저장 확장자 (원본과 동일)
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]

        # (선택) 패딩 적용 지점 — 현재는 padding_ratio 미사용
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return  # 잘못된 박스는 스킵

        cropped = image[y1:y2, x1:x2]
        ch, cw = cropped.shape[:2]
        if ch == 0 or cw == 0:
            return

        # YOLO: 대상이 화면 전체 → 단일 라벨 한 줄
        yolo_line = f"{cls_id} {0.5:.6f} {0.5:.6f} {1.0:.6f} {1.0:.6f}"

        new_base = f"{base_name}_aug{count}"
        cv2.imwrite(os.path.join(yolo_img_dir, new_base + out_ext), cropped)
        with open(os.path.join(yolo_label_dir, new_base + ".txt"), 'w', encoding='utf-8') as f:
            f.write(yolo_line + "\n")

        class_img_map[cls_id].add(new_base)
        img_class_map[new_base] = {cls_id}

        # 크롭된 이미지에 추가 변환 N회 적용
        for i in range(5):
            aug = random.choice(AUGMENTATIONS)
            try:
                # 크롭 이미지는 bbox가 화면 전체: 픽셀 기준 xyxy로 전달
                aug_img = aug(
                    image=cropped,
                    bboxes=[[0, 0, cw, ch]],
                    category_ids=[cls_id]
                )['image']
            except Exception:
                # bbox 처리 문제 발생 시, 동일 변환을 이미지에만 일단 적용 시도
                try:
                    aug_img = A.Compose([aug.transforms[0]])(image=cropped)['image']
                except Exception:
                    continue

            aug_base = f"{base_name}_aug{count}_crop{i}"
            cv2.imwrite(os.path.join(yolo_img_dir, aug_base + out_ext), aug_img)
            with open(os.path.join(yolo_label_dir, aug_base + ".txt"), 'w', encoding='utf-8') as f:
                f.write(yolo_line + "\n")

            class_img_map[cls_id].add(aug_base)
            img_class_map[aug_base] = {cls_id}

    # ---- 메인 루프 ----
    total_success_count = 0
    from itertools import cycle

    # 3) 클래스별로 목표 수에 도달할 때까지 반복 증강
    for cid in sorted_few_class_ids:
        img_list = few_classes[cid]
        print(f"\n📈 클래스 {cid} ({cat_id_to_name.get(cid, cid)}), 목표 증강 수: {max_classes_threshold - len(img_list)}")

        img_cycle = cycle(img_list)  # 이미지 순환자
        success_count = 0

        while len(class_img_map[cid]) < max_classes_threshold:
            base = next(img_cycle)
            if "_aug" in base:  # 이미 증강된 샘플은 스킵
                continue

            # 원본 이미지 경로 탐색(확장자 혼재 지원)
            image_path = find_image_path(yolo_img_dir, base, IMG_EXTS)
            label_path = os.path.join(yolo_label_dir, base + ".txt")
            if not (image_path and os.path.exists(label_path)):
                continue

            out_ext = os.path.splitext(image_path)[1].lower()  # 결과 저장 시 원본 확장자 유지
            image = cv2.imread(image_path)
            if image is None:
                continue
            h, w = image.shape[:2]

            # YOLO → pascal_voc(xyxy, pixel)
            bboxes_xyxy, class_ids = [], []
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, bw, bh = map(float, parts)
                    bboxes_xyxy.append(yolo_to_xyxy([x, y, bw, bh], w, h))
                    class_ids.append(int(cls))

            # (A) 해당 이미지에 목표치를 이미 달성한 클래스가 있다면 → 대상 cid만 크롭 증강
            if any(len(class_img_map[c]) >= max_classes_threshold for c in img_class_map[base]):
                for box, cls in zip(bboxes_xyxy, class_ids):
                    if cls == cid:
                        crop_and_save(image, box, cls, base, success_count, out_ext)
                        success_count += 1
                        total_success_count += 1
                        if len(class_img_map[cid]) >= max_classes_threshold:
                            break
            else:
                # (B) 전체 이미지에 변환 적용 → 좌표 재계산 후 YOLO로 저장
                aug = random.choice(AUGMENTATIONS)
                try:
                    aug_data = aug(image=image, bboxes=bboxes_xyxy, category_ids=class_ids)
                except Exception:
                    # 변환 실패(회전/박스포맷 등 이슈) 시 스킵
                    continue

                aug_img = aug_data['image']
                aug_bboxes_xyxy = aug_data['bboxes']     # xyxy(pixel)
                aug_class_ids = aug_data['category_ids']

                # ⚠️ 반드시 "증강 결과 이미지 크기" 기준으로 정규화해야 함
                aug_h, aug_w = aug_img.shape[:2]

                new_lines = []
                for (x1, y1, x2, y2), cls in zip(aug_bboxes_xyxy, aug_class_ids):
                    # 잘못된(면적 0 이하) 박스 필터링
                    if x2 <= x1 or y2 <= y1:
                        continue
                    # xyxy(pixel) → YOLO(cx,cy,w,h,normalized)
                    yolo_box = [max(0, min(1, v)) for v in xyxy_to_yolo([x1, y1, x2, y2], aug_w, aug_h)]
                    new_lines.append(f"{cls} {' '.join(f'{v:.6f}' for v in yolo_box)}")

                if not new_lines:
                    # 유효 박스가 하나도 없으면 샘플 생성 스킵
                    continue

                new_base = f"{base}_aug{success_count}"
                cv2.imwrite(os.path.join(yolo_img_dir, new_base + out_ext), aug_img)
                with open(os.path.join(yolo_label_dir, new_base + ".txt"), 'w', encoding='utf-8') as f:
                    f.write("\n".join(new_lines))

                for c in set(aug_class_ids):
                    class_img_map[c].add(new_base)
                img_class_map[new_base] = set(aug_class_ids)

                success_count += 1
                total_success_count += 1

    print(f"\n✅ 총 증강된 이미지 수: {total_success_count}개")


def fine_tune_delete_by_class_popularity_relaxed(
    yolo_img_dir,
    yolo_label_dir,
    target_count=75,
    margin=10
):
    """
    과다 클래스의 샘플을 '완화 규칙(margin)'으로 삭제하여 클래스 균형을 맞춤.

    규칙:
      - 어떤 이미지를 삭제하면 그 이미지에 포함된 모든 클래스의 카운트가 1씩 감소.
      - 삭제 후에도 모든 관련 클래스가 (target_count - margin) 이상이면 삭제 허용.
      - '_aug', '_copy' 등 증강/복제 샘플을 우선적으로 삭제.

    Args:
        yolo_img_dir (str): 이미지 폴더
        yolo_label_dir (str): 라벨(.txt) 폴더
        target_count (int): 각 클래스가 맞춰야 할 목표 이미지 수
        margin (int): 목표치 대비 허용 하한 (target_count - margin)

    Note:
        - 확장자 혼재를 고려해 실제 파일 삭제 전 `find_image_path`로 경로 확인.
    """
    image_to_classes = {}
    class_to_images = defaultdict(set)
    class_counts = Counter()

    # 이미지/라벨 스캔 → 각 이미지가 포함하는 클래스 집합 구축
    for fname in os.listdir(yolo_img_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in IMG_EXTS:
            continue

        classes = set()
        lbl_path = os.path.join(yolo_label_dir, base + ".txt")
        if os.path.exists(lbl_path):
            with open(lbl_path, encoding='utf-8') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        classes.add(parts[0])

        image_to_classes[base] = classes
        for c in classes:
            class_counts[c] += 1
            class_to_images[c].add(base)

    current_counts = class_counts.copy()
    to_delete = set()
    classes_sorted = sorted(class_counts.keys(), key=lambda c: class_counts[c], reverse=True)

    print(f"초기 클래스별 이미지 개수: {dict(class_counts)}")

    # 과다 클래스부터 조정
    for c in classes_sorted:
        if current_counts[c] <= target_count:
            continue

        imgs = list(class_to_images[c] - to_delete)
        # 포함된 클래스 수가 적은 이미지(= 영향이 작은)부터 삭제 검토
        imgs.sort(key=lambda img: len(image_to_classes[img]))

        for img in imgs:
            img_classes = image_to_classes[img]
            # 모든 관련 클래스가 (target_count - margin) 이상 유지된다면 삭제 가능
            if all(current_counts[ic] - 1 >= target_count - margin for ic in img_classes):
                to_delete.add(img)
                for ic in img_classes:
                    current_counts[ic] -= 1
                if current_counts[c] <= target_count:
                    break

    print(f"\n삭제 예정 이미지 수: {len(to_delete)}")
    print(f"삭제 후 예상 클래스별 이미지 개수: {dict(current_counts)}")

    # 실제 파일 삭제: 증강/복제 샘플 우선
    special = [b for b in to_delete if ('_aug' in b or '_copy' in b)]
    normal = [b for b in to_delete if b not in special]
    random.shuffle(special); random.shuffle(normal)

    for base in special + normal:
        img_path = find_image_path(yolo_img_dir, base, IMG_EXTS)  # 확장자 확인
        lbl_path = os.path.join(yolo_label_dir, base + ".txt")
        if img_path and os.path.exists(img_path):
            os.remove(img_path); print(f"삭제: {img_path}")
        if os.path.exists(lbl_path):
            os.remove(lbl_path); print(f"삭제: {lbl_path}")

    print("\n삭제 완료.")
