# -*- coding: utf-8 -*-
"""
data_preprocessing.py

COCO → YOLO 데이터 준비 파이프라인의 핵심 유틸리티 모듈.

주요 기능:
- 여러 COCO JSON(폴더 재귀 탐색)을 하나로 병합
- COCO → YOLO 라벨 변환(txt)
- 파일명 규칙(약물 개수)과 실제 어노테이션 개수를 비교해 불완전 데이터 삭제
- 이미지 복사 및 학습/검증 데이터 분할
- YOLO 학습용 data.yaml 자동 생성

주의:
- 병합 시 annotation id는 재부여되며, image/category는 id 기준으로 중복 제거합니다.
- convert_coco_to_yolo는 COCO bbox(x,y,w,h, pixel)를 YOLO 형식(cx,cy,w,h, normalized)로 변환합니다.
"""

import os
import json
import glob
import shutil
import random
from collections import defaultdict
from PIL import Image


def merge_coco_jsons(root_json_dir, merged_json_path):
    """
    주어진 디렉터리(하위 폴더 포함) 내의 모든 COCO JSON을 하나로 병합합니다.

    Args:
        root_json_dir (str): COCO JSON들이 들어있는 최상위 폴더 경로.
        merged_json_path (str): 병합 결과를 저장할 경로(파일명 포함).

    동작:
        - 모든 json을 읽어 images/annotations/categories를 모읍니다.
        - image.id / category.id 기준으로 중복을 제거합니다.
        - annotation.id는 1부터 재부여합니다.
    """
    os.makedirs(os.path.dirname(merged_json_path), exist_ok=True)

    # 재귀적으로 모든 .json 찾기
    json_files = glob.glob(os.path.join(root_json_dir, '**', '*.json'), recursive=True)
    print(f"[DEBUG] 찾은 JSON 파일 개수: {len(json_files)}")
    print(f"[DEBUG] 예시 JSON 파일들: {json_files[:5]}")

    merged = {"images": [], "annotations": [], "categories": []}
    annotation_id = 1
    image_id_set, category_set = set(), set()

    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 이미지: id 기준으로 중복 방지
        for img in data.get("images", []):
            if img["id"] not in image_id_set:
                merged["images"].append(img)
                image_id_set.add(img["id"])

        # 어노테이션: 새 id로 재부여
        for ann in data.get("annotations", []):
            ann["id"] = annotation_id
            annotation_id += 1
            merged["annotations"].append(ann)

        # 카테고리: id 기준으로 중복 방지
        for cat in data.get("categories", []):
            if cat["id"] not in category_set:
                merged["categories"].append(cat)
                category_set.add(cat["id"])

    # 병합 저장
    with open(merged_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"[DEBUG] 병합된 JSON 저장 완료: {merged_json_path}")


def get_master_categories(json_paths):
    """
    하나 이상의 COCO JSON에서 category id→name 매핑과,
    정렬된 id 순서대로 0..N-1 인덱스를 부여한 매핑을 만듭니다.

    Args:
        json_paths (list[str]): COCO JSON 파일 경로 리스트.

    Returns:
        tuple(dict, dict):
            - cat_id_to_index: {category_id: 0..N-1}
            - cat_id_to_name: {category_id: "name"}
    """
    cat_id_to_name = {}
    for p in json_paths:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for cat in data['categories']:
            cat_id_to_name[cat['id']] = cat['name']

    # category id 정렬 후 0..N-1 인덱스 부여
    sorted_cat_ids = sorted(cat_id_to_name.keys())
    cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
    return cat_id_to_index, cat_id_to_name


def convert_coco_to_yolo(annotation_file, source_img_dir, target_yolo_dir, master_cat_id_to_idx):
    """
    COCO 형식의 어노테이션(JSON)을 YOLO 형식의 라벨(txt)로 변환합니다.

    Args:
        annotation_file (str): COCO JSON 경로.
        source_img_dir (str): 이미지들이 위치한 폴더(파일명은 JSON의 images.file_name와 일치해야 함).
        target_yolo_dir (str): YOLO 라벨(.txt) 출력 폴더.
        master_cat_id_to_idx (dict): category_id → YOLO class index 매핑.

    동작:
        - 각 image_id에 해당하는 annotations를 모아 txt 1개를 생성합니다.
        - bbox(x,y,w,h, pixel) → (cx,cy,w,h, normalized)로 변환하여 기록합니다.
    """
    os.makedirs(target_yolo_dir, exist_ok=True)

    # COCO JSON 로드
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # image_id ↔ file_name 매핑
    img_id_to_file = {img['id']: img['file_name'] for img in data['images']}

    # image_id별 어노테이션 모으기
    anns_per_img = defaultdict(list)
    for ann in data['annotations']:
        anns_per_img[ann['image_id']].append(ann)

    # 이미지 단위로 YOLO txt 생성
    for img_id, anns in anns_per_img.items():
        img_file = img_id_to_file[img_id]
        label_path = os.path.join(target_yolo_dir, img_file.rsplit('.', 1)[0] + ".txt")
        image_path = os.path.join(source_img_dir, img_file)

        if not os.path.exists(image_path):
            print(f"[경고] 이미지 없음: {image_path}")
            continue

        # 이미지 크기 획득 (정규화에 필요)
        img = Image.open(image_path)
        width, height = img.size

        with open(label_path, 'w', encoding='utf-8') as f:
            for ann in anns:
                # COCO bbox: 좌상단 x,y와 너비/높이 (pixel)
                x, y, w, h = ann['bbox']

                # YOLO bbox: 중심 좌표와 너비/높이 (0~1 정규화)
                xc = (x + w / 2) / width
                yc = (y + h / 2) / height
                wn = w / width
                hn = h / height

                # 카테고리 id → YOLO class index
                cls_idx = master_cat_id_to_idx.get(ann['category_id'])
                if cls_idx is None:
                    print(f"[경고] category_id {ann['category_id']}가 master categories에 없음")
                    continue

                f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")


def count_drugs_from_filename(filename):
    """
    파일명에서 약물 개수를 추정합니다.
    예) "K-003351-020238-033880_..." → 'K-' 제거 후 하이픈으로 분리된 토큰 개수 = 3

    Args:
        filename (str): 이미지 파일명 (확장자 포함).

    Returns:
        int: 파일명 규칙으로부터 유추한 약물 개수.
    """
    prefix = filename.split('_')[0]  # 'K-...' 등 앞부분만 취함
    if prefix.startswith("K-"):
        prefix = prefix[2:]           # 'K-' 제거
    return len(prefix.split('-'))     # 하이픈 수 + 1


def clean_invalid_yolo_files_from_dir(merged_json_path, image_dir, label_dir):
    """
    파일명 기반 약물 수와 실제 어노테이션 내 'category 종류 수'를 비교하여
    불완전한 데이터(어노 수 < 파일명에서 유추한 약물 개수)를 삭제합니다.

    Args:
        merged_json_path (str): 병합된 COCO JSON 경로(이미지/어노/카테고리 포함).
        image_dir (str): 원본 이미지 폴더.
        label_dir (str): 원본 라벨(JSON) 폴더(재귀 탐색).

    동작:
        - merged_json에서 image_id별로 등장한 category_id의 unique 개수를 계산합니다.
        - image_dir의 각 .png 파일에 대해, 파일명에서 약물 개수를 추정합니다.
        - 실제 카테고리 수 < 파일명 유추 수인 경우 이미지를 삭제하고,
          label_dir 아래 해당 베이스 이름의 .json 라벨 파일들도 모두 삭제합니다.
    """
    with open(merged_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("📂 JSON 로드 완료")

    # file_name → image_id 매핑
    file_to_id = {img["file_name"]: img["id"] for img in data["images"]}

    # image_id별로 등장한 category_id set
    image_id_to_cat_ids = defaultdict(set)
    for ann in data["annotations"]:
        image_id_to_cat_ids[ann["image_id"]].add(ann["category_id"])

    removed = 0
    print(f"🔍 검사 중: {image_dir} / {label_dir}")

    # 현재 로직은 PNG만 검사 (필요시 확장자 범위를 넓히세요)
    for fname in os.listdir(image_dir):
        if not fname.endswith(".png"):
            continue

        base = os.path.splitext(fname)[0]
        drug_count = count_drugs_from_filename(fname)

        image_id = file_to_id.get(fname)
        if image_id is None:
            print(f"⚠️ JSON에 없음: {fname}")
            continue

        actual_category_count = len(image_id_to_cat_ids[image_id])

        # 실제 어노테이션의 카테고리 종류 수가 더 적으면 불완전으로 간주
        if actual_category_count < drug_count:
            # 이미지 삭제
            img_path = os.path.join(image_dir, fname)
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"🗑️ 이미지 삭제: {img_path}")

            # 라벨 JSON(하위 폴더 포함) 삭제
            for jp in glob.glob(os.path.join(label_dir, "**", base + ".json"), recursive=True):
                if os.path.exists(jp):
                    os.remove(jp)
                    print(f"🗑️ JSON 삭제: {jp}")
            removed += 1

    print(f"\n✅ 총 삭제된 항목 수: {removed}개\n")


def get_image_files_from_coco_json(json_path):
    """
    COCO JSON의 images.file_name 목록을 추출합니다.

    Args:
        json_path (str): COCO JSON 경로.

    Returns:
        list[str]: 이미지 파일명 리스트.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [img['file_name'] for img in data.get('images', [])]


def copy_images(file_names, source_img_dir, target_img_dir):
    """
    주어진 파일명 목록을 source → target으로 복사합니다.

    Args:
        file_names (list[str]): 복사할 이미지 파일명들.
        source_img_dir (str): 원본 이미지 폴더.
        target_img_dir (str): 복사 대상 폴더.

    Note:
        - 존재하지 않는 파일은 경고만 출력하고 건너뜁니다.
    """
    os.makedirs(target_img_dir, exist_ok=True)
    copied = 0
    for fname in file_names:
        sp = os.path.join(source_img_dir, fname)
        tp = os.path.join(target_img_dir, fname)
        if os.path.exists(sp):
            shutil.copy2(sp, tp)
            copied += 1
        else:
            print(f"[WARNING] 원본 이미지 없음: {sp}")
    print(f"이미지 복사 완료: {copied}개")


def split_train_val(image_dir, label_dir, train_img_dir, val_img_dir, train_label_dir, val_label_dir, val_ratio=0.15, seed=42):
    """
    이미지/라벨을 학습(train)과 검증(val)으로 분리합니다.

    Args:
        image_dir (str): 이미지 폴더(분리 전).
        label_dir (str): YOLO 라벨(.txt) 폴더(분리 전).
        train_img_dir (str): 학습 이미지 대상 폴더.
        val_img_dir (str): 검증 이미지 대상 폴더.
        train_label_dir (str): 학습 라벨 대상 폴더.
        val_label_dir (str): 검증 라벨 대상 폴더.
        val_ratio (float): 검증 비율(0~1), 기본 0.15.
        seed (int): 셔플 시드.

    동작:
        - image_dir에서 png/jpg/jpeg만 대상으로 정렬 후 셔플합니다.
        - 앞쪽 val_ratio 비율만큼을 val로, 나머지는 train으로 이동합니다.
        - 이미지/라벨 쌍을 각각 대응 폴더로 이동합니다(존재하는 경우에만).
    """
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # 지원 확장자
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    random.seed(seed)
    random.shuffle(image_files)

    val_count = int(len(image_files) * val_ratio)
    val_images = set(image_files[:val_count])

    for img_file in image_files:
        lbl = os.path.splitext(img_file)[0] + ".txt"
        src_img = os.path.join(image_dir, img_file)
        src_lbl = os.path.join(label_dir, lbl)

        # 대상 경로 결정
        if img_file in val_images:
            dst_img = os.path.join(val_img_dir, img_file)
            dst_lbl = os.path.join(val_label_dir, lbl)
        else:
            dst_img = os.path.join(train_img_dir, img_file)
            dst_lbl = os.path.join(train_label_dir, lbl)

        # 실제 파일이 있는 경우에만 이동
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)


def create_yolo_yaml(root_dir, merged_json_path, yaml_path):
    """
    YOLO 학습용 data.yaml을 생성합니다.

    Args:
        root_dir (str): 데이터셋 루트 경로 (train/val 상대경로 기준).
        merged_json_path (str): 카테고리 정보를 추출할 COCO JSON 경로.
        yaml_path (str): 생성할 yaml 파일 경로.

    동작:
        - get_master_categories로 {cat_id: index}, {cat_id: name}을 얻습니다.
        - index 기준으로 names를 채우고, path/train/val/nc 필드를 기록합니다.
    """
    cat_id_to_index, cat_id_to_name = get_master_categories([merged_json_path])

    # YOLO 인덱스 → 클래스명 딕셔너리
    names_dict = {idx: cat_id_to_name[cat_id] for cat_id, idx in cat_id_to_index.items()}

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {root_dir}\n")            # 데이터셋 루트(상대 경로 기준점)
        f.write(f"train: images/train_images\n")  # 학습 이미지 경로(루트 기준)
        f.write(f"val: images/val_images\n")      # 검증 이미지 경로(루트 기준)
        f.write(f"nc: {len(names_dict)}\n")       # 클래스 수
        f.write("names:\n")                       # 인덱스 순서대로 클래스명
        for idx in range(len(names_dict)):
            f.write(f"  {idx}: '{names_dict[idx]}'\n")

    print(f"data.yaml 파일이 생성되었습니다: {yaml_path}")
