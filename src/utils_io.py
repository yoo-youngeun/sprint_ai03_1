# -*- coding: utf-8 -*-
"""
utils_io.py
- 파일 입출력, 경로 생성, 이미지/라벨 확인 등 프로젝트 전역 유틸 함수 모음
"""
import os
import shutil

# 프로젝트 전역에서 통일해서 쓰기 위한 이미지 확장자 정의
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def find_image_path(images_dir, base_name, exts=IMG_EXTS):
    """
    주어진 base_name과 일치하는 실제 이미지 파일 경로를 반환.
    예: base='foo' → foo.png, foo.jpg, ...
    
    Args:
        images_dir (str): 이미지 폴더 경로
        base_name (str): 파일명(확장자 제외)
        exts (tuple): 탐색할 확장자 목록
        
    Returns:
        str | None: 일치하는 이미지 경로, 없으면 None
    """
    for ext in exts:
        p = os.path.join(images_dir, base_name + ext)
        if os.path.exists(p):
            return p
    return None


def check_images_labels(img_dir, labels_dir):
    """
    이미지-라벨 매칭 여부를 검사 (확장자 다양성 고려)
    
    Args:
        img_dir (str): 이미지 디렉터리
        labels_dir (str): 라벨 디렉터리
    
    Returns:
        bool: 모든 매칭이 정상인지 여부
    """
    if not os.path.exists(labels_dir):
        print(f"❌ 라벨 폴더가 없습니다: {labels_dir}")
        return False

    img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS]
    img_names = set(os.path.splitext(f)[0] for f in img_files)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    label_names = set(os.path.splitext(f)[0] for f in label_files)

    print(f"이미지 파일 개수: {len(img_files)}")
    print(f"라벨 파일 개수: {len(label_files)}")

    # 이미지 있는데 라벨 없는 경우
    missing_labels = img_names - label_names
    # 라벨 있는데 이미지 없는 경우
    missing_images = label_names - img_names

    if missing_labels:
        print(f"❌ 라벨이 없는 이미지 파일 ({len(missing_labels)}개):")
        for name in sorted(missing_labels):
            print(f"  - {name}")
    else:
        print("✅ 모든 이미지에 대응하는 라벨이 존재합니다.")

    if missing_images:
        print(f"⚠️ 라벨은 있으나 이미지가 없는 파일 ({len(missing_images)}개):")
        for name in sorted(missing_images):
            print(f"  - {name}")
    else:
        print("✅ 모든 라벨 파일에 대응하는 이미지가 존재합니다.")

    return len(missing_labels) == 0 and len(missing_images) == 0


def count_classes_in_yolo_txt(labels_dir):
    """
    (YOLO 형식) 라벨 텍스트 파일에서 클래스별 객체 개수를 세는 함수
    
    Args:
        labels_dir (str): YOLO 라벨(.txt) 폴더 경로
        
    Returns:
        dict: {클래스 ID: 객체 수}
    """
    from collections import defaultdict
    class_count = defaultdict(int)

    for root, _, files in os.walk(labels_dir):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                cid = int(float(parts[0]))
                            except ValueError:
                                print(f"[경고] 클래스 ID 변환 실패: {parts[0]} in {file}")
                                continue
                            class_count[cid] += 1
    return class_count


def clear_yolo_dir(paths_to_clear, paths_to_make):
    """
    YOLO 학습/출력 폴더 초기화
    
    Args:
        paths_to_clear (list): 삭제할 디렉터리 경로 리스트
        paths_to_make (list): 새로 생성할 디렉터리 경로 리스트
    """
    for d in paths_to_clear:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"✅ Deleted: {d}")
        else:
            print(f"⚠️ Not found: {d}")
    for d in paths_to_make:
        os.makedirs(d, exist_ok=True)
