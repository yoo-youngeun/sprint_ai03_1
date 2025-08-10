def yolo_to_xyxy(bbox, img_w, img_h):
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Invalid image size")
    x, y, w, h = bbox
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2]

def xyxy_to_yolo(bbox, img_w, img_h):
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Invalid image size")
    x1, y1, x2, y2 = bbox
    x = ((x1 + x2) / 2) / img_w
    y = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [x, y, w, h]