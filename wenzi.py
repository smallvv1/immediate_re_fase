import os
import sys
import cv2
import glob
import re
from pathlib import Path
from typing import Dict, List
from ultralytics import YOLO

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from paddleocr import PaddleOCR

MODEL_PATH   = r"D:\1\EzYOLO-main\runs\detect\runs\train\exp_2\weights\best.pt" # 使用原始字符串避免转义
#IMAGE_FOLDER = r"D:\1000\pliers_images" # 使用原始字符串
IMAGE_FOLDER = r"D:\1\EzYOLO-main"  # 默认为当前目录，扫描capture_clear_*.jpg

CONF_THRESH  = 0.4
OUTPUT_LOG   = os.path.join(os.path.dirname(__file__), "inference_log.txt")
CAPTURE_PATTERN = "capture_clear_*.jpg"  # 匹配test_hk_opecv.py的输出文件
# ====================================================

# 初始化 OCR
ocr = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)
model = YOLO(MODEL_PATH)


def _extract_digits(text: str) -> str:
    if not text:
        return ""
    return "".join(re.findall(r"\d", str(text)))


def _resize_crop_for_ocr(crop, max_side: int = 640):
    h, w = crop.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return crop
    ratio = max_side / float(longest)
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _extract_text_with_engine(engine, crop) -> str:
    try:
        res = list(engine.predict(crop))
    except Exception:
        return ""
    if not res:
        return ""
    first = res[0]
    if isinstance(first, dict) and first.get("rec_texts"):
        text = " ".join(first["rec_texts"])
    elif hasattr(first, "rec_texts") and getattr(first, "rec_texts"):
        text = " ".join(first.rec_texts)
    elif isinstance(first, list):
        text = " ".join([line[1][0] for line in first if len(line) > 1 and len(line[1]) > 0])
    else:
        text = ""
    return _extract_digits(text.strip())


def detect_and_ocr_with_wenzi(
    image_paths: List[Path],
    model,
    ocr_engine,
    conf_threshold: float,
    code_pattern: str,
    annotated_dir: Path,
    min_box_area: int = 900,
    max_die_boxes: int = 0,
) -> List[Dict]:
    reports: List[Dict] = []
    annotated_dir.mkdir(parents=True, exist_ok=True)
    code_re = re.compile(code_pattern, re.IGNORECASE)

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            reports.append(
                {
                    "image_path": str(image_path),
                    "status": "failed",
                    "reason": "image_read_failed",
                    "detections": [],
                }
            )
            continue

        results = model(str(image_path), conf=conf_threshold, verbose=False, imgsz=640)
        result = results[0]
        names = result.names if hasattr(result, "names") else {}
        die_candidates = []
        other_candidates = []

        for box in result.boxes:
            class_id = int(box.cls.item()) if box.cls is not None else -1
            class_name = names.get(class_id, str(class_id)) if class_id >= 0 else "unknown"
            conf = float(box.conf.item()) if box.conf is not None else 0.0

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area < int(min_box_area):
                continue

            item = (area, class_id, class_name, conf, x1, y1, x2, y2)
            if class_name.lower() == "die":
                die_candidates.append(item)
            else:
                other_candidates.append(item)

        if max_die_boxes and max_die_boxes > 0 and len(die_candidates) > max_die_boxes:
            die_candidates = sorted(die_candidates, key=lambda x: x[0], reverse=True)[:max_die_boxes]
        candidates = die_candidates + other_candidates

        detections = []
        for _area, class_id, class_name, conf, x1, y1, x2, y2 in candidates:
            crop = img[y1:y2, x1:x2]
            crop = _resize_crop_for_ocr(crop, max_side=640)
            should_ocr = class_name.lower() == "die"
            if ocr_engine is not None and should_ocr:
                text = _extract_text_with_engine(ocr_engine, crop).strip()
                ocr_texts = [text] if text else []
                ocr_codes = [m.group(0).upper() for m in code_re.finditer(text)] if text else []
                if text.isdigit():
                    synthetic_code = f"TH{text.zfill(2)}"
                    if code_re.fullmatch(synthetic_code):
                        ocr_codes.append(synthetic_code)
                ocr_codes = sorted(list({c for c in ocr_codes if c}))
            else:
                ocr_texts = []
                ocr_codes = []

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "ocr_texts": ocr_texts,
                    "ocr_codes": ocr_codes,
                }
            )

        annotated = result.plot()
        annotated_path = annotated_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        cv2.imwrite(str(annotated_path), annotated)

        reports.append(
            {
                "image_path": str(image_path),
                "status": "ok",
                "detection_count": len(detections),
                "annotated_image_path": str(annotated_path),
                "detections": detections,
            }
        )

    return reports

def process_images_from_camera(image_folder=None, pattern=CAPTURE_PATTERN):
    """
    处理海康相机拍摄的图像
    Args:
        image_folder: 图像所在文件夹，默认为当前目录
        pattern: 匹配模式，默认为capture_clear_*.jpg
    """
    if image_folder is None:
        image_folder = IMAGE_FOLDER
    
    with open(OUTPUT_LOG, "a", encoding="utf-8") as log_file:
        def log_line(message=""):
            print(message)
            log_file.write(f"{message}\n")

        log_line("===== 配件文字识别（海康相机输出）=====")
        log_line()

        # 获取所有匹配模式的图片文件
        file_pattern = os.path.join(image_folder, pattern)
        image_files = sorted(glob.glob(file_pattern), key=os.path.getctime, reverse=True)
        
        if not image_files:
            log_line(f"[WARN] 未找到匹配模式 '{pattern}' 的图片文件在 {image_folder}")
            return

        # 遍历图片（从最新开始）
        for img_path in image_files:
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                log_line(f"[WARN] 图片读取失败: {filename}，已跳过")
                continue
            
            results = model(img_path, conf=CONF_THRESH, verbose=False)

            log_line(f"========== {filename} ==========")

            # 逐个检测框识别文字
            for idx, box in enumerate(results[0].boxes):
                class_id = int(box.cls.item()) if box.cls is not None else -1
                cls_name = results[0].names.get(class_id, str(class_id)) if class_id >= 0 else "unknown"

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    log_line(f"[{cls_name} {idx+1}] 检测框无效，已跳过")
                    continue

                crop = img[y1:y2, x1:x2]

                # OCR 识别
                res = list(ocr.predict(crop))
                text = ""
                if res:
                    first = res[0]
                    if isinstance(first, dict) and first.get("rec_texts"):
                        text = " ".join(first["rec_texts"])
                    elif hasattr(first, "rec_texts") and getattr(first, "rec_texts"):
                        text = " ".join(first.rec_texts)
                    elif isinstance(first, list):
                        text = " ".join([line[1][0] for line in first if len(line) > 1 and len(line[1]) > 0])

                log_line(f"[{cls_name} {idx+1}] 识别文字：{text.strip()}")

            log_line()

print(f"识别结果已保存到：{OUTPUT_LOG}")

if __name__ == "__main__":
    # 如果直接运行此脚本，处理最新的相机拍摄图像
    process_images_from_camera()
