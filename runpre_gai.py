from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import torch
import re
import json
import numpy as np
from paddleocr import PaddleOCR

app = FastAPI()

yolo_model = YOLO('./weights/v8/best.pt')
yolo_model_v2 = YOLO('./weights/v8v2/best.pt')

if torch.cuda.is_available():
    yolo_model.to('cuda')
    yolo_model_v2.to('cuda')
else:
    yolo_model.to('cpu')
    yolo_model_v2.to('cpu')

ocr = PaddleOCR(
    det_model_dir='./weights/det/Teacher',
    rec_model_dir='./weights/rec',
    rec_char_dict_path="./weights/dict/en_dict.txt",
    use_gpu=torch.cuda.is_available()
)

def process_frame_with_yolo_v2(img_pil):
    """Process image with YOLOv8v2 and return cropped images and positions."""
    results_yolo_v2 = yolo_model_v2(img_pil)
    crops_with_positions = []

    for box in results_yolo_v2[0].boxes:
        score = box.conf.item()

        if score > 0.80:
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
            cropped_img = img_pil.crop((xmin, ymin, xmax, ymax))
            center_y = (ymin + ymax) / 2
            center_x = (xmin + xmax) / 2
            crops_with_positions.append((cropped_img, (center_x, center_y), score))

    return crops_with_positions

def sort_text_by_position(texts_with_positions):
    """Sort text by position."""
    sorted_by_y = sorted(texts_with_positions, key=lambda x: x[1][1])

    if len(sorted_by_y) == 3:
        top_text = sorted_by_y[0]
        bottom_texts = sorted(sorted_by_y[1:], key=lambda x: x[1][0])
        return [top_text[0]] + [text[0] for text in bottom_texts]

    return [text[0] for text in sorted_by_y]

def is_valid_text(result_text):
    """Validate the recognized text."""
    if not (9 <= len(result_text) <= 15):  
        return False
    if not re.match(r'^[A-Za-z0-9 ]+$', result_text):  
        return False
    if not re.search(r'[A-Za-z]', result_text):  
        return False
    if not re.search(r'[0-9]', result_text):  
        return False
    return True

def process_video(video_dir):
    """Process a single video file."""
    cap = cv2.VideoCapture(video_dir)
    if not cap.isOpened():
        return None

    frame_count = 0
    best_result = None
    best_result_info = {
        'text': None,
        'frame_avg_confidence': 0,
        'detections_count': 0
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 20 == 0:
            height, width, _ = frame.shape
            cropped_frame = frame[150:height - 150, :]
            img_pil = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))

            results_yolo = yolo_model(img_pil)

            for box in results_yolo[0].boxes:
                score = box.conf.item()
                cls = box.cls.item()

                if score > 0.80 and int(cls) == 0:
                    xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                    first_crop = img_pil.crop((xmin, ymin, xmax, ymax))

                    crops_with_positions = process_frame_with_yolo_v2(first_crop)

                    if crops_with_positions:
                        texts_with_positions = []
                        frame_detections = []

                        for crop, position, detection_confidence in crops_with_positions:
                            enhanced_img_np = np.array(crop)
                            enhanced_img_np = enhanced_img_np[:, :, ::-1]
                            text = ocr.ocr(enhanced_img_np, cls=False, det=False)

                            if text:
                                texts_with_positions.append((text[0][0][0], position))
                                frame_detections.append(detection_confidence)

                        if texts_with_positions:
                            sorted_texts = sort_text_by_position(texts_with_positions)
                            result_text = ''.join(sorted_texts)

                            if is_valid_text(result_text):
                                frame_avg_confidence = np.mean(frame_detections)

                                if (best_result is None or 
                                    frame_avg_confidence > best_result_info['frame_avg_confidence']):
                                    best_result = result_text
                                    best_result_info = {
                                        'text': result_text,
                                        'frame_avg_confidence': frame_avg_confidence,
                                        'detections_count': len(frame_detections)
                                    }

    cap.release()
    return best_result_info

@app.get("/process_videos/")
async def process_video_endpoint(
    video_file: str = Query(r"D:\company\HuangHuaGang\video\2024-08-01-14-37-51.mp4", description="Path to the video file"),
    out_path: str = Query(r"D:\BaiduNetdiskDownload\deploy\input\res", description="Path to save recognition results")
):
    """API endpoint to process video and return recognition results."""
    text_result_info = process_video(video_file)

    os.makedirs(out_path, exist_ok=True)

    result = [text_result_info['text']] if text_result_info and text_result_info['text'] else []

    output_file = os.path.join(out_path, "detection_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

    return JSONResponse(result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
