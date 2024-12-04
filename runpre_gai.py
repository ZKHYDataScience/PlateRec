# from fastapi import FastAPI, Query
# from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import torch
import subprocess
import re
import json
import numpy as np
from paddleocr import PaddleOCR


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
    use_gpu=True  
)


            
def process_frame_with_yolo_v2(img_pil):
    """使用 YOLOv8v2 处理图像并返回裁剪后的图像和位置信息"""
    results_yolo_v2 = yolo_model_v2(img_pil)
    crops_with_positions = []
    
    for box in results_yolo_v2[0].boxes:
        score = box.conf.item()

        if score > 0.80:
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
            print(xmin, ymin, xmax, ymax)
            cropped_img = img_pil.crop((xmin, ymin, xmax, ymax))
            center_y = (ymin + ymax) / 2
            center_x = (xmin + xmax) / 2
            crops_with_positions.append((cropped_img, (center_x, center_y)))
    
    return crops_with_positions


def sort_text_by_position(texts_with_positions):
    """根据位置对文本进行排序"""
    sorted_by_y = sorted(texts_with_positions, key=lambda x: x[1][1])
    
    if len(sorted_by_y) == 3:
        top_text = sorted_by_y[0]
        bottom_texts = sorted(sorted_by_y[1:], key=lambda x: x[1][0])
        return [top_text[0]] + [text[0] for text in bottom_texts]
    
    return [text[0] for text in sorted_by_y]

def is_valid_text(result_text):
    """验证识别出的文本是否符合条件"""
    if not (9 <= len(result_text) <= 15):  # 字符串长度限制
        return False
    if not re.match(r'^[A-Za-z0-9 ]+$', result_text):  # 包含字母、数字或空格
        return False
    if not re.search(r'[A-Za-z]', result_text):  # 必须有字母
        return False
    if not re.search(r'[0-9]', result_text):  # 必须有数字
        return False
    return True

def process_video(video_dir):
    """处理单个视频文件"""
    cap = cv2.VideoCapture(video_dir)
    if not cap.isOpened():
        return ""
    
    frame_count = 0
    best_result = None
    successful_detections = 0
    detection_results = []
    
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
                        avg_confidence = 0
                        total_confidence =0
                        for crop, position in crops_with_positions:
                            enhanced_img_np = np.array(crop)
                            enhanced_img_np = enhanced_img_np[:, :, ::-1]
                            text = ocr.ocr(enhanced_img_np, cls=False,det=False)
                            # text = run_paddleocr_on_image(crop)
                            if text:
                                detect_text = text[0][0][0]
                                confidence = text[0][1]
                                avg_confidence += confidence
                                total_confidence += 1
                                texts_with_positions.append((detect_text, position))
                                
                        if total_confidence > 0:
                            avg_confidence /= total_confidence
                        
                        if texts_with_positions:
                            sorted_texts = sort_text_by_position(texts_with_positions)
                            result_text = ''.join(sorted_texts)
                            
                            if is_valid_text(result_text):
                                # if best_result is None:
                                #     best_result = result_text  
                                successful_detections += 1
                                detection_results.append({
                                    "text": result_text,
                                    "avg_confidence": avg_confidence
                                })                            
                                if successful_detections >= 10:
                                    best_result = max(detection_results, key=lambda x: x["avg_confidence"])
                                    
                                    cap.release()
                                    return best_result

    cap.release()
    if detection_results:  
        best_result = max(detection_results, key=lambda x: x["avg_confidence"])
    return best_result


def detect_text(video_dir,out_path):
    """处理视频并返回文本检测结果的API端点"""
    
    # # 检查视频文件是否存在
    # if not os.path.exists(video_dir):
    #     result = []  # 空数组表示失败
    #     with open(os.path.join(out_path, "detection_results.json"), 'w', encoding='utf-8') as f:
    #         json.dump(result, f, ensure_ascii=False)
    #     return JSONResponse(content=result)
    # 
    # # 检查文件是否是MP4格式
    # if not video_dir.lower().endswith('.mp4'):
    #     result = []  # 空数组表示失败
    #     with open(os.path.join(out_path, "detection_results.json"), 'w', encoding='utf-8') as f:
    #         json.dump(result, f, ensure_ascii=False)
    #     return JSONResponse(content=result)

    # 处理视频
    text_result = process_video(video_dir)
    
    # 确保输出路径存在
    os.makedirs(out_path, exist_ok=True)
    
    # 准备结果（使用数组格式）
    result = [text_result] if text_result else []
    
    # 保存结果到JSON文件
    output_file = os.path.join(out_path, "detection_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    
    # 返回结果数组
    # return JSONResponse(content=result)

# 启动服务
if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8002)D:\BaiduNetdiskDownload\deploy\run.py
    detect_text(r'D:\company\HuangHuaGang\video\2024-08-01-14-37-51.mp4',r'D:\BaiduNetdiskDownload\deploy\input\res')
    