from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image
import os
import cv2
import numpy as np
import torch
import paddle
import json

# 初始化 FastAPI
app = FastAPI()

# 加载 YOLOv8 模型
yolo_model = YOLO('D:/company/vscode/DetRec/wrap/weights/best.pt')  # YOLOv8 自定义模型路径

# 将模型加载到 GPU（如果可用）
if torch.cuda.is_available():
    yolo_model.to('cuda')
    print("YOLOv8 is using CUDA (GPU).")
else:
    yolo_model.to('cpu')
    print("YOLOv8 is using CPU.")

# 初始化 PaddleOCR 并指定使用 GPU
ocr = PaddleOCR(
    det_model_dir='D:/company/vscode/DetRec/wrap/Teacher',
    rec_model_dir='D:/company/vscode/DetRec/wrap/rec2',
    rec_char_dict_path="D:/company/vscode/DetRec/wrap/en_dict.txt",
    use_gpu=True  # 指定使用 GPU
)

# 检查 PaddleOCR 是否在使用 CUDA
if paddle.device.is_compiled_with_cuda():
    print("PaddleOCR is using CUDA (GPU).")
else:
    print("PaddleOCR is using CPU.")

# 处理目录中的视频并返回识别结果的 API
@app.post("/process_video/")
async def process_video(video_dir: str = Query(...)):
    # 检查路径是否存在
    if not os.path.exists(video_dir):
        return JSONResponse(content={"error": "Video directory not found."}, status_code=404)

    # 获取目录下的所有 .mp4 文件
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    # 确保只有一个视频文件
    if len(video_files) == 0:
        return JSONResponse(content={"error": "No video files found in the directory."}, status_code=400)
    elif len(video_files) > 1:
        return JSONResponse(content={"error": "Multiple video files found, only one allowed."}, status_code=400)
    
    # 获取唯一的视频文件路径
    video_path = os.path.join(video_dir, video_files[0])

    # 加载视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return JSONResponse(content={"error": "Failed to open video."}, status_code=400)

    frame_count = 0
    best_result = None
    result_count = 0

    while cap.isOpened() and result_count < 10:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 每隔20帧处理一次
        if frame_count % 20 == 0:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 使用 YOLOv8 进行检测
            results = yolo_model(img_pil)

            for box in results[0].boxes:
                score = box.conf.item()
                if score > 0.85:  # 仅处理置信度大于 85% 的目标
                    # 获取目标框坐标并裁剪
                    xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                    cropped_img = img_pil.crop((xmin, ymin, xmax, ymax))

                    # 将裁剪后的图像转换为 NumPy 数组（OCR 需要 NumPy 数组格式）
                    cropped_img_np = np.array(cropped_img)

                    # 使用 PaddleOCR 进行文本识别
                    ocr_result = ocr.ocr(cropped_img_np, cls=True)
                    if ocr_result[0]:
                        # 拼接识别的文本
                        recognized_text = ''.join([d[1][0] for line in ocr_result for d in line])

                        # 计算平均置信度
                        avg_confidence = sum([d[1][1] for line in ocr_result for d in line]) / len(ocr_result[0])

                        # 检查是否为最佳结果
                        if best_result is None or avg_confidence > best_result["avg_confidence"]:
                            best_result = {
                                "text": recognized_text,
                                "avg_confidence": avg_confidence,
                                "frame": frame_count
                            }
                        result_count += 1

    cap.release()

    if best_result:
        # 保存结果到本地JSON
        output_path = os.path.join("D:/HuangHuaGang/output", f"result_{best_result['frame']}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"text": best_result["text"]}, f, ensure_ascii=False, indent=4)

        # 返回JSON响应
        return JSONResponse(content={"text": best_result["text"], "file": output_path})

    return JSONResponse(content={"error": "No valid results found."}, status_code=400)


# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
