import requests
import os
import json

# 设置 FastAPI 服务器地址
# api_url = "http://127.0.0.1:8000/process_video/"
api_url = "http://192.168.8.72:8000/process_video/"


# 设置视频目录路径
video_dir = "D:/HuangHuaGang/video"

# 构造请求参数
params = {
    "video_dir": video_dir
}

# 发送 POST 请求调用 API
try:
    response = requests.post(api_url, params=params)

    # 检查请求是否成功
    if response.status_code == 200:
        # 解析 JSON 响应
        result = response.json()

        # 提取识别的文本
        recognized_text = result.get("text")
        json_file_path = result.get("file")

        # 打印识别结果
        print("识别结果文本:", recognized_text)
        print("结果已保存到:", json_file_path)

        # 可选：如果你想将结果保存到其他地方，重新读取并处理结果
        if json_file_path and os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                print("从JSON文件读取的内容:", saved_data)
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print("响应内容:", response.text)

except Exception as e:
    print(f"请求处理时发生错误: {e}")
