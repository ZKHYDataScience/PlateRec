
# 运煤车车号识别系统


## 项目简介

本项目旨在通过YOLOv8模型和PaddleOCR进行运煤车车号的识别，并通过FastAPI提供接口供其他应用调用。项目未来计划添加车厢异物识别和车厢图像拼接功能。

## 项目结构

```
DetRec-ID/
│
├── ID-stream.py      # 实现车厢号识别的主功能
├── ID-call.py        # 用于调用识别API的客户端脚本
└── README.md         # 项目说明文档
```

## 功能描述

### 车厢号识别 - ID-stream.py

- **YOLOv8**: 用于检测车厢号的位置。
- **PaddleOCR**: 用于识别车厢号中的字符。
- **FastAPI**: 提供HTTP接口，通过API调用识别功能。
- 该脚本会从指定的视频文件目录中提取视频，逐帧分析车厢号(10帧），返回最优的识别结果。

### 客户端调用 - ID-call.py

- 使用 `requests` 模块发送POST请求，调用FastAPI接口。
- 输入为视频文件的路径，输出为识别的文本和保存的JSON文件。

## 使用方法

### 环境准备

1. 克隆本项目到本地：

    ```bash
    git clone https://github.com/your-username/DetRec-ID.git
    ```

2. 安装所需的依赖项：

    ```bash
    pip install -r requirements.txt
    ```

3. 下载并准备YOLOv8模型和PaddleOCR所需的模型文件，确保在`ID-stream.py`中正确设置模型路径。

### 启动服务

1. 运行识别服务（FastAPI）：

    ```bash
    python ID-stream.py
    ```

    服务启动后，将在`0.0.0.0:8000`上运行。

### 调用API

1. 使用`ID-call.py`进行调用：

    ```bash
    python ID-call.py
    ```

2. 在`ID-call.py`中设置API的URL地址和视频文件路径，调用后结果将保存为JSON文件。

## 后续需求补充

- 添加车厢异物识别功能接口
- 添加车厢图像拼接功能接口
- 进一步优化识别准确性

## 作者

