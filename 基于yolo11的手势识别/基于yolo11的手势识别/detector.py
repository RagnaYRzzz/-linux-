import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

class HandSignDetector:
    def __init__(self, model_path="best.pt"):
        """
        初始化手语识别检测器
        
        Args:
            model_path: YOLO模型路径，默认为None（将使用预训练的yolo11n模型）
        """
        # 如果没有提供模型路径，使用默认模型
        if model_path is None or not os.path.exists(model_path):
            self.model = YOLO("yolo11n.pt")
        else:
            self.model = YOLO(model_path)
            
        self.results = None
        self.current_frame = None
        self.detection_history = []
        
    def detect_image(self, image_path):
        """
        检测单张图片中的手语标志
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            处理后的图像和检测结果
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到图片文件: {image_path}")
            
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
            
        # 复制原始图像用于显示
        img_display = img.copy()
        
        # 进行检测
        start_time = time.time()
        results = self.model(img)
        end_time = time.time()
        
        # 保存结果
        self.results = results
        self.current_frame = img_display
        
        # 在图像上绘制检测结果
        annotated_img = results[0].plot()
        
        # 记录检测历史
        detection_info = {
            "type": "image",
            "source": image_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detections": len(results[0].boxes),
            "processing_time": end_time - start_time,
            "classes": [int(box.cls) for box in results[0].boxes]
        }
        self.detection_history.append(detection_info)
        
        return annotated_img, results[0]
        
    def detect_video(self, video_path, output_path=None, progress_callback=None):
        """
        检测视频中的手语标志
        
        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径，默认为None（不保存）
            progress_callback: 进度回调函数
            
        Returns:
            处理后的最后一帧图像和所有帧的检测结果列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到视频文件: {video_path}")
            
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 准备输出视频
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_results = []
        frame_idx = 0
        
        start_time = time.time()
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # 进行检测
            results = self.model(frame)
            all_results.append(results[0])
            
            # 在帧上绘制检测结果
            annotated_frame = results[0].plot()
            
            # 保存到输出视频
            if output_path:
                out.write(annotated_frame)
                
            # 更新当前帧和结果（用于显示）
            self.current_frame = annotated_frame
            self.results = results[0]
            
            # 更新进度
            frame_idx += 1
            progress = frame_idx / frame_count * 100
            if progress_callback:
                progress_callback(progress, annotated_frame)
                
        # 释放资源
        cap.release()
        if output_path:
            out.release()
            
        end_time = time.time()
        
        # 记录检测历史
        detection_info = {
            "type": "video",
            "source": video_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frames": frame_count,
            "processing_time": end_time - start_time
        }
        self.detection_history.append(detection_info)
        
        return self.current_frame, all_results
        
    def detect_webcam(self, cam_id=0, stop_signal=None, frame_callback=None):
        """
        实时检测摄像头视频中的手语标志
        
        Args:
            cam_id: 摄像头ID，默认为0
            stop_signal: 停止信号函数，返回True时停止检测
            frame_callback: 每一帧处理后的回调函数
            
        Returns:
            最后一帧图像和最后的检测结果
        """
        # 打开摄像头
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头ID: {cam_id}")
            
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            # 检查是否应该停止
            if stop_signal and stop_signal():
                break
                
            success, frame = cap.read()
            if not success:
                break
                
            # 进行检测
            results = self.model(frame)
            
            # 在帧上绘制检测结果
            annotated_frame = results[0].plot()
            
            # 更新当前帧和结果
            self.current_frame = annotated_frame
            self.results = results[0]
            
            # 调用回调函数
            if frame_callback:
                frame_callback(annotated_frame, results[0])
                
            frame_count += 1
            
        # 释放资源
        cap.release()
        
        end_time = time.time()
        
        # 记录检测历史
        detection_info = {
            "type": "webcam",
            "source": f"摄像头 ID: {cam_id}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frames": frame_count,
            "processing_time": end_time - start_time
        }
        self.detection_history.append(detection_info)
        
        return self.current_frame, self.results
        
    def get_detection_history(self):
        """
        获取检测历史记录
        
        Returns:
            检测历史记录列表
        """
        return self.detection_history
        
    def save_current_frame(self, save_path):
        """
        保存当前处理过的帧
        
        Args:
            save_path: 保存路径
        
        Returns:
            是否保存成功
        """
        if self.current_frame is None:
            return False
            
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        return cv2.imwrite(save_path, self.current_frame) 