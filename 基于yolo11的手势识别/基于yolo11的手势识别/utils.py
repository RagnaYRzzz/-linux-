import os
import sys
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QFont
from PySide6.QtWidgets import QMessageBox

# 确保在所有系统中中文显示正常
def ensure_font_support():
    """确保应用程序支持中文字体"""
    if sys.platform == 'win32':
        # Windows系统
        font = QFont("Microsoft YaHei", 9)  # 微软雅黑
    elif sys.platform == 'darwin':
        # macOS系统
        font = QFont("PingFang SC", 9)  # 苹果平方
    else:
        # Linux系统
        font = QFont("Noto Sans CJK SC", 9)  # Noto Sans CJK
    
    return font

# 转换OpenCV图像到Qt图像
def cv_to_qt_image(cv_img):
    """
    将OpenCV图像转换为Qt图像
    
    Args:
        cv_img: OpenCV格式的图像（numpy数组）
        
    Returns:
        QImage对象
    """
    if cv_img is None:
        return QImage()
        
    # 确保图像是RGB格式（OpenCV默认是BGR）
    if len(cv_img.shape) == 3:
        h, w, ch = cv_img.shape
        if ch == 3:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            return QImage(rgb_img.data, w, h, w * ch, QImage.Format_RGB888)
        elif ch == 4:
            return QImage(cv_img.data, w, h, w * ch, QImage.Format_RGBA8888)
    
    # 灰度图像
    h, w = cv_img.shape
    return QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
    
def cv_to_pixmap(cv_img, target_size=None):
    """
    将OpenCV图像转换为Qt Pixmap并调整大小
    
    Args:
        cv_img: OpenCV格式的图像
        target_size: 目标大小，如果指定则调整图像大小
        
    Returns:
        QPixmap对象
    """
    qt_img = cv_to_qt_image(cv_img)
    pixmap = QPixmap.fromImage(qt_img)
    
    if target_size:
        pixmap = pixmap.scaled(
            target_size.width(), 
            target_size.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
    
    return pixmap

def create_styled_button(text, color="#4169E1", hover_color="#3151B5", 
                        text_color="#FFFFFF", border_radius=6, height=36):
    """
    创建具有现代科技感的按钮样式
    
    Args:
        text: 按钮文本
        color: 按钮背景颜色
        hover_color: 鼠标悬停时的颜色
        text_color: 文本颜色
        border_radius: 边框圆角半径
        height: 按钮高度
        
    Returns:
        样式表字符串
    """
    style = f"""
        QPushButton {{
            background-color: {color};
            color: {text_color};
            border: none;
            border-radius: {border_radius}px;
            padding: 8px 16px;
            font-weight: bold;
            height: {height}px;
        }}
        QPushButton:hover {{
            background-color: {hover_color};
        }}
        QPushButton:pressed {{
            background-color: {color};
        }}
        QPushButton:disabled {{
            background-color: #B0B0B0;
            color: #E0E0E0;
        }}
    """
    return style

def export_detection_history(history, export_path):
    """
    导出检测历史为CSV文件
    
    Args:
        history: 检测历史记录列表
        export_path: 导出文件路径
    """
    if not history:
        return False
        
    # 创建数据帧
    df = pd.DataFrame(history)
    
    # 确保目录存在
    export_dir = os.path.dirname(export_path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    # 导出到CSV
    df.to_csv(export_path, index=False, encoding='utf-8-sig')  # 使用带BOM的UTF-8确保Excel正确显示中文
    
    return True

def show_info_message(parent, title, message):
    """显示信息对话框"""
    QMessageBox.information(parent, title, message)
    
def show_error_message(parent, title, message):
    """显示错误对话框"""
    QMessageBox.critical(parent, title, message)
    
def show_question_message(parent, title, message):
    """显示询问对话框，返回用户选择（是/否）"""
    return QMessageBox.question(
        parent, 
        title, 
        message, 
        QMessageBox.Yes | QMessageBox.No
    ) == QMessageBox.Yes

def get_default_save_path(file_type):
    """获取默认保存路径"""
    base_dir = os.path.join(os.path.expanduser("~"), "手语识别结果")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if file_type == "image":
        return os.path.join(base_dir, f"检测结果_{timestamp}.jpg")
    elif file_type == "video":
        return os.path.join(base_dir, f"视频检测_{timestamp}.mp4")
    elif file_type == "csv":
        return os.path.join(base_dir, f"检测历史_{timestamp}.csv")
    else:
        return os.path.join(base_dir, f"结果_{timestamp}") 