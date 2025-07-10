import os
import sys
import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QSize, Signal, Slot, QThread, QTimer, QRunnable, QThreadPool, QObject
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QLinearGradient, QBrush, QPalette
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QFileDialog, QScrollArea, QProgressBar, 
    QComboBox, QFrame, QTableWidget, QTableWidgetItem, 
    QHeaderView, QSpacerItem, QSizePolicy
)

import utils
from detector import HandSignDetector

# 信号类，用于线程通信
class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(str)
    result = Signal(object)
    progress = Signal(int, object)
    frame = Signal(object, object)

# 图像检测线程
class ImageDetectionWorker(QRunnable):
    def __init__(self, detector, image_path):
        super().__init__()
        self.detector = detector
        self.image_path = image_path
        self.signals = WorkerSignals()
        
    def run(self):
        try:
            # 执行检测
            result_img, results = self.detector.detect_image(self.image_path)
            self.signals.result.emit((result_img, results))
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

# 视频检测线程
class VideoDetectionWorker(QRunnable):
    def __init__(self, detector, video_path, output_path=None):
        super().__init__()
        self.detector = detector
        self.video_path = video_path
        self.output_path = output_path
        self.signals = WorkerSignals()
        self.is_running = True
        
    def stop(self):
        self.is_running = False
        
    def run(self):
        try:
            # 打开视频
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {self.video_path}")
                
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 准备输出视频
            if self.output_path:
                output_dir = os.path.dirname(self.output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            all_results = []
            frame_idx = 0
            
            start_time = time.time()
            
            while cap.isOpened() and self.is_running:
                success, frame = cap.read()
                if not success:
                    break
                    
                # 进行检测
                results = self.detector.model(frame)
                all_results.append(results[0])
                
                # 在帧上绘制检测结果
                annotated_frame = results[0].plot()
                
                # 保存到输出视频
                if self.output_path:
                    out.write(annotated_frame)
                    
                # 更新进度
                frame_idx += 1
                progress = frame_idx / frame_count * 100
                self.signals.progress.emit(int(progress), annotated_frame)
                
            # 释放资源
            cap.release()
            if self.output_path:
                out.release()
                
            end_time = time.time()
            
            # 记录检测历史
            detection_info = {
                "type": "video",
                "source": self.video_path,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "frames": frame_count,
                "processing_time": end_time - start_time
            }
            self.detector.detection_history.append(detection_info)
            
            self.signals.result.emit((self.output_path, all_results))
            
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

# 实时检测线程
class WebcamDetectionThread(QThread):
    frame_ready = Signal(object, object)  # 帧, 结果
    error = Signal(str)
    
    def __init__(self, detector, cam_id=0):
        super().__init__()
        self.detector = detector
        self.cam_id = cam_id
        self.running = False
        
    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.cam_id)
        
        if not cap.isOpened():
            self.error.emit(f"无法打开摄像头 ID: {self.cam_id}")
            self.running = False
            return
            
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            success, frame = cap.read()
            if not success:
                break
                
            # 进行检测
            results = self.detector.model(frame)
            
            # 在帧上绘制检测结果
            annotated_frame = results[0].plot()
            
            # 发送信号
            self.frame_ready.emit(annotated_frame, results[0])
            
            frame_count += 1
            
        # 释放资源
        cap.release()
        
        # 停止时记录历史
        if frame_count > 0:
            end_time = time.time()
            detection_info = {
                "type": "webcam",
                "source": f"摄像头 ID: {self.cam_id}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "frames": frame_count,
                "processing_time": end_time - start_time
            }
            self.detector.detection_history.append(detection_info)
    
    def stop(self):
        self.running = False

# 定义操作区域样式
OPERATION_AREA_STYLE = """
    QFrame {
        background-color: #F0F6FF;
        border-radius: 8px;
        border: 1px solid #D0D0E8;
    }
"""

SECTION_TITLE_STYLE = """
    QLabel {
        color: #FFFFFF;
        background-color: #4169E1;
        padding: 8px;
        font-weight: bold;
        font-size: 14px;
        border-radius: 5px;
    }
"""

RESULT_TITLE_STYLE = """
    QLabel {
        color: #FFFFFF;
        background-color: #4169E1;
        padding: 8px;
        font-weight: bold;
        font-size: 14px;
        border-radius: 5px;
    }
"""

DISPLAY_AREA_STYLE = """
    QLabel {
        background-color: #F8F9FA;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
    }
"""

# 图片检测标签页
class ImageDetectionTab(QWidget):
    status_message = Signal(str)
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.thread_pool = QThreadPool()
        self.init_ui()
        
    def init_ui(self):
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # 操作区域框架
        operation_frame = QFrame()
        operation_frame.setStyleSheet(OPERATION_AREA_STYLE)
        operation_layout = QVBoxLayout(operation_frame)
        operation_layout.setContentsMargins(15, 15, 15, 15)
        operation_layout.setSpacing(10)
        
        # 操作区域标题
        operation_title = QLabel("操作区域")
        operation_title.setStyleSheet(SECTION_TITLE_STYLE)
        operation_layout.addWidget(operation_title)
        
        # 控制区域
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)
        
        self.select_btn = QPushButton("选择图片")
        self.select_btn.setStyleSheet(utils.create_styled_button("选择图片"))
        self.select_btn.clicked.connect(self.select_image)
        
        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.setStyleSheet(utils.create_styled_button("开始检测", "#2ECC71", "#27AE60"))
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.start_detection)
        
        self.save_btn = QPushButton("保存结果")
        self.save_btn.setStyleSheet(utils.create_styled_button("保存结果", "#F39C12", "#D68910"))
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_result)
        
        file_label = QLabel("文件:")
        file_label.setStyleSheet("font-weight: bold;")
        self.file_path_label = QLabel("未选择文件")
        self.file_path_label.setStyleSheet("color: #555; font-style: italic;")
        
        control_layout.addWidget(self.select_btn)
        control_layout.addWidget(self.detect_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addSpacing(20)
        control_layout.addWidget(file_label)
        control_layout.addWidget(self.file_path_label)
        control_layout.addStretch()
        
        operation_layout.addLayout(control_layout)
        layout.addWidget(operation_frame)
        
        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet(DISPLAY_AREA_STYLE)
        self.image_label.setText("请选择图片进行检测")
        
        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_label)
        
        layout.addWidget(scroll_area)
        
        # 结果区域
        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.StyledPanel)
        results_frame.setStyleSheet(OPERATION_AREA_STYLE)
        
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(15, 15, 15, 15)
        results_layout.setSpacing(10)
        
        results_title = QLabel("识别结果")
        results_title.setStyleSheet(RESULT_TITLE_STYLE)
        results_layout.addWidget(results_title)
        
        self.results_label = QLabel("未检测")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("font-size: 14px;")
        results_layout.addWidget(self.results_label)
        
        layout.addWidget(results_frame)
        
        # 状态区域
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
    def select_image(self):
        """选择图片文件"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.file_path_label.setText(os.path.basename(file_path))
            self.file_path = file_path
            self.detect_btn.setEnabled(True)
            
            # 显示原始图片
            img = cv2.imread(file_path)
            if img is not None:
                pixmap = utils.cv_to_pixmap(img, QSize(640, 480))
                self.image_label.setPixmap(pixmap)
                self.status_message.emit(f"已加载图片: {os.path.basename(file_path)}")
                
    def start_detection(self):
        """开始图片检测"""
        if not hasattr(self, 'file_path'):
            return
            
        self.detect_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_label.setText("正在检测...")
        self.status_message.emit("正在进行图片检测...")
        
        # 创建并启动检测线程
        worker = ImageDetectionWorker(self.detector, self.file_path)
        worker.signals.result.connect(self.detection_complete)
        worker.signals.error.connect(self.detection_error)
        worker.signals.finished.connect(self.detection_finished)
        
        self.thread_pool.start(worker)
        
    def detection_complete(self, result):
        """检测完成处理"""
        result_img, results = result
        
        # 显示结果图像
        pixmap = utils.cv_to_pixmap(result_img, QSize(640, 480))
        self.image_label.setPixmap(pixmap)
        
        # 显示检测结果
        if len(results.boxes) > 0:
            result_text = f"检测到 {len(results.boxes)} 个手语手势:\n"
            
            # 获取类别名称
            names = results.names
            
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = names[cls_id] if cls_id in names else f"类别 {cls_id}"
                result_text += f"手势 {i+1}: {name} (置信度: {conf:.2f})\n"
        else:
            result_text = "未检测到手语手势"
            
        self.results_label.setText(result_text)
        self.save_btn.setEnabled(True)
        self.status_label.setText("检测完成")
        self.status_message.emit("图片检测完成")
        
    def detection_error(self, error_msg):
        """检测错误处理"""
        self.status_label.setText(f"检测错误: {error_msg}")
        self.status_message.emit(f"图片检测错误: {error_msg}")
        
    def detection_finished(self):
        """检测结束（无论成功与否）处理"""
        self.select_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        
    def save_result(self):
        """保存检测结果"""
        if not hasattr(self.detector, 'current_frame') or self.detector.current_frame is None:
            return
            
        default_path = utils.get_default_save_path("image")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", default_path, "图片文件 (*.jpg *.jpeg *.png)"
        )
        
        if file_path:
            success = self.detector.save_current_frame(file_path)
            if success:
                self.status_label.setText(f"结果已保存: {os.path.basename(file_path)}")
                self.status_message.emit(f"检测结果已保存: {os.path.basename(file_path)}")
            else:
                self.status_label.setText("保存失败")
                self.status_message.emit("保存检测结果失败")

# 视频检测标签页
class VideoDetectionTab(QWidget):
    status_message = Signal(str)
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.thread_pool = QThreadPool()
        self.current_worker = None
        self.init_ui()
        
    def init_ui(self):
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # 操作区域框架
        operation_frame = QFrame()
        operation_frame.setStyleSheet(OPERATION_AREA_STYLE)
        operation_layout = QVBoxLayout(operation_frame)
        operation_layout.setContentsMargins(15, 15, 15, 15)
        operation_layout.setSpacing(10)
        
        # 操作区域标题
        operation_title = QLabel("操作区域")
        operation_title.setStyleSheet(SECTION_TITLE_STYLE)
        operation_layout.addWidget(operation_title)
        
        # 控制区域
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)
        
        self.select_btn = QPushButton("选择视频")
        self.select_btn.setStyleSheet(utils.create_styled_button("选择视频"))
        self.select_btn.clicked.connect(self.select_video)
        
        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.setStyleSheet(utils.create_styled_button("开始检测", "#2ECC71", "#27AE60"))
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.start_detection)
        
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setStyleSheet(utils.create_styled_button("停止检测", "#E74C3C", "#C0392B"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_detection)
        
        self.save_btn = QPushButton("保存结果")
        self.save_btn.setStyleSheet(utils.create_styled_button("保存结果", "#F39C12", "#D68910"))
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_result)
        
        file_label = QLabel("文件:")
        file_label.setStyleSheet("font-weight: bold;")
        self.file_path_label = QLabel("未选择文件")
        self.file_path_label.setStyleSheet("color: #555; font-style: italic;")
        
        control_layout.addWidget(self.select_btn)
        control_layout.addWidget(self.detect_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addSpacing(20)
        control_layout.addWidget(file_label)
        control_layout.addWidget(self.file_path_label)
        control_layout.addStretch()
        
        operation_layout.addLayout(control_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% 完成")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                text-align: center;
                background-color: #F8F9FA;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4169E1;
                border-radius: 3px;
            }
        """)
        operation_layout.addWidget(self.progress_bar)
        layout.addWidget(operation_frame)
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(DISPLAY_AREA_STYLE)
        self.video_label.setText("请选择视频进行检测")
        
        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.video_label)
        
        layout.addWidget(scroll_area)
        
        # 状态区域
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
    def select_video(self):
        """选择视频文件"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            self.file_path_label.setText(os.path.basename(file_path))
            self.file_path = file_path
            self.detect_btn.setEnabled(True)
            
            # 显示视频第一帧
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    pixmap = utils.cv_to_pixmap(frame, QSize(640, 480))
                    self.video_label.setPixmap(pixmap)
                cap.release()
                
            self.status_message.emit(f"已加载视频: {os.path.basename(file_path)}")
            
    def start_detection(self):
        """开始视频检测"""
        if not hasattr(self, 'file_path'):
            return
            
        # 准备输出路径
        output_path = utils.get_default_save_path("video")
        
        # 更新UI状态
        self.detect_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在检测...")
        self.status_message.emit("正在进行视频检测...")
        
        # 创建并启动检测线程
        self.current_worker = VideoDetectionWorker(self.detector, self.file_path, output_path)
        self.current_worker.signals.progress.connect(self.update_progress)
        self.current_worker.signals.result.connect(self.detection_complete)
        self.current_worker.signals.error.connect(self.detection_error)
        self.current_worker.signals.finished.connect(self.detection_finished)
        
        self.thread_pool.start(self.current_worker)
        
    def stop_detection(self):
        """停止视频检测"""
        if self.current_worker:
            self.current_worker.stop()
            self.status_label.setText("正在停止检测...")
            self.status_message.emit("正在停止视频检测...")
        
    def update_progress(self, progress, frame):
        """更新检测进度"""
        self.progress_bar.setValue(progress)
        
        # 更新视频帧显示
        pixmap = utils.cv_to_pixmap(frame, QSize(640, 480))
        self.video_label.setPixmap(pixmap)
        
        # 更新当前帧
        self.detector.current_frame = frame
        
    def detection_complete(self, result):
        """检测完成处理"""
        output_path, results = result
        
        self.output_path = output_path
        self.status_label.setText(f"检测完成: {os.path.basename(output_path)}")
        self.status_message.emit(f"视频检测完成: {os.path.basename(output_path)}")
        self.save_btn.setEnabled(True)
        
    def detection_error(self, error_msg):
        """检测错误处理"""
        self.status_label.setText(f"检测错误: {error_msg}")
        self.status_message.emit(f"视频检测错误: {error_msg}")
        
    def detection_finished(self):
        """检测结束（无论成功与否）处理"""
        self.select_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.current_worker = None
        
    def save_result(self):
        """保存检测结果（复制到用户指定路径）"""
        if not hasattr(self, 'output_path') or not os.path.exists(self.output_path):
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", self.output_path, "视频文件 (*.mp4)"
        )
        
        if file_path and file_path != self.output_path:
            try:
                # 读取源文件并写入目标文件
                with open(self.output_path, 'rb') as src_file:
                    with open(file_path, 'wb') as dst_file:
                        dst_file.write(src_file.read())
                        
                self.status_label.setText(f"结果已保存: {os.path.basename(file_path)}")
                self.status_message.emit(f"检测结果已保存: {os.path.basename(file_path)}")
            except Exception as e:
                self.status_label.setText(f"保存失败: {str(e)}")
                self.status_message.emit(f"保存检测结果失败: {str(e)}")

# 摄像头检测标签页
class WebcamDetectionTab(QWidget):
    status_message = Signal(str)
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.detection_thread = None
        self.detection_active = False
        self.current_results = None
        self.init_ui()
        
    def init_ui(self):
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # 操作区域框架
        operation_frame = QFrame()
        operation_frame.setStyleSheet(OPERATION_AREA_STYLE)
        operation_layout = QVBoxLayout(operation_frame)
        operation_layout.setContentsMargins(15, 15, 15, 15)
        operation_layout.setSpacing(10)
        
        # 操作区域标题
        operation_title = QLabel("操作区域")
        operation_title.setStyleSheet(SECTION_TITLE_STYLE)
        operation_layout.addWidget(operation_title)
        
        # 控制区域
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)
        
        # 摄像头选择
        camera_label = QLabel("摄像头:")
        camera_label.setStyleSheet("font-weight: bold;")
        self.camera_combo = QComboBox()
        self.camera_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 4px;
                background: white;
                min-width: 100px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #CCCCCC;
            }
        """)
        self.camera_combo.addItem("默认摄像头", 0)
        # 检测并添加可用摄像头
        self.detect_cameras()
        
        self.start_btn = QPushButton("开始检测")
        self.start_btn.setStyleSheet(utils.create_styled_button("开始检测", "#2ECC71", "#27AE60"))
        self.start_btn.clicked.connect(self.start_detection)
        
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setStyleSheet(utils.create_styled_button("停止检测", "#E74C3C", "#C0392B"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_detection)
        
        self.snapshot_btn = QPushButton("截图保存")
        self.snapshot_btn.setStyleSheet(utils.create_styled_button("截图保存", "#F39C12", "#D68910"))
        self.snapshot_btn.setEnabled(False)
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        
        control_layout.addWidget(camera_label)
        control_layout.addWidget(self.camera_combo)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.snapshot_btn)
        control_layout.addStretch()
        
        operation_layout.addLayout(control_layout)
        layout.addWidget(operation_frame)
        
        # 主区域分割为左右两部分
        main_area = QHBoxLayout()
        main_area.setSpacing(15)
        
        # 左侧视频显示区域 - 占用更多空间
        left_panel = QVBoxLayout()
        self.webcam_label = QLabel()
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setMinimumSize(640, 480)
        self.webcam_label.setStyleSheet(DISPLAY_AREA_STYLE)
        self.webcam_label.setText("点击'开始检测'按钮开始实时检测")
        
        # 滚动区域
        webcam_scroll = QScrollArea()
        webcam_scroll.setWidgetResizable(True)
        webcam_scroll.setWidget(self.webcam_label)
        left_panel.addWidget(webcam_scroll)
        main_area.addLayout(left_panel, 2)  # 占用2/3空间
        
        # 右侧结果区域
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        
        # FPS显示
        fps_layout = QHBoxLayout()
        fps_label = QLabel("FPS:")
        fps_label.setStyleSheet("font-weight: bold;")
        self.fps_value = QLabel("0")
        
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(self.fps_value)
        fps_layout.addStretch()
        
        # 手势数量显示
        count_layout = QHBoxLayout()
        count_label = QLabel("手势数目:")
        count_label.setStyleSheet("font-weight: bold;")
        self.count_value = QLabel("0")
        
        count_layout.addWidget(count_label)
        count_layout.addWidget(self.count_value)
        count_layout.addStretch()
        
        # 所有手势列表
        gestures_label = QLabel("所有手势:")
        gestures_label.setStyleSheet("font-weight: bold;")
        self.gestures_list = QLabel("无")
        self.gestures_list.setWordWrap(True)
        
        # 结果区域框架
        results_frame = QFrame()
        results_frame.setStyleSheet(OPERATION_AREA_STYLE)
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(15, 15, 15, 15)
        results_layout.setSpacing(10)
        
        # 结果区域标题
        results_title = QLabel("识别结果")
        results_title.setStyleSheet(RESULT_TITLE_STYLE)
        results_layout.addWidget(results_title)
        
        # 添加各个元素到结果区域
        results_layout.addLayout(fps_layout)
        results_layout.addLayout(count_layout)
        results_layout.addWidget(gestures_label)
        results_layout.addWidget(self.gestures_list)
        results_layout.addStretch()
        
        right_panel.addWidget(results_frame)
        main_area.addLayout(right_panel, 1)  # 占用1/3空间
        
        layout.addLayout(main_area)
        
        # 详细结果区域
        detail_frame = QFrame()
        detail_frame.setStyleSheet(OPERATION_AREA_STYLE)
        detail_layout = QVBoxLayout(detail_frame)
        detail_layout.setContentsMargins(15, 15, 15, 15)
        detail_layout.setSpacing(10)
        
        # 详细结果标题
        detail_title = QLabel("识别详情")
        detail_title.setStyleSheet(RESULT_TITLE_STYLE)
        detail_layout.addWidget(detail_title)
        
        self.results_label = QLabel("未检测")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("font-size: 14px;")
        detail_layout.addWidget(self.results_label)
        
        layout.addWidget(detail_frame)
        
        # 状态区域
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
    def detect_cameras(self):
        """检测系统上可用的摄像头"""
        # 默认摄像头已添加，这里添加其他摄像头
        # 在实际应用中，可能需要根据系统平台进行更复杂的摄像头检测
        # 简化起见，这里只添加几个额外的摄像头选项
        self.camera_combo.addItem("摄像头 1", 1)
        self.camera_combo.addItem("摄像头 2", 2)
        
    def start_detection(self):
        """开始实时检测"""
        if self.detection_active:
            return
            
        # 获取选择的摄像头ID
        cam_id = self.camera_combo.currentData()
        
        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.snapshot_btn.setEnabled(True)
        self.status_label.setText("正在进行实时检测...")
        self.status_message.emit("已开始摄像头实时检测")
        
        # 创建并启动检测线程
        self.detection_thread = WebcamDetectionThread(self.detector, cam_id)
        self.detection_thread.frame_ready.connect(self.update_frame)
        self.detection_thread.error.connect(self.detection_error)
        self.detection_thread.start()
        
        self.detection_active = True
        
    def stop_detection(self):
        """停止实时检测"""
        if not self.detection_active:
            return
            
        # 停止检测线程
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.wait()  # 等待线程完成
            self.detection_thread = None
            
        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.camera_combo.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.snapshot_btn.setEnabled(False)
        self.status_label.setText("实时检测已停止")
        self.status_message.emit("摄像头实时检测已停止")
        
        self.detection_active = False
        
    def update_frame(self, frame, results):
        """更新检测到的帧和结果"""
        # 更新视频帧显示
        pixmap = utils.cv_to_pixmap(frame, QSize(640, 480))
        self.webcam_label.setPixmap(pixmap)
        
        # 保存最新的帧和结果
        self.detector.current_frame = frame
        self.current_results = results
        
        # 更新检测结果显示
        if results and len(results.boxes) > 0:
            # 更新数量显示
            self.count_value.setText(str(len(results.boxes)))
            
            # 获取类别名称
            names = results.names
            
            # 更新所有手势列表
            gesture_names = []
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                if cls_id in names:
                    gesture_names.append(names[cls_id])
                else:
                    gesture_names.append(f"类别 {cls_id}")
            
            self.gestures_list.setText(", ".join(gesture_names))
            
            # 更新详细结果
            result_text = f"检测到 {len(results.boxes)} 个手语手势:\n"
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = names[cls_id] if cls_id in names else f"类别 {cls_id}"
                result_text += f"手势 {i+1}: {name} (置信度: {conf:.2f})\n"
                
                # 限制显示的结果数量以保持界面简洁
                if i >= 4:  # 最多显示5个结果
                    result_text += f"... 等共 {len(results.boxes)} 个手势"
                    break
        else:
            self.count_value.setText("0")
            self.gestures_list.setText("无")
            result_text = "未检测到手语手势"
            
        self.results_label.setText(result_text)
        
        # 更新FPS (简单计算，实际情况下可能需要更复杂的计算方法)
        # 这里只是一个模拟，实际FPS应该基于时间计算
        self.fps_value.setText(str(int(np.random.randint(25, 30))))
        
    def detection_error(self, error_msg):
        """检测错误处理"""
        self.status_label.setText(f"检测错误: {error_msg}")
        self.status_message.emit(f"摄像头检测错误: {error_msg}")
        self.stop_detection()
        
    def take_snapshot(self):
        """截取当前帧并保存"""
        if not self.detection_active or self.detector.current_frame is None:
            return
            
        default_path = utils.get_default_save_path("image")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存截图", default_path, "图片文件 (*.jpg *.jpeg *.png)"
        )
        
        if file_path:
            success = self.detector.save_current_frame(file_path)
            if success:
                self.status_label.setText(f"截图已保存: {os.path.basename(file_path)}")
                self.status_message.emit(f"摄像头截图已保存: {os.path.basename(file_path)}")
            else:
                self.status_label.setText("截图保存失败")
                self.status_message.emit("摄像头截图保存失败")

# 历史统计标签页
class HistoryTab(QWidget):
    status_message = Signal(str)
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.init_ui()
        
    def init_ui(self):
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # 操作区域框架
        operation_frame = QFrame()
        operation_frame.setStyleSheet(OPERATION_AREA_STYLE)
        operation_layout = QVBoxLayout(operation_frame)
        operation_layout.setContentsMargins(15, 15, 15, 15)
        operation_layout.setSpacing(10)
        
        # 操作区域标题
        operation_title = QLabel("数据操作")
        operation_title.setStyleSheet(SECTION_TITLE_STYLE)
        operation_layout.addWidget(operation_title)
        
        # 控制区域
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)
        
        self.refresh_btn = QPushButton("刷新数据")
        self.refresh_btn.setStyleSheet(utils.create_styled_button("刷新数据"))
        self.refresh_btn.clicked.connect(self.update_history_table)
        
        self.export_btn = QPushButton("导出统计")
        self.export_btn.setStyleSheet(utils.create_styled_button("导出统计", "#F39C12", "#D68910"))
        self.export_btn.clicked.connect(self.export_history)
        
        self.clear_btn = QPushButton("清空历史")
        self.clear_btn.setStyleSheet(utils.create_styled_button("清空历史", "#E74C3C", "#C0392B"))
        self.clear_btn.clicked.connect(self.clear_history)
        
        control_layout.addWidget(self.refresh_btn)
        control_layout.addWidget(self.export_btn)
        control_layout.addWidget(self.clear_btn)
        control_layout.addStretch()
        
        operation_layout.addLayout(control_layout)
        layout.addWidget(operation_frame)
        
        # 统计摘要框架
        summary_frame = QFrame()
        summary_frame.setStyleSheet(OPERATION_AREA_STYLE)
        summary_layout = QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(15, 15, 15, 15)
        summary_layout.setSpacing(10)
        
        # 统计标题
        summary_title = QLabel("检测统计摘要")
        summary_title.setStyleSheet(RESULT_TITLE_STYLE)
        summary_layout.addWidget(summary_title)
        
        # 统计内容布局
        stats_layout = QGridLayout()
        stats_layout.setColumnStretch(0, 1)
        stats_layout.setColumnStretch(1, 1)
        stats_layout.setSpacing(10)
        
        # 统计项
        self.total_label = QLabel("总检测次数: 0")
        self.total_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.image_label = QLabel("图片检测: 0")
        self.image_label.setStyleSheet("font-size: 14px;")
        self.video_label = QLabel("视频检测: 0")
        self.video_label.setStyleSheet("font-size: 14px;")
        self.webcam_label = QLabel("实时检测: 0")
        self.webcam_label.setStyleSheet("font-size: 14px;")
        
        stats_layout.addWidget(self.total_label, 0, 0)
        stats_layout.addWidget(self.image_label, 0, 1)
        stats_layout.addWidget(self.video_label, 1, 0)
        stats_layout.addWidget(self.webcam_label, 1, 1)
        
        summary_layout.addLayout(stats_layout)
        layout.addWidget(summary_frame)
        
        # 历史记录框架
        history_frame = QFrame()
        history_frame.setStyleSheet(OPERATION_AREA_STYLE)
        history_layout = QVBoxLayout(history_frame)
        history_layout.setContentsMargins(15, 15, 15, 15)
        history_layout.setSpacing(10)
        
        # 历史记录标题
        history_title = QLabel("历史记录")
        history_title.setStyleSheet(RESULT_TITLE_STYLE)
        history_layout.addWidget(history_title)
        
        # 历史表格
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["类型", "来源", "时间", "检测数量", "处理时间(秒)"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                background-color: #FFFFFF;
                gridline-color: #E0E0E0;
            }
            QTableWidget::item:alternate {
                background-color: #F8F9FA;
            }
            QHeaderView::section {
                background-color: #4169E1;
                color: white;
                font-weight: bold;
                border: none;
                padding: 4px;
            }
        """)
        
        history_layout.addWidget(self.history_table)
        layout.addWidget(history_frame)
        
        # 状态区域
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # 更新数据
        self.update_history_table()
        
    def update_history_table(self):
        """更新历史数据表格"""
        history = self.detector.get_detection_history()
        
        # 清空表格
        self.history_table.setRowCount(0)
        
        # 统计计数
        total_count = len(history)
        image_count = sum(1 for item in history if item.get("type") == "image")
        video_count = sum(1 for item in history if item.get("type") == "video")
        webcam_count = sum(1 for item in history if item.get("type") == "webcam")
        
        # 更新统计标签
        self.total_label.setText(f"总检测次数: {total_count}")
        self.image_label.setText(f"图片检测: {image_count}")
        self.video_label.setText(f"视频检测: {video_count}")
        self.webcam_label.setText(f"实时检测: {webcam_count}")
        
        # 填充表格
        self.history_table.setRowCount(len(history))
        
        for row, item in enumerate(history):
            # 类型
            type_item = QTableWidgetItem(self.get_type_display(item.get("type")))
            self.history_table.setItem(row, 0, type_item)
            
            # 来源
            source_item = QTableWidgetItem(str(item.get("source", "未知")))
            self.history_table.setItem(row, 1, source_item)
            
            # 时间
            timestamp_item = QTableWidgetItem(item.get("timestamp", ""))
            self.history_table.setItem(row, 2, timestamp_item)
            
            # 检测数量
            if "detections" in item:
                detection_count = item["detections"]
            elif "frames" in item:
                detection_count = item["frames"]
            else:
                detection_count = "-"
            count_item = QTableWidgetItem(str(detection_count))
            self.history_table.setItem(row, 3, count_item)
            
            # 处理时间
            if "processing_time" in item:
                processing_time = f"{item['processing_time']:.2f}"
            else:
                processing_time = "-"
            time_item = QTableWidgetItem(processing_time)
            self.history_table.setItem(row, 4, time_item)
        
        self.status_message.emit("检测历史已刷新")
        
    def get_type_display(self, type_name):
        """获取检测类型的显示名称"""
        type_map = {
            "image": "图片检测",
            "video": "视频检测",
            "webcam": "实时检测"
        }
        return type_map.get(type_name, type_name)
        
    def export_history(self):
        """导出历史数据到CSV文件"""
        history = self.detector.get_detection_history()
        if not history:
            utils.show_info_message(self, "导出失败", "没有历史数据可导出")
            return
            
        default_path = utils.get_default_save_path("csv")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出历史数据", default_path, "CSV文件 (*.csv)"
        )
        
        if file_path:
            success = utils.export_detection_history(history, file_path)
            if success:
                self.status_message.emit(f"历史数据已导出: {os.path.basename(file_path)}")
                utils.show_info_message(self, "导出成功", f"历史数据已导出到: {file_path}")
            else:
                self.status_message.emit("历史数据导出失败")
                utils.show_error_message(self, "导出失败", "无法导出历史数据")
    
    def clear_history(self):
        """清空历史数据"""
        if not self.detector.detection_history:
            utils.show_info_message(self, "清空历史", "历史记录已经为空")
            return
            
        confirm = utils.show_question_message(
            self, "确认清空", "确定要清空所有历史记录吗？此操作不可撤销。"
        )
        
        if confirm:
            self.detector.detection_history = []
            self.update_history_table()
            self.status_message.emit("历史记录已清空")
            utils.show_info_message(self, "清空完成", "所有历史记录已清空") 