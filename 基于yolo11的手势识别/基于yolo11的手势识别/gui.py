import os
import sys
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QSize, Signal, Slot, QThread, QTimer
from PySide6.QtGui import QImage, QPixmap, QIcon, QFont, QAction, QGuiApplication, QLinearGradient, QColor, QPalette, QBrush
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, 
    QTabWidget, QScrollArea, QProgressBar, QComboBox,
    QSplitter, QFrame, QTableWidget, QTableWidgetItem, QHeaderView
)

import utils
from detector import HandSignDetector
from widgets import (
    ImageDetectionTab, VideoDetectionTab, 
    WebcamDetectionTab, HistoryTab
)

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化检测器
        self.detector = HandSignDetector()
        
        # 设置窗口属性
        self.setWindowTitle("基于yolov11智能手势识别系统 v1.0")
        self.setMinimumSize(1024, 768)
        
        # 设置应用程序字体
        app_font = utils.ensure_font_support()
        QApplication.setFont(app_font)
        
        # 初始化UI
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        # 创建中央小部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建标题栏（蓝色渐变背景）
        title_bar = QWidget()
        # 设置固定高度
        title_bar.setFixedHeight(90)
        # 创建蓝色渐变背景
        gradient = QLinearGradient(0, 0, 0, title_bar.height())
        gradient.setColorAt(0, QColor("#4169E1"))  # 浅蓝色
        gradient.setColorAt(1, QColor("#1E3A8A"))  # 深蓝色
        
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(gradient))
        title_bar.setAutoFillBackground(True)
        title_bar.setPalette(palette)
        
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 10, 20, 10)
        
        # 标题文本
        title_label = QLabel("基于yolov11智能手势识别系统 v1.0")
        title_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: white;
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        main_layout.addWidget(title_bar)
        
        # 内容区域
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: white;")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(10)
        
        # 创建选项卡窗口
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #CCCCCC;
                background: #FFFFFF;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #F0F0F0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #4169E1;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #D0D0D0;
            }
        """)
        
        # 创建各个选项卡
        self.image_tab = ImageDetectionTab(self.detector)
        self.video_tab = VideoDetectionTab(self.detector)
        self.webcam_tab = WebcamDetectionTab(self.detector)
        self.history_tab = HistoryTab(self.detector)
        
        # 添加选项卡
        self.tabs.addTab(self.webcam_tab, "实时检测")
        self.tabs.addTab(self.image_tab, "图片检测")
        self.tabs.addTab(self.video_tab, "视频检测")
        self.tabs.addTab(self.history_tab, "检测统计")
        
        content_layout.addWidget(self.tabs)
        main_layout.addWidget(content_widget)
        
        # 底部状态栏
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background: #F0F0F0;
                color: #333333;
                border-top: 1px solid #CCCCCC;
            }
        """)
        self.statusBar().showMessage("就绪")
        
        # 连接信号
        self.image_tab.status_message.connect(self.update_status)
        self.video_tab.status_message.connect(self.update_status)
        self.webcam_tab.status_message.connect(self.update_status)
        self.history_tab.status_message.connect(self.update_status)
        
        # 更新历史记录选项卡
        self.tabs.currentChanged.connect(self.tab_changed)
    
    @Slot(str)
    def update_status(self, message):
        """更新状态栏消息"""
        self.statusBar().showMessage(message)
    
    @Slot(int)
    def tab_changed(self, index):
        """选项卡切换时的处理"""
        # 如果切换到历史记录选项卡，刷新历史数据
        if index == 3:  # 历史选项卡索引
            self.history_tab.update_history_table()
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 停止所有正在进行的检测
        if hasattr(self.webcam_tab, 'stop_detection'):
            self.webcam_tab.stop_detection()
        if hasattr(self.video_tab, 'stop_detection'):
            self.video_tab.stop_detection()
        
        # 让基类处理剩下的
        super().closeEvent(event) 