#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于YOLO11的手语识别系统
主程序入口文件
"""

import os
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from gui import MainWindow

def setup_font_support():
    """设置字体支持，确保在所有系统上中文正常显示"""
    # 添加本地字体（如果需要）
    font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    if os.path.exists(font_dir):
        for font_file in os.listdir(font_dir):
            if font_file.endswith(('.ttf', '.otf')):
                font_path = os.path.join(font_dir, font_file)
                QFontDatabase.addApplicationFont(font_path)

def main():
    """主函数"""
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 设置字体支持
    setup_font_support()
    
    # 设置应用程序名称和组织
    app.setApplicationName("基于yolov11智能手势识别系统")
    app.setOrganizationName("YOLO11手语识别")
    
    # 设置全局样式表
    app.setStyleSheet("""
        QWidget {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 
                         "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
            font-size: 13px;
            color: #333333;
        }
        QMainWindow, QDialog {
            background-color: #FFFFFF;
        }
        QPushButton {
            border-radius: 4px;
            padding: 6px 12px;
            background-color: #4169E1;
            color: white;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #3151B5;
        }
        QPushButton:pressed {
            background-color: #2C49A0;
        }
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            border: 1px solid #CCCCCC;
            border-radius: 4px;
            padding: 4px;
            background: white;
        }
        QToolTip {
            border: 1px solid #CCCCCC;
            background-color: #F8F9FA;
            color: #333333;
            padding: 2px;
        }
        QLabel {
            color: #333333;
        }
        QTabBar::tab {
            background: #F0F0F0;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        }
        QTabBar::tab:selected {
            background: #4169E1;
            color: white;
        }
        QTabWidget::pane {
            border: 1px solid #CCCCCC;
            background: #FFFFFF;
            border-radius: 4px;
        }
        QProgressBar {
            border: 1px solid #E0E0E0;
            border-radius: 4px;
            text-align: center;
            background-color: #F8F9FA;
        }
        QProgressBar::chunk {
            background-color: #4169E1;
            border-radius: 3px;
        }
        QStatusBar {
            background: #F0F0F0;
            color: #333333;
        }
    """)
    
    # 创建并显示主窗口
    main_window = MainWindow()
    main_window.show()
    
    # 运行应用程序
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 