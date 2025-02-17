import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
                             QFileDialog, QWidget)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class ImageDescriptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # تهيئة النموذج
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        # إعداد واجهة المستخدم
        self.initUI()

    def initUI(self):
        # تعيين عنوان النافذة
        self.setWindowTitle('وصف الصور بالذكاء الاصطناعي')
        self.setGeometry(100, 100, 700, 600)

        # إنشاء الويدجت الرئيسي
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # إنشاء تخطيط رأسي
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # زر اختيار الصورة
        select_button = QPushButton('اختر صورة')
        select_button.setFont(QFont('Arial', 12))
        select_button.clicked.connect(self.select_image)
        main_layout.addWidget(select_button)

        # مساحة عرض الصورة
        self.image_label = QLabel('سيتم عرض الصورة هنا')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedHeight(300)
        main_layout.addWidget(self.image_label)

        # مساحة عرض الوصف
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setFont(QFont('Arial', 12))
        main_layout.addWidget(self.description_text)

    def select_image(self):
        # فتح مربع حوار اختيار الملف
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            'اختر صورة', 
            '', 
            'ملفات الصور (*.png *.jpg *.jpeg *.bmp *.gif)'
        )
        
        if file_name:
            # عرض الصورة
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            # وصف الصورة
            descriptions = self.describe_image(file_name)
            
            # عرض الأوصاف
            self.description_text.clear()
            for i, desc in enumerate(descriptions, 1):
                self.description_text.append(f"{i}. {desc}")

    def describe_image(self, image_path, num_descriptions=3):
        try:
            # تحميل الصورة
            image = Image.open(image_path)
            
            # معالجة الصورة
            inputs = self.processor(image, return_tensors="pt")

            # توليد الأوصاف
            out = self.model.generate(
                **inputs, 
                max_length=50,
                num_return_sequences=num_descriptions,
                num_beams=4,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7
            )

            # فك ترميز الأوصاف
            captions = [
                self.processor.decode(caption, skip_special_tokens=True) 
                for caption in out
            ]
            
            return captions

        except Exception as e:
            return [f"حدث خطأ: {str(e)}"]

def main():
    # إنشاء التطبيق
    app = QApplication(sys.argv)
    
    # إنشاء النافذة الرئيسية
    window = ImageDescriptionApp()
    
    # عرض النافذة
    window.show()
    
    # تشغيل التطبيق
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()