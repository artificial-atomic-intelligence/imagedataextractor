import re
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFilter

from .textdetection import TextDetector

"""
This file will contain all code required to perform OCR.

"""


class OCR:

    def __init__(self):
        self.text_detector = TextDetector()
        self.valid_text_expression = r'[1-9]\d*\s*(μ|u|n)m'

    def perform_ocr(self, image: np.ndarray, lang: str='eng', custom_config: str=None) -> str:
        custom_config = '--psm 6 -c tessedit_char_whitelist=0123456789\sunm'
        image = self.preprocess_image(image)
        text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
        match = re.search(self.valid_text_expression, text)
        if match:
            text = match[0]
        else:
            text = None
        return text

    def get_scalebar_text(self, image: np.ndarray):
        valid_match = False
        rois = self.text_detector.get_text_rois(image)
        for roi in rois:
            text = self.perform_ocr(roi)
            if text is not None:
                valid_match = True
            if valid_match:
                break
        return text

    def preprocess_image(self, image):
        h, w = image.shape[:2]
        image = Image.fromarray(image).resize((int(w*2.5), int(h*2.5)), resample=Image.BICUBIC)
        image = np.array(image)
        return image
