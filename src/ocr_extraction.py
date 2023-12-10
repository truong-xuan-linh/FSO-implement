from paddleocr import PaddleOCR
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import requests
import numpy as np
from PIL import Image, ImageTransform

from src.config import Config

class OCRDetector:
  def __init__(self) -> None:
    self.paddle_ocr = PaddleOCR(lang='en', 
                                use_angle_cls=False, 
                                use_gpu=True if Config.device == "cpu" else False, 
                                show_log=False )
    # config['weights'] = './weights/transformerocr.pth'
    
    vietocr_config = Cfg.load_config_from_name('vgg_transformer')
    vietocr_config['weights'] = Config.ocr_path
    vietocr_config['cnn']['pretrained']=False
    vietocr_config['device'] = Config.device
    vietocr_config['predictor']['beamsearch']=False
    self.viet_ocr = Predictor(vietocr_config)

  def find_box(self, image):
    '''Xác định box dựa vào mô hình paddle_ocr'''
    result = self.paddle_ocr.ocr(image, cls = False, rec=False)
    result = result[0]
    # Extracting detected components
    boxes = result #[res[0] for res in result]
    boxes = np.array(boxes).astype(int)

    # scores = [res[1][1] for res in result]
    return boxes

  def cut_image_polygon(self, image, box):
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
    w = x2 - x1
    h = y4 - y1
    scl = h//7
    new_box = [max(x1-scl,0), max(y1 - scl, 0)], [x2+scl, y2-scl], [x3+scl, y3+scl], [x4-scl, y4+scl]
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = new_box
    # Define 8-tuple with x,y coordinates of top-left, bottom-left, bottom-right and top-right corners and apply
    transform = [x1, y1, x4, y4, x3, y3, x2, y2]
    result = image.transform((w,h), ImageTransform.QuadTransform(transform))
    return result

  def vietnamese_text(self, boxes, image):
    '''Xác định text dựa vào mô hình viet_ocr'''
    results = []
    for box in boxes:
      try:
        cut_image = self.cut_image_polygon(image, box)
        # cut_image = Image.fromarray(np.uint8(cut_image))
        text, score = self.viet_ocr.predict(cut_image, return_prob=True)
        if score > Config.vietocr_threshold:
          results.append({"text": text,
                        "score": score,
                        "box": box})
      except:
        continue
    return results

  #Merge
  def text_detector(self, image_path, is_local=False):
    if is_local:
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    # np_image = np.array(image)

    boxes = self.find_box(image_path)
    if not boxes.any():
        return None

    results = self.vietnamese_text(boxes, image)
    if results != []:
        return results
    else:
        return None