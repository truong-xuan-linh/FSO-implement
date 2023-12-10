import re
import torch
import requests
from PIL import Image, ImageFont, ImageDraw, ImageTransform
from transformers import AutoImageProcessor, ViTModel, AutoTokenizer, T5EncoderModel

from src.config import Config
from src.ocr_extraction import OCRDetector

class ViT:
    def __init__(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model.to(Config.device)
        
    def extraction(self, image_url, is_local=True):
        if not is_local:
            images = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        else:
            images = Image.open(image_url).convert("RGB")
            
        inputs = self.processor(images, return_tensors="pt").to(Config.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        attention_mask = torch.ones((last_hidden_states.shape[0], last_hidden_states.shape[1]))

        return last_hidden_states.to(Config.device), attention_mask.to(Config.device)
    
    def pooling_extraction(self, image):
        image_inputs = self.processor(image, return_tensors="pt").to(Config.device)

        with torch.no_grad():
            image_outputs = self.model(**image_inputs)
            image_pooler_output = image_outputs.pooler_output
            image_pooler_output = torch.unsqueeze(image_pooler_output, 0)
            image_attention_mask = torch.ones((image_pooler_output.shape[0], image_pooler_output.shape[1]))

        return image_pooler_output.to(Config.device), image_attention_mask.to(Config.device)
    
class ViT5:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
        self.model = T5EncoderModel.from_pretrained("VietAI/vit5-base")
        self.model.to(Config.device)

    def extraction(self, content, max_length, padding, normalize=True):
        # padding = "max_length"
        if normalize:
            content = ViT5.question_nomalization(content)
        
        print(content)
        inputs = self.tokenizer(content,
                            padding=padding,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt").to(Config.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        
        return last_hidden_states.to(Config.device), inputs.attention_mask.to(Config.device)

    @staticmethod
    def question_nomalization(question):
        return question.lower()
        return re.sub(r'[^a-zA-Z0-9À-Ỹà-ỹ ]', '', question).lower()

class OCR:
    def __init__(self) -> None:
        self.ocr_detector = OCRDetector()
        self.vit = ViT()
        self.vit5 = ViT5()
        self.ocr_fixer = {
                            " 0gmail" : "@gmail",
                            "0gmail" : "@gmail",
                            "wwww," : "www.",
                            " qgmail" : "@gmail",
                            "qgmail" : "@gmail",
                            " agmail" : "@gmail",
                            "agmail" : "@gmail",
                            " ggmail" : "@gmail",
                            "ggmail" : "@gmail",
                            "1 gmail" : "@gmail",
                            "emaii" : "email",
                            " comail" : "@gmail",
                            "comail" : "@gmail",
                            " agmail" : "@gmail",
                            "agmail" : "@gmail",
                            " gmail" : "@gmail",
                            ",com" : ".com",
                            " ogmail" : "@gmail",
                            "ogmail" : "@gmail",
                            " %": "%",
                            " @": "@",
                            "% ": "%",
                            "@ ": "@",
                            ".": " . ",
                            "!": " ! ",
                            '"': ' " ',
                            "&": " & ",
                            ",": " , ",
                            "-": " - ",
                            ":": " : "
                        }

    def extraction(self, image_dir):
        image = Image.open(image_dir).convert("RGB")
        
        ocr_results = self.ocr_detector.text_detector(image_dir, is_local=True)
        if not ocr_results:
            print("NOT OCR1")
            embedding = torch.zeros(1, Config.ocr_maxlen_v2, 768).to(Config.device)
            mask = torch.zeros(embedding.shape[0], embedding.shape[1]).to(Config.device)
            
            return embedding, mask

        ocrs = self.post_process(ocr_results)
        if not ocrs:
            print("NOT OCR2")
            embedding = torch.zeros(1, Config.ocr_maxlen_v2, 768).to(Config.device)
            mask = torch.zeros(embedding.shape[0], embedding.shape[1]).to(Config.device)
            
            return embedding, mask
        
        embedding = []
        ocrs.reverse()
        print(ocrs)

        for idx, ocr in enumerate(ocrs[:Config.num_ocr]):
            text = ocr["text"]
            text = f"{idx}: {text} ảnh: "
            
            box = ocr["box"]
            content_last, content_mask = self.vit5.extraction(content=text, max_length=Config.ocr_maxlen_v2, padding=False, normalize=False)
            embedding.append(content_last)
            
            cut_image = OCR.cut_image_polygon(image, box)
            image_pooler_output, image_attention_mask = self.vit.pooling_extraction(cut_image)
            embedding.append(image_pooler_output)
            print(image_pooler_output.shape)
        embedding = torch.cat(embedding, dim=1)
        embedding = embedding[:,:Config.ocr_maxlen_v2, :]
        mask = torch.ones(embedding.shape[0], embedding.shape[1])
        print(embedding.shape)
        embedding = torch.concat([embedding, torch.zeros(embedding.shape[0], Config.ocr_maxlen_v2 - embedding.shape[1], embedding.shape[2])], dim=1).to(Config.device)
        mask = torch.concat([mask, torch.zeros(mask.shape[0], Config.ocr_maxlen_v2 - mask.shape[1])], dim=1).to(Config.device)
        
        print(embedding.shape)
        return embedding.to(Config.device), mask.to(Config.device)

    def post_process(self,ocr_results):
        ocrs = []
        for result in ocr_results:
            text = result["text"]
            if len(text) <=3:
                continue
            if len(set(text.replace(" ", ""))) <=2:
                continue
            box = result["box"]

            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
            w = x2 - x1
            h = y4 - y1
            if h > w:
                continue
            if w*h <= 1000:
                continue
  
            for k,v in self.ocr_fixer.items():
                text = text.replace(k, v)
            text = " ".join(text.split(" "))
            ocrs.append(
                {"text": text.lower(),
                "box": box}
            )
        return ocrs

    @staticmethod
    def cut_image_polygon(image, box):
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