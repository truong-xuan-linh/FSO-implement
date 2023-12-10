import os
import gdown
import torch
from typing import *
from transformers import T5ForConditionalGeneration

from src.config import Config
from src.feature_extraction import ViT, ViT5, OCR

class Model:
    def __init__(self) -> None:
        if not os.path.isdir("storage"):
            print("DOWNLOADING model")
            gdown.download_folder(Config.model_url, output="storage")

        self.vit = ViT()
        self.vit5 = ViT5()
        self.ocr = OCR()
        self.model = torch.load(Config.model_path, map_location=torch.device(Config.device))
        self.model.to(Config.device)
        
    
    def get_inputs(self, image_dir: str, question: str):
        #VIT
        image_last, image_mask = self.vit.extraction(image_dir)

        # OCR
        # ocr_last = torch.zeros(1, Config.ocr_maxlen_v2, image_last.shape[-1]).to(Config.device)
        # ocr_mask = torch.zeros(ocr_last.shape[0], ocr_last.shape[1]).to(Config.device)
        ocr_last, ocr_mask = self.ocr.extraction(image_dir)
        
        #VIT5
        question_last, question_mask = self.vit5.extraction(question, max_length=Config.question_maxlen, padding="max_length")

        inputs_embeds = torch.cat((image_last, ocr_last, question_last), 1)
        mask = torch.cat((image_mask, ocr_mask, question_mask), 1)

        return {
            "inputs_embeds": inputs_embeds.to(Config.device),
            "attention_mask": mask.to(Config.device)
        }
        
    def inference(self, image_dir: str, question: str):
        inputs = self.get_inputs(image_dir, question)
        with torch.no_grad():
            inputs_embeds = inputs["inputs_embeds"]
            attention_mask = inputs["attention_mask"]
            
            generated_ids = self.model.generate(
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask,
                num_beams = Config.num_beams,
                max_length = Config.answer_maxlen
            )
            
            pred_answer = self.vit5.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return pred_answer
