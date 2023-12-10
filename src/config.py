class Config:
    device = "cpu"
    model_url = "https://drive.google.com/drive/folders/1YQsGFk7teZ7V0hmLmOraljkvoMpX-9AD"
    ocr_path = "storage/vlsp_transfomer_vietocr.pth"
    model_path = "storage/vit_base_vit5_base_v2_1.3197_0.4732_3.5212.pt"
    question_maxlen = 32
    vietocr_threshold = 0.8
    answer_maxlen = 56
    ocr_maxlen_v2 = 200
    num_ocr = 32
    num_beams = 3