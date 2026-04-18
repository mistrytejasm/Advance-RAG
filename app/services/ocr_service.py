from paddleocr import PaddleOCR
import paddle
from app.utils.logger import logger

use_gpu = paddle.device.is_compiled_with_cuda()
logger.info(f"PaddleOCR GPU Hardware Found: {use_gpu}")

# Disable mkldnn (OneDNN) in CPU mode to prevent the ConvertPirAttribute2RuntimeAttribute C++ error
ocr = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False)


def extract_text_from_image(image_path):

    logger.info("Running OCR")

    result = ocr.ocr(image_path)

    text_list = []

    for line in result[0]:

        text = line[1][0]

        text_list.append(text)

    return " ".join(text_list)