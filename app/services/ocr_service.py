from paddleocr import PaddleOCR
import paddle
from app.utils.logger import logger

use_gpu = paddle.device.is_compiled_with_cuda()
logger.info(f"PaddleOCR GPU Hardware Found: {use_gpu}")

# Disable mkldnn (OneDNN) in CPU mode to prevent the ConvertPirAttribute2RuntimeAttribute C++ error
ocr = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False)


def extract_text_from_image(image_path: str) -> str:
    """Extract text from image path with safety bounds and null checks."""
    logger.info(f"Running OCR on image: {image_path}")

    try:
        result = ocr.ocr(image_path)
        if not result or not result[0]:
            return ""

        text_list = []
        for line in result[0]:
            if line and len(line) > 1 and line[1]:
                text = line[1][0]
                if text:
                    text_list.append(str(text).strip())

        return " ".join(text_list)
    except Exception as exc:
        logger.warning(f"OCR extraction failed on {image_path}: {exc}")
        return ""