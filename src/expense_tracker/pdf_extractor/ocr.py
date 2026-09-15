from pathlib import Path
import cv2
import numpy as np
import pypdfium2 as pdfium
from paddleocr import PaddleOCR


INPUT_PDF = Path("data/receipts/image3.pdf")
MAX_IMAGE_SIZE = 4000
PDF_RENDER_SCALE = 4
DEFAULT_CONFIDENCE_THRESHOLD = 0.9


def render_pdf_page(
    pdf: pdfium.PdfDocument,
    page_index: int,
) -> cv2.typing.MatLike:
    """Render one PDF page as a BGR OpenCV image."""

    page = pdf[page_index]
    bitmap = page.render(scale=PDF_RENDER_SCALE)
    rgb_image = np.asarray(bitmap.to_pil())
    page.close()

    return cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR,
    )


def preprocess_image(
    image: cv2.typing.MatLike,
    max_size: int = MAX_IMAGE_SIZE,
) -> cv2.typing.MatLike:
    """Reduce the image so its longest side is at most max_size."""

    height, width = image.shape[:2]
    longest_side = max(width, height)

    if longest_side <= max_size:
        return image

    scale = max_size / longest_side
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_LANCZOS4,
    )


def extract_elements(
    data: dict,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """Convert one PaddleOCR result and blank low-confidence text."""

    result_data = data["res"]
    elements = []

    for index, (text, score, box, polygon) in enumerate(
        zip(
            result_data["rec_texts"],
            result_data["rec_scores"],
            result_data["rec_boxes"],
            result_data["rec_polys"],
        )
    ):
        confidence = float(score)
        output_text = (
            str(text)
            if confidence >= confidence_threshold
            else ""
        )

        elements.append(
            {
                "index": index,
                "text": output_text,
                "confidence": confidence,
                "box": {
                    "x1": int(box[0]),
                    "y1": int(box[1]),
                    "x2": int(box[2]),
                    "y2": int(box[3]),
                },
                "polygon": [
                    [int(point[0]), int(point[1])]
                    for point in polygon
                ],
            }
        )

    return elements


def pdf_to_ocr_dict(
    input_path: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Render every PDF page and return its OCR results as a dictionary."""

    if not input_path.exists():
        raise FileNotFoundError(f"PDF not found: {input_path}")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            "confidence_threshold must be between 0.0 and 1.0"
        )

    ocr = PaddleOCR(
        lang="german",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    pdf = pdfium.PdfDocument(str(input_path))
    pages = []

    try:
        for page_index in range(len(pdf)):
            image = preprocess_image(
                render_pdf_page(pdf, page_index)
            )
            results = ocr.predict(image)
            result = next(iter(results))
            elements = extract_elements(
                result.json,
                confidence_threshold=confidence_threshold,
            )

            pages.append(
                {
                    "page": page_index + 1,
                    "image_size": {
                        "width": int(image.shape[1]),
                        "height": int(image.shape[0]),
                    },
                    "elements": elements,
                }
            )
    finally:
        pdf.close()

    return {
        "source_pdf": str(input_path),
        "confidence_threshold": confidence_threshold,
        "page_count": len(pages),
        "pages": pages,
    }