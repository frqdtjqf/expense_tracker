from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from paddleocr import PaddleOCR


# =============================================================================
# Configuration
# =============================================================================

INPUT_IMAGE = Path("data/receipts/image3.pdf")
DEBUG_DIR = Path("data/debug")

MAX_IMAGE_SIZE = 4000
JPEG_QUALITY = 95


# =============================================================================
# Image preprocessing
# =============================================================================

def render_pdf_page(
    input_path: Path,
    page_index: int = 0,
    scale: int = 4,
) -> cv2.typing.MatLike:
    """Render one PDF page as a BGR OpenCV image."""

    pdf = pdfium.PdfDocument(str(input_path))

    if page_index >= len(pdf):
        pdf.close()
        raise IndexError(
            f"PDF page does not exist: {page_index}"
        )

    page = pdf[page_index]
    bitmap = page.render(scale=scale)
    rgb_image = np.asarray(bitmap.to_pil())

    page.close()
    pdf.close()

    return cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR,
    )

def preprocess_image(
    input_path: Path,
    output_path: Path,
    max_size: int = MAX_IMAGE_SIZE,
) -> cv2.typing.MatLike:
    """
    Load image and reduce its resolution so that the longest side
    is at most max_size.

    The original image is never modified.
    """

    if input_path.suffix.lower() == ".pdf":
        image = render_pdf_page(input_path)
    else:
        image = cv2.imread(
            str(input_path),
            cv2.IMREAD_COLOR,
        )

    if image is None:
        raise ValueError(
            f"Could not read image: {input_path}"
        )

    height, width = image.shape[:2]

    original_width = width
    original_height = height

    longest_side = max(width, height)

    print()
    print("=" * 70)
    print("IMAGE PREPROCESSING")
    print("=" * 70)

    print(
        f"Original size: {original_width} x {original_height}"
    )

    if longest_side > max_size:

        scale = max_size / longest_side

        new_width = round(width * scale)
        new_height = round(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LANCZOS4,
        )

        print(
            f"Resized:       {new_width} x {new_height}"
        )

        print(
            f"Scale factor:  {scale:.6f}"
        )

    else:

        print(
            "Resize:        not necessary"
        )

        print(
            f"Size:          {width} x {height}"
        )

    success = cv2.imwrite(
        str(output_path),
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
        ],
    )

    if not success:
        raise OSError(
            f"Could not write image: {output_path}"
        )

    print(
        f"Output:        {output_path}"
    )

    return image


# =============================================================================
# Drawing helpers
# =============================================================================

def draw_polygon(
    image: cv2.typing.MatLike,
    polygon: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a polygon on an OpenCV image."""

    for i in range(len(polygon)):

        start = polygon[i]
        end = polygon[(i + 1) % len(polygon)]

        cv2.line(
            image,
            start,
            end,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_label(
    image: cv2.typing.MatLike,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a readable label with a white background."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    (
        label_width,
        label_height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    image_height, image_width = image.shape[:2]

    # Prefer label above the OCR element.
    label_x = max(0, x)
    label_y = y - label_height - baseline - 6

    # If there is not enough space above,
    # put it below.
    if label_y < 0:
        label_y = y + 4

    # Keep label inside image.
    label_x = min(
        label_x,
        max(0, image_width - label_width - 10),
    )

    label_y = min(
        label_y,
        max(0, image_height - label_height - baseline - 2),
    )

    # Background
    cv2.rectangle(
        image,
        (
            label_x,
            label_y,
        ),
        (
            min(
                image_width - 1,
                label_x + label_width + 8,
            ),
            min(
                image_height - 1,
                label_y
                + label_height
                + baseline
                + 6,
            ),
        ),
        (255, 255, 255),
        cv2.FILLED,
    )

    # Text
    cv2.putText(
        image,
        text,
        (
            label_x + 4,
            label_y + label_height + 2,
        ),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# =============================================================================
# OCR
# =============================================================================

def run_ocr(
    image_path: Path,
) -> list[dict]:
    """
    Run PaddleOCR and convert its output into a clean Python structure.
    """

    print()
    print("=" * 70)
    print("PADDLEOCR")
    print("=" * 70)

    print(
        f"Input: {image_path}"
    )

    ocr = PaddleOCR(
        lang="german",

        # IMPORTANT:
        # PaddleOCR must not rotate or geometrically transform
        # the document behind our back.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    results = ocr.predict(
        str(image_path)
    )

    elements: list[dict] = []

    for result_index, result in enumerate(results):

        data = result.json
        res = data["res"]

        texts = res["rec_texts"]
        scores = res["rec_scores"]
        boxes = res["rec_boxes"]
        polys = res["rec_polys"]

        print()
        print(
            f"Result {result_index}"
        )

        print(
            f"Detected elements: {len(texts)}"
        )

        for index, (
            text,
            score,
            box,
            poly,
        ) in enumerate(
            zip(
                texts,
                scores,
                boxes,
                polys,
            )
        ):

            text = str(text)
            score = float(score)

            box = [
                int(value)
                for value in box
            ]

            x1, y1, x2, y2 = box

            polygon = [
                [
                    int(point[0]),
                    int(point[1]),
                ]
                for point in poly
            ]

            element = {
                "index": index,
                "text": text,
                "confidence": score,
                "box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "polygon": polygon,
            }

            elements.append(element)

            # Console debug output
            print(
                f"{index:03d} "
                f"{score:.2f} "
                f"box=({x1},{y1},{x2},{y2}) "
                f"{text!r}"
            )

    return elements


# =============================================================================
# Visualization
# =============================================================================

def create_debug_image(
    image_path: Path,
    elements: list[dict],
    output_path: Path,
    draw_boxes: bool = False,
    draw_polys: bool = True,
    draw_centers: bool = False,
    draw_labels: bool = True,
) -> None:
    """
    Draw OCR results on top of the exact image that was passed to OCR.

    RED   = rec_boxes
    GREEN = rec_polys
    BLUE  = center of rec_box
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    image_height, image_width = image.shape[:2]

    print()
    print("=" * 70)
    print("DEBUG VISUALIZATION")
    print("=" * 70)

    print(
        f"Image size: {image_width} x {image_height}"
    )

    for element in elements:

        text = element["text"]
        confidence = element["confidence"]

        box = element["box"]

        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]

        polygon = [
            (
                point[0],
                point[1],
            )
            for point in element["polygon"]
        ]

        # ---------------------------------------------------------------------
        # RED = rec_boxes
        # ---------------------------------------------------------------------

        if draw_boxes:
            cv2.rectangle(
                image,
                (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        # ---------------------------------------------------------------------
        # GREEN = rec_polys
        # ---------------------------------------------------------------------

        if draw_polys:
            draw_polygon(
            image,
            polygon,
            color=(0, 255, 0),
            thickness=2,
        )

        # ---------------------------------------------------------------------
        # BLUE = center
        # ---------------------------------------------------------------------

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if draw_centers:
            cv2.circle(
                image,
                (
                    center_x,
                    center_y,
                ),
                5,
                (255, 0, 0),
                -1,
            )

        # ---------------------------------------------------------------------
        # Label
        # ---------------------------------------------------------------------

        label = (
            f"{element['index']}: "
            f"{text} "
            f"[{confidence:.2f}]"
        )

        if draw_labels:
            draw_label(
                image,
                label,
                x1,
                y1,
                color=(0, 0, 255),
            )

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    draw_label(
        image,
        "RED = rec_boxes | GREEN = rec_polys | BLUE = center",
        10,
        30,
        color=(0, 0, 0),
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise OSError(
            f"Could not write debug image: {output_path}"
        )

    print(
        f"Debug image: {output_path}"
    )


# =============================================================================
# JSON
# =============================================================================

def save_json(
    source_image: Path,
    processed_image: Path,
    elements: list[dict],
    output_path: Path,
) -> None:
    """Save OCR results as structured JSON."""

    image = cv2.imread(
        str(processed_image),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {processed_image}"
        )

    image_height, image_width = image.shape[:2]

    json_data = {
        "source_image": str(source_image),
        "processed_image": str(processed_image),

        "image_size": {
            "width": image_width,
            "height": image_height,
        },

        "ocr": {
            "engine": "PaddleOCR",

            "document_orientation_classification": False,
            "document_unwarping": False,

            "elements_count": len(elements),
        },

        "elements": elements,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"OCR JSON: {output_path}"
    )


# =============================================================================
# Main pipeline
# =============================================================================

def process_receipt(
    image_path: Path,
) -> None:
    """Complete OCR debug pipeline."""

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("RECEIPT OCR DEBUG")
    print("=" * 70)

    print(
        f"Source: {image_path}"
    )

    # -------------------------------------------------------------------------
    # 1. Preprocess
    # -------------------------------------------------------------------------

    normalized_path = (
        DEBUG_DIR
        / f"{image_path.stem}_normalized.jpg"
    )

    preprocess_image(
        image_path,
        normalized_path,
    )

    # -------------------------------------------------------------------------
    # 2. OCR
    # -------------------------------------------------------------------------

    elements = run_ocr(
        normalized_path
    )

    # -------------------------------------------------------------------------
    # 3. JSON
    # -------------------------------------------------------------------------

    output_json = (
        DEBUG_DIR
        / f"{image_path.stem}_ocr.json"
    )

    save_json(
        source_image=image_path,
        processed_image=normalized_path,
        elements=elements,
        output_path=output_json,
    )

    # -------------------------------------------------------------------------
    # 4. Visualization
    # -------------------------------------------------------------------------

    output_image = (
        DEBUG_DIR
        / f"{image_path.stem}_ocr.png"
    )

    create_debug_image(
        image_path=normalized_path,
        elements=elements,
        output_path=output_image,
    )

    # -------------------------------------------------------------------------
    # 5. Summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"OCR elements: {len(elements)}"
    )

    print(
        f"Normalized:   {normalized_path}"
    )

    print(
        f"JSON:         {output_json}"
    )

    print(
        f"Debug image:  {output_image}"
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    process_receipt(
        INPUT_IMAGE
    )
