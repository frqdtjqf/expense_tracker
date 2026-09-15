import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from paddle.static import data

from ..helper import load_json_file, save_json_file, load_image, save_image


@dataclass
class OcrElement:
    index: int
    text: str
    confidence: float
    polygon: list[tuple[int, int]]
    box: dict[str, int] | None = None

    @property
    def min_x(self) -> int:
        return min(x for x, _ in self.polygon)

    @property
    def max_x(self) -> int:
        return max(x for x, _ in self.polygon)

    @property
    def min_y(self) -> int:
        return min(y for _, y in self.polygon)

    @property
    def max_y(self) -> int:
        return max(y for _, y in self.polygon)

    @property
    def width(self) -> int:
        return self.max_x - self.min_x

    @property
    def height(self) -> int:
        return self.max_y - self.min_y

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) / 2

    @property
    def center_y(self) -> float:
        return (self.min_y + self.max_y) / 2


def load_ocr_elements(data: dict) -> list[OcrElement]:
    """Loads OCR elements from JSON dictionary and returns a list of OcrElement instances."""
    elements = []

    for element in data["elements"]:
        polygon = [
            (int(point[0]), int(point[1]))
            for point in element["polygon"]
        ]

        elements.append(
            OcrElement(
                index=element.get("index"),
                text=element.get("text"),
                confidence=element.get("confidence"),
                box=element.get("box"),
                polygon=polygon
            )
        )

    return elements


def vertical_distance(a: OcrElement, b: OcrElement) -> float:
    """
    Vertical distance between two OCR elements.
    Uses the center y-coordinates of the elements to calculate the distance.
    """

    return abs(a.center_y - b.center_y)


def row_tolerance(
    a: OcrElement,
    b: OcrElement,
    factor: float = 0.5,
) -> float:
    """
    Calculates the allowed vertical distance.

    The tolerance is based on the size of the OCR elements.
    This makes the method work better with different font sizes.
    """

    reference_height = min(a.height, b.height)

    return reference_height * factor


def same_row(
    a: OcrElement,
    b: OcrElement,
    tolerance_factor: float = 0.5,
) -> bool:
    """
    Determines if two OCR elements are in the same row based on their vertical distance and a tolerance factor.
    """

    distance = vertical_distance(a, b)
    tolerance = row_tolerance(a, b, tolerance_factor)
    is_same_row = distance <= tolerance

    return is_same_row


def group_into_rows(
    elements: list[OcrElement],
    tolerance_factor: float = 0.5,
) -> list[list[OcrElement]]:
    """
    Groups OCR elements into rows based on their vertical positions and a specified tolerance factor.

    Starts with the topmost element and finds all elements that are in the same row based on the vertical distance and tolerance.
    The process is repeated until all elements are assigned to rows.
    """

    remaining = elements.copy()
    rows: list[list[OcrElement]] = []

    while remaining:
        # topmost element as reference for the current row
        reference = min(
            remaining,
            key=lambda element: element.center_y,
        )

        row = [reference]

        # search for other elements in the same row in remaining elements
        for element in remaining:
            if element is reference:
                continue

            if same_row(
                reference,
                element,
                tolerance_factor=tolerance_factor,
            ):
                row.append(element)

        # sort elements in the row by their horizontal position (min_x)
        row.sort(key=lambda element: element.min_x)

        rows.append(row)

        # remove elements that are already assigned to the current row from the remaining elements
        row_indices = {
            element.index
            for element in row
        }

        remaining = [
            element
            for element in remaining
            if element.index not in row_indices
        ]

    return rows

# --------- debugging and visualization functions ---------
def draw_polygon(
    image: np.ndarray,
    polygon: list[tuple[int, int]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """draws an OCR-Polygon."""

    points = np.array(polygon, dtype=np.int32)

    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=color,
        thickness=thickness,
    )


def draw_layout(
    image_path: Path,
    rows: list[list[OcrElement]],
    output_path: Path,
) -> None:
    """
    Draws the detected rows on the image.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Bild konnte nicht geladen werden: {image_path}"
        )

    for row_index, row in enumerate(rows):
        for element in row:
            draw_polygon(
                image,
                element.polygon,
            )

            # Position des Labels etwas oberhalb des Polygons.
            label_x = element.min_x
            label_y = max(20, element.min_y - 5)

            cv2.putText(
                image,
                f"R{row_index}",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    save_image(
        image,
        output_path,
    )


def print_layout(rows: list[list[OcrElement]]) -> None:
    """Prints the detected structure on the console."""

    for row_index, row in enumerate(rows):
        texts = [
            element.text
            for element in row
        ]

        print(
            f"Row {row_index}: "
            + " | ".join(texts)
        )
# ------------------------------------------------------

def parse_ocr_table(rows: list[list[OcrElement]]) -> dict:
    """
    Parses the detected rows into a dict structure that can be saved as JSON.
    The structure is a list of rows, where each row contains a list of elements with their text and bounding box information.
    """

    data = {
            "rows": [
                {
                    "row": row_index,
                    "elements": [
                        {
                            "index": element.index,
                            "text": element.text,
                            "confidence": element.confidence,
                            "polygon": [
                                [x, y]
                                for x, y in element.polygon
                            ],
                            "min_x": element.min_x,
                            "max_x": element.max_x,
                            "min_y": element.min_y,
                            "max_y": element.max_y,
                            "center_x": element.center_x,
                            "center_y": element.center_y,
                        }
                        for element in row
                    ],
                }
                for row_index, row in enumerate(rows)
            ]
        }

    return data


def generate_layout(ocr_data: dict, output_json: Path | None = None, debug: bool = False, output_image: Path | None = None) -> dict:

    elements = load_ocr_elements(ocr_data)

    rows = group_into_rows(
        elements,
        tolerance_factor=0.5,
    )

    if debug:
        print_layout(rows)

        image_path = Path(ocr_data.get("image_path", "unknown_image.png"))
        if output_image is None: output_image = Path("debug_layout.png")
        draw_layout(
            image_path,
            rows,
            output_image,
        )

    data = parse_ocr_table(rows)
    if output_json is not None:
        save_json_file(
            data,
            output_json,
        )
        
    return data
