from pathlib import Path

from ..helper import save_json_file
from .grouping import build_cell_grid, group_into_columns, group_into_rows
from .models import OcrElement
from .visualization import (
    create_abstract_layout_images,
    layout_to_table,
    print_layout_table,
)


def load_ocr_elements(data: dict) -> list[OcrElement]:
    """Convert OCR dictionary elements into OcrElement instances."""

    return [
        OcrElement(
            index=int(element["index"]),
            text=str(element.get("text", "")),
            confidence=float(element.get("confidence", 0.0)),
            polygon=[
                (int(point[0]), int(point[1]))
                for point in element["polygon"]
            ],
            box=element.get("box"),
        )
        for element in data["elements"]
    ]


def parse_ocr_table(rows: list[list[OcrElement]]) -> dict:
    """Serialize grouped OCR rows into plain Python data."""

    return {
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


def generate_page_layout(
    page_data: dict,
    tolerance_factor: float = 0.5,
) -> dict:
    """Build rows and a complete cell grid for one PDF page."""

    elements = load_ocr_elements(page_data)
    rows = group_into_rows(elements, tolerance_factor)
    cells = build_cell_grid(rows, tolerance_factor)

    return {
        "page": page_data["page"],
        "image_size": page_data.get("image_size"),
        "rows": parse_ocr_table(rows)["rows"],
        "columns": len(cells[0]) if cells else 0,
        "cells": cells,
    }


def generate_layout(
    ocr_data: dict,
    output_json: Path | None = None,
    debug: bool = False,
    output_image: Path | None = None,
) -> dict:
    """Build page layouts from OCR data and optionally save them as JSON."""

    del debug, output_image

    data = {
        "source_pdf": ocr_data["source_pdf"],
        "page_count": ocr_data["page_count"],
        "pages": [
            generate_page_layout(page_data)
            for page_data in ocr_data["pages"]
        ],
    }

    if output_json is not None:
        save_json_file(data, output_json)

    return data


__all__ = [
    "OcrElement",
    "build_cell_grid",
    "create_abstract_layout_images",
    "generate_layout",
    "generate_page_layout",
    "group_into_columns",
    "group_into_rows",
    "layout_to_table",
    "load_ocr_elements",
    "parse_ocr_table",
    "print_layout_table",
]