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


def generate_page_layout(
    page_data: dict,
    tolerance_factor: float = 0.5,
) -> dict:
    """Build a complete cell grid for one PDF page."""

    elements = load_ocr_elements(page_data)
    cells = build_cell_grid(elements, tolerance_factor)

    return {
        "page": page_data["page"],
        "image_size": page_data.get("image_size"),
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
    "print_layout_table",
]