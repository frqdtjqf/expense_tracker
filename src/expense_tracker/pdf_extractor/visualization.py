from pathlib import Path

import cv2
import numpy as np

from ..helper import save_image


def layout_to_table(layout_data: dict) -> str:
    """Convert layout data into a readable text table."""

    headers = [
        "Page",
        "Row",
        "Column",
        "Span",
        "Index",
        "Text",
        "Confidence",
    ]
    table_rows = []

    for page in layout_data["pages"]:
        for row_index, cells in enumerate(page.get("cells", [])):
            for column_index, cell in enumerate(cells):
                if cell is None:
                    table_rows.append([
                        str(page["page"]),
                        str(row_index),
                        str(column_index),
                        "1",
                        "",
                        "",
                        "",
                    ])
                    continue
                table_rows.append([
                    str(page["page"]),
                    str(cell["row"]),
                    str(cell["column"]),
                    str(cell["column_span"]),
                    str(cell["index"]),
                    cell["text"],
                    f"{cell['confidence']:.2f}",
                ])

    if not table_rows:
        return "No OCR elements found."

    widths = [
        max(len(header), *(len(row[index]) for row in table_rows))
        for index, header in enumerate(headers)
    ]

    def format_row(row: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([
        format_row(headers),
        separator,
        *(format_row(row) for row in table_rows),
    ])


def print_layout_table(layout_data: dict) -> None:
    """Print layout data as a text table."""

    print(layout_to_table(layout_data))


def create_abstract_layout_images(
    layout_data: dict,
    output_dir: Path,
) -> list[Path]:
    """Create normalized table images from the cell grid."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for page in layout_data["pages"]:
        grid_rows = page.get("cells", [])
        column_count = max((len(row) for row in grid_rows), default=0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 1
        cell_padding = 24
        row_height = 52
        header_height = 48

        column_widths = [40] * column_count
        for row in grid_rows:
            for cell in row:
                if cell is None or not cell["text"]:
                    continue
                text_width = cv2.getTextSize(
                    str(cell["text"]),
                    font,
                    font_scale,
                    thickness,
                )[0][0]
                start = cell["column_start"]
                span = cell["column_span"]
                current_width = sum(column_widths[start:start + span])
                required_width = text_width + cell_padding
                if required_width > current_width:
                    column_widths[start] += required_width - current_width

        table_width = max(sum(column_widths) + 2, 240)
        table_height = header_height + row_height * len(grid_rows) + 2
        image = np.full((table_height, table_width, 3), 255, dtype=np.uint8)

        x_positions = [1]
        for column_width in column_widths:
            x_positions.append(x_positions[-1] + column_width)

        cv2.rectangle(
            image,
            (1, 1),
            (table_width - 2, header_height),
            (220, 220, 220),
            cv2.FILLED,
        )
        for column_index in range(column_count):
            cv2.putText(
                image,
                str(column_index + 1),
                (x_positions[column_index] + 12, 31),
                font,
                0.55,
                (40, 40, 40),
                thickness,
                cv2.LINE_AA,
            )

        for row_index, row in enumerate(grid_rows):
            top = header_height + row_index * row_height
            bottom = top + row_height
            fill = (250, 250, 250) if row_index % 2 == 0 else (238, 238, 238)
            cv2.rectangle(image, (1, top), (table_width - 2, bottom), fill, cv2.FILLED)

            for cell in row:
                if cell is None:
                    continue
                left = x_positions[cell["column_start"]]
                right = x_positions[cell["column_end"] + 1]
                cv2.rectangle(image, (left, top), (right, bottom), (190, 190, 190), 1)
                if cell["text"]:
                    cv2.putText(
                        image,
                        str(cell["text"]),
                        (left + 12, top + 32),
                        font,
                        font_scale,
                        (30, 30, 30),
                        thickness,
                        cv2.LINE_AA,
                    )

        output_path = output_dir / f"page_{page['page']:03d}_abstract.png"
        save_image(image, output_path)
        output_paths.append(output_path)

    return output_paths
