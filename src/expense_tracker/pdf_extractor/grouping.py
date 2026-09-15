from .models import OcrElement


def group_into_rows(
    elements: list[OcrElement],
    tolerance_factor: float = 0.5,
) -> list[list[OcrElement]]:
    """Group OCR elements by their vertical centers."""

    remaining = elements.copy()
    rows: list[list[OcrElement]] = []

    while remaining:
        reference = min(remaining, key=lambda element: element.center_y)
        row = [
            element
            for element in remaining
            if abs(reference.center_y - element.center_y)
            <= min(reference.height, element.height) * tolerance_factor
        ]
        row.sort(key=lambda element: element.min_x)
        rows.append(row)

        row_indices = {element.index for element in row}
        remaining = [
            element
            for element in remaining
            if element.index not in row_indices
        ]

    return rows


def group_into_columns(
    elements: list[OcrElement],
    tolerance_factor: float = 0.5,
) -> list[list[OcrElement]]:
    """Group OCR elements by their horizontal centers."""

    remaining = elements.copy()
    columns: list[list[OcrElement]] = []

    while remaining:
        reference = min(remaining, key=lambda element: element.center_x)
        column = [
            element
            for element in remaining
            if abs(reference.center_x - element.center_x)
            <= min(reference.width, element.width) * tolerance_factor
        ]
        column.sort(key=lambda element: element.center_y)
        columns.append(column)

        column_indices = {element.index for element in column}
        remaining = [
            element
            for element in remaining
            if element.index not in column_indices
        ]

    columns.sort(key=lambda column: column[0].center_x)
    return columns


def build_cell_grid(
    rows: list[list[OcrElement]],
    tolerance_factor: float = 0.5,
) -> list[list[dict | None]]:
    """Build a complete row/column grid, including spanning cells."""

    elements = [element for row in rows for element in row]
    columns = group_into_columns(elements, tolerance_factor)
    column_centers = [
        sum(element.center_x for element in column) / len(column)
        for column in columns
    ]
    column_by_index = {
        element.index: column_index
        for column_index, column in enumerate(columns)
        for element in column
    }

    grid: list[list[dict | None]] = []
    for row_index, row in enumerate(rows):
        cell_row: list[dict | None] = [None] * len(columns)

        for element in row:
            overlapping = [
                column_index
                for column_index, center_x in enumerate(column_centers)
                if element.min_x <= center_x <= element.max_x
            ]
            if not overlapping:
                overlapping = [column_by_index[element.index]]

            column_start = min(overlapping)
            column_end = max(overlapping)
            cell_row[column_start] = {
                "row": row_index,
                "column": column_start,
                "column_start": column_start,
                "column_end": column_end,
                "column_span": column_end - column_start + 1,
                "index": element.index,
                "text": element.text,
                "confidence": element.confidence,
                "polygon": [[x, y] for x, y in element.polygon],
                "box": element.box,
            }

        grid.append(cell_row)

    return grid
