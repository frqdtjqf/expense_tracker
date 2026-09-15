from .layout import (
    create_abstract_layout_images,
    generate_layout,
    print_layout_table,
)
from .ocr import DEFAULT_CONFIDENCE_THRESHOLD, pdf_to_ocr_dict

def build_pdf_layout(
    input_pdf_path,
    output_json_path=None,
    debug=False,
    output_image_path=None,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
):
    """Build the layout of a PDF file and save it as a JSON file."""

    ocr_data = pdf_to_ocr_dict(
        input_pdf_path,
        confidence_threshold=confidence_threshold,
    )
    layout_data = generate_layout(
        ocr_data,
        output_json=output_json_path,
        debug=debug,
        output_image=output_image_path,
    )
    return layout_data


def visualize_layout_table(layout_data: dict) -> None:
    """Print layout data as a table grouped by PDF page and row."""

    print_layout_table(layout_data)


def visualize_abstract_layout(
    layout_data: dict,
    output_dir,
):
    """Create abstract, text-free images that preserve the page layout."""

    return create_abstract_layout_images(
        layout_data,
        output_dir,
    )