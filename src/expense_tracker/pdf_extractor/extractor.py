from .layout import generate_layout
from .ocr import pdf_to_ocr_dict

def build_pdf_layout(input_pdf_path, output_json_path=None, debug=False, output_image_path=None):
    """Build the layout of a PDF file and save it as a JSON file."""

    ocr_data = pdf_to_ocr_dict(input_pdf_path)
    layout_data = generate_layout(
        ocr_data,
        output_json=output_json_path,
        debug=debug,
        output_image=output_image_path,
    )
    return layout_data