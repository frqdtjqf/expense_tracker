from .pdf_extractor import build_pdf_layout

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Build the layout of a PDF file and save it as a JSON file."
    )
    parser.add_argument(
        "input_pdf_path",
        type=Path,
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--output_json_path",
        type=Path,
        default=None,
        help="Path to the output JSON file. If not provided, the layout will not be saved.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to print layout information and save a debug image.",
    )
    parser.add_argument(
        "--output_image_path",
        type=Path,
        default=None,
        help="Path to the output debug image. If not provided, a default name will be used.",
    )

    args = parser.parse_args()

    layout = build_pdf_layout(
        args.input_pdf_path,
        output_json_path=args.output_json_path,
        debug=args.debug,
        output_image_path=args.output_image_path,
    )

    print("Layout extraction completed.")