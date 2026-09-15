from pathlib import Path


def load_json_file(file_path: Path) -> dict:
    """Load a JSON file and return its content as a dictionary."""
    import json
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: dict, file_path: Path) -> None:
    """Save a dictionary as a JSON file."""
    import json
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_image(file_path: Path):
    """Load an image from a file path."""
    import cv2
    return cv2.imread(str(file_path))

def save_image(image, output_path: Path) -> None:
    """Save an image to a file path."""
    import cv2

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        image,
    )