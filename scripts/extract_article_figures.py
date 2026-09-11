"""Extract embedded article figures from a PDF into a local output directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the article PDF")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(args.pdf))
    extracted = 0
    for page_number, page in enumerate(reader.pages, start=1):
        for image in getattr(page, "images", []):
            extracted += 1
            suffix = Path(image.name).suffix or ".bin"
            output_path = args.output_dir / f"extracted-{extracted:02d}{suffix}"
            output_path.write_bytes(image.data)
            print(f"page={page_number} source={image.name} output={output_path} bytes={len(image.data)}")


if __name__ == "__main__":
    main()