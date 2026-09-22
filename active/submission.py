"""
File: submission.py
Project: Yizkor Pipeline
Author: Elijah Greenberg
Description: File submission handling for Yizkor Books.
"""


"""
Desired Method of Upload:
 - From Computer
 - Single PDF file only

These submissions are stored locally on the user's computer.
"""

import os


ACCEPTED_FILE_FORMATS = ['.pdf']


# ---------------------------------------------------------------------------
# Submission prompts
# ---------------------------------------------------------------------------
def get_submission_path():
    print("Please enter the path to the PDF file:")
    return input("File path: ")

# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------
def verify_file_path(path):
    if not os.path.isfile(path):
        raise ValueError("Invalid file path")

    _, ext = os.path.splitext(path)
    if ext.lower() not in ACCEPTED_FILE_FORMATS:
        raise ValueError("File format not accepted")

    return path


def normalize_path(path):
    if not isinstance(path, str):
        return path

    return path.strip().strip('"').strip("'")


def verify_submission_path(path):
    normalized_path = normalize_path(path)

    if not os.path.isfile(normalized_path):
        raise ValueError("Invalid file path")

    return verify_file_path(normalized_path)

# ---------------------------------------------------------------------------
# Pdf Splitting
# ---------------------------------------------------------------------------
def _render_pdf_page_to_png(pdf_path, page_index, output_dir):
    """Render a single PDF page to a PNG file in a worker process."""
    import fitz

    document = fitz.open(pdf_path)
    page = document.load_page(page_index)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

    page_path = os.path.join(output_dir, f"page_{page_index + 1:03d}.png")
    pixmap.save(page_path)
    document.close()
    return page_path


def split_pdf_into_pages(path, output_dir=None, max_workers=None):
    """
    Split a PDF into one-page PNG images and return the paths to the extracted files.

    Default storage location:
        next to the original PDF, inside a folder named '<original_name>_pages/'

    Example:
        C:/documents/submission.pdf
        -> C:/documents/submission_pages/page_001.png
        -> C:/documents/submission_pages/page_002.png
    """
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required to convert PDF pages to PNG. Install it with: pip install pymupdf"
        ) from exc

    validated_path = verify_submission_path(path)

    if output_dir is None:
        pdf_dir = os.path.dirname(os.path.abspath(validated_path))
        pdf_name = os.path.splitext(os.path.basename(validated_path))[0]
        output_dir = os.path.join(pdf_dir, f"{pdf_name}_pages")

    os.makedirs(output_dir, exist_ok=True)

    document = fitz.open(validated_path)
    try:
        page_count = document.page_count
        if page_count == 0:
            raise ValueError("PDF contains no pages")
    finally:
        document.close()

    if max_workers is None:
        max_workers = max(1, min(os.cpu_count() or 1, page_count, 4))

    page_paths = []
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_render_pdf_page_to_png, validated_path, page_index, output_dir): page_index
            for page_index in range(page_count)
        }

        for future in as_completed(futures):
            page_paths.append(future.result())

    page_paths.sort(key=lambda page_path: int(os.path.basename(page_path).replace("page_", "").replace(".png", "")))
    return page_paths

# ---------------------------------------------------------------------------
# Verify Quality of pngs
# ---------------------------------------------------------------------------

def verify_page_quality(page_paths, blank_threshold=0.99, min_width=512, min_height=512):
    """
    Verify the quality of the extracted PNG pages in a single pass.

    Rules:
        - blank pages are replaced with a white placeholder to preserve page numbering
        - low-resolution pages are flagged but not rejected
        - duplicate pages are intentionally not checked
    """
    from PIL import Image
    import numpy as np

    blank_pages = []
    low_res_pages = []

    for page_path in page_paths:
        with Image.open(page_path) as img:
            width, height = img.size

            if width < min_width or height < min_height:
                low_res_pages.append(page_path)

            sample = img.convert("L")
            sample.thumbnail((64, 64))
            sample_array = np.asarray(sample)
            white_ratio = np.mean(sample_array > 250)

            if white_ratio >= blank_threshold:
                blank_pages.append(page_path)
                blank_page = Image.new("RGB", (width, height), color=(255, 255, 255))
                blank_page.save(page_path)

    if blank_pages:
        print("Warning: Blank pages detected and replaced with blank placeholders:")
        for page in blank_pages:
            print(f" - {page}")

    if low_res_pages:
        print("Warning: Low resolution pages detected:")
        for page in low_res_pages:
            print(f" - {page}")

    return {
        "blank_pages": blank_pages,
        "low_res_pages": low_res_pages,
    }


def _extract_page_number(page_path):
    """Return the page number implied by a page image filename, if present."""
    filename = os.path.basename(page_path)
    stem = os.path.splitext(filename)[0]

    if not stem.startswith("page_"):
        return None

    number_text = stem.replace("page_", "")
    if not number_text.isdigit():
        return None

    return int(number_text)


def _format_page_ranges(page_paths):
    """Convert page path values into a compact, human-readable page-range summary."""
    page_numbers = []
    for page_path in page_paths:
        page_number = _extract_page_number(page_path)
        if page_number is not None:
            page_numbers.append(page_number)

    if not page_numbers:
        return "none"

    page_numbers = sorted(set(page_numbers))
    ranges = []
    start = prev = page_numbers[0]

    for current in page_numbers[1:]:
        if current == prev + 1:
            prev = current
            continue

        if start == prev:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{prev}")

        start = prev = current

    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")

    return ", ".join(ranges)


def confirm_quality_issues(quality_report):
    """Ask the user whether to continue when any page-quality issues are flagged."""
    flag_labels = {
        "blank_pages": "blank pages",
        "low_res_pages": "low-resolution pages",
    }

    flagged_categories = []
    total_flag_count = 0

    for key, label in flag_labels.items():
        pages = quality_report.get(key, [])
        if pages:
            flagged_categories.append((label, pages))
            total_flag_count += len(pages)

    if not flagged_categories:
        return True

    print(f"Quality review found {total_flag_count} flagged pages in total.")
    print("The extracted pages have the following quality flags:")
    for label, pages in flagged_categories:
        page_ranges = _format_page_ranges(pages)
        print(f" - {label}: pages {page_ranges}")

    while True:
        response = input("Do you want to continue with this submission? (y/n): ").strip().lower()
        if response in ["y", "yes"]:
            return True
        if response in ["n", "no"]:
            return False
        print("Please enter 'y' for yes or 'n' for no.")

# ---------------------------------------------------------------------------
# Submission flow
# ---------------------------------------------------------------------------
def handle_submission_request():
    while True:
        path = get_submission_path()

        try:
            validated_path = verify_submission_path(path)
            page_paths = split_pdf_into_pages(validated_path)
            quality_report = verify_page_quality(page_paths)

            if not confirm_quality_issues(quality_report):
                print("Submission canceled by user after quality review.")
                return None

            print("Submission successful. PDF validated and split into pages:")
            for page_path in page_paths:
                print(page_path)

            return {
                "source_pdf": validated_path,
                "page_images": page_paths,
                "quality_report": quality_report,
            }
        except ValueError as error:
            print(f"Submission error: {error}")
            print("Please try a different PDF path.")


def main():
    handle_submission_request()


if __name__ == "__main__":
    main()