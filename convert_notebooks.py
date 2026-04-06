# /// script
# requires-python = ">=3.11"
# dependencies = ["nbconvert", "nbformat"]
# ///
"""Convert all .ipynb notebooks to markdown with extracted images.

Source notebooks live in notebooks/ and mirror the docs/ structure.
Generated .md files (and image dirs) are written into docs/.
A copy of each .ipynb is placed in docs/assets/notebooks/ so the site
can serve it as a downloadable asset.

Usage:
    uv run convert_notebooks.py           # convert all notebooks
    uv run convert_notebooks.py --check   # show what would be converted
    uv run convert_notebooks.py --inline  # embed images as base64 in the md

Output layout (default):
    notebooks/examples/linear_elasticity.ipynb        <- source, edit this
    docs/examples/linear_elasticity.md                <- generated, committed
    docs/examples/linear_elasticity_files/            <- generated, committed
        output_5_0.png
    docs/assets/notebooks/examples/linear_elasticity.ipynb  <- served for download
"""

import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOKS_DIR = ROOT / "notebooks"
DOCS_DIR = ROOT / "docs"

GITHUB_REPO = "smec-ethz/tatva-docs"


# ---------------------------------------------------------------------------
# Notebook → markdown conversion
# ---------------------------------------------------------------------------


def find_notebooks() -> list[Path]:
    return sorted(NOTEBOOKS_DIR.rglob("*.ipynb"))


def needs_conversion(nb_path: Path) -> bool:
    """Return True if the notebook is newer than its corresponding markdown."""
    rel = nb_path.relative_to(NOTEBOOKS_DIR)
    md_path = DOCS_DIR / rel.parent / (nb_path.stem + ".md")
    if not md_path.exists():
        return True
    return nb_path.stat().st_mtime > md_path.stat().st_mtime


def convert_notebook(nb_path: Path, inline: bool = False) -> Path:
    """Convert a single notebook to markdown. Returns the output .md path."""
    rel = nb_path.relative_to(NOTEBOOKS_DIR)
    out_dir = DOCS_DIR / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "markdown",
            "--output-dir",
            str(out_dir),
            str(nb_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        raise RuntimeError(f"nbconvert failed for {nb_path}")

    md_path = out_dir / (nb_path.stem + ".md")

    if inline:
        _inline_images(md_path)

    # Copy notebook as static asset for download
    asset_path = DOCS_DIR / "assets" / "notebooks" / rel
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nb_path, asset_path)

    # Apply post-processing (badges, collapse directives, tags)
    _post_process(md_path, nb_rel=rel)

    return md_path


def _inline_images(md_path: Path) -> None:
    """Replace extracted image file references with base64 data URIs."""
    content = md_path.read_text()

    def replace(m: re.Match) -> str:
        img_path = md_path.parent / m.group(1)
        if img_path.exists() and img_path.suffix.lower() == ".png":
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            img_path.unlink()
            return f"![](data:image/png;base64,{b64})"
        return m.group(0)

    content = re.sub(r"!\[.*?\]\(([^)]+\.png)\)", replace, content)
    md_path.write_text(content)

    files_dir = md_path.parent / (md_path.stem + "_files")
    if files_dir.is_dir() and not any(files_dir.iterdir()):
        files_dir.rmdir()


# ---------------------------------------------------------------------------
# Post-processing: badges, collapse directives, tags
# ---------------------------------------------------------------------------


def _post_process(md_path: Path, nb_rel: Path) -> None:
    content = md_path.read_text()
    content, frontmatter = _extract_tags(content)  # remove YAML frontmatter
    content = _apply_cell_directives(content)
    content = _prepend_header(
        content, nb_rel, frontmatter
    )  # frontmatter + badges before content
    md_path.write_text(content)


def _prepend_header(content: str, nb_rel: Path, frontmatter: str = "") -> str:
    """Prepend YAML frontmatter (if present), then Colab badge and download button."""
    tag = os.environ.get("LIB_TAG", "main")
    nb_rel_str = nb_rel.as_posix()

    colab_url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{tag}/notebooks/{nb_rel_str}"
    download_path = f"/assets/notebooks/{nb_rel_str}"

    download_icon = (
        '<svg class="nb-download-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M12 16l-6-6 1.41-1.41L11 13.17V4h2v9.17l3.59-3.58L18 11l-6 6z"/>'
        '<path d="M5 18h14v2H5z"/>'
        "</svg>"
    )
    header = (
        '<div class="nb-header">'
        f'<a href="{colab_url}" target="_blank">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>'
        "</a>"
        f'<a href="{download_path}" download="{nb_rel.name}" class="nb-download-btn">'
        f"{download_icon} Download"
        "</a>"
        "</div>\n\n"
    )
    prefix = frontmatter + "\n\n" if frontmatter else ""
    return prefix + header + content


def _extract_tags(content: str) -> tuple[str, str]:
    """Remove YAML frontmatter from content and return (cleaned_content, frontmatter).

    Frontmatter is a block delimited by ``---`` at the very start of the file.
    """
    match = re.match(r"^(---\n.*?\n---)\n*", content, flags=re.DOTALL)
    if not match:
        return content, ""

    frontmatter = match.group(1)
    content = content[match.end() :]
    return content, frontmatter


def _apply_cell_directives(content: str) -> str:
    """Transform # [collapse: code] / # [collapse: all] / # [output: hide] directives."""
    pattern = r"```python\s+# \[(.*?)\]\s*(.*?)\n(.*?)\n```(\n.*?)(?=\n```python|\Z)"

    def transform(m: re.Match) -> str:
        directives = m.group(1).lower()
        title = m.group(2).strip() or "Code"
        code_body = m.group(3)
        output_body = (m.group(4) or "").strip()

        def indent(text: str) -> str:
            return "\n".join(f"    {line}" for line in text.splitlines())

        if "collapse: code" in directives or "collapse: all" in directives:
            code_block = (
                f'??? example "{title}"\n    ```python\n{indent(code_body)}\n    ```'
            )
        else:
            code_block = f"```python\n{code_body}\n```"

        if output_body:
            if "output: hide" in directives or "collapse: all" in directives:
                output_block = f'??? info "Output"\n{indent(output_body)}'
            else:
                output_block = output_body
        else:
            output_block = ""

        return f"{code_block}\n\n{output_block}\n\n"

    return re.sub(pattern, transform, content, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true", help="list notebooks without converting"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="convert all notebooks even if they are up to date",
    )
    parser.add_argument(
        "--inline",
        action="store_true",
        help="embed images as base64 instead of separate files",
    )
    args = parser.parse_args()

    notebooks = find_notebooks()
    if not notebooks:
        print("No notebooks found.")
        return

    if args.check:
        print(f"Found {len(notebooks)} notebook(s):")
        for nb in notebooks:
            print(f"  {nb.relative_to(ROOT)}")
        return

    stale = (
        notebooks if args.force else [nb for nb in notebooks if needs_conversion(nb)]
    )
    if not stale:
        print("All notebooks are up to date.")
        return

    mode = "inline base64" if args.inline else "separate image files"
    print(f"Converting {len(stale)}/{len(notebooks)} changed notebook(s) [{mode}]...\n")

    ok, failed = 0, 0
    for nb in stale:
        try:
            md = convert_notebook(nb, inline=args.inline)
            print(f"  ok  {nb.relative_to(ROOT)}  ->  {md.relative_to(ROOT)}")
            ok += 1
        except RuntimeError:
            print(f"  FAIL  {nb.relative_to(ROOT)}")
            failed += 1

    print(f"\nDone: {ok} converted, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
