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

Parallel notebook pairing:
    If foo_parallel.ipynb exists alongside foo.ipynb, the two are merged into
    a single page with Serial/Parallel content tabs. See CONTRIBUTING for the
    cell tagging convention (cell.metadata.tags):
        tab:some_id    — in both notebooks: shown as paired Serial/Parallel tabs
        parallel-only  — in parallel notebook only: shown as a !!! note admonition

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
import tempfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).parent
NOTEBOOKS_DIR = ROOT / "notebooks"
DOCS_DIR = ROOT / "docs"

GITHUB_REPO = "smec-ethz/tatva-docs"


# ---------------------------------------------------------------------------
# Notebook → markdown conversion
# ---------------------------------------------------------------------------


def find_notebooks() -> list[Path]:
    """Return all notebooks, excluding _parallel variants (merged automatically)."""
    return sorted(
        nb for nb in NOTEBOOKS_DIR.rglob("*.ipynb")
        if not nb.stem.endswith("_parallel")
    )


def needs_conversion(nb_path: Path) -> bool:
    """Return True if the notebook (or its parallel pair) is newer than the markdown."""
    rel = nb_path.relative_to(NOTEBOOKS_DIR)
    md_path = DOCS_DIR / rel.parent / (nb_path.stem + ".md")
    if not md_path.exists():
        return True
    parallel_path = nb_path.parent / (nb_path.stem + "_parallel.ipynb")
    sources = [nb_path] + ([parallel_path] if parallel_path.exists() else [])
    return any(s.stat().st_mtime > md_path.stat().st_mtime for s in sources)


def convert_notebook(nb_path: Path, inline: bool = False) -> Path:
    """Convert a single notebook to markdown. Returns the output .md path."""
    rel = nb_path.relative_to(NOTEBOOKS_DIR)
    out_dir = DOCS_DIR / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    parallel_path = nb_path.parent / (nb_path.stem + "_parallel.ipynb")
    parallel_rel = (
        parallel_path.relative_to(NOTEBOOKS_DIR) if parallel_path.exists() else None
    )

    if parallel_path.exists():
        serial_nb = nbformat.read(nb_path, as_version=4)
        parallel_nb = nbformat.read(parallel_path, as_version=4)
        merged_nb = _merge_notebooks(serial_nb, parallel_nb)

        with tempfile.NamedTemporaryFile(
            suffix=".ipynb", mode="w", delete=False, encoding="utf-8"
        ) as f:
            nbformat.write(merged_nb, f)
            tmp_path = Path(f.name)

        try:
            result = subprocess.run(
                [
                    "jupyter", "nbconvert", "--to", "markdown",
                    "--output", nb_path.stem,
                    "--output-dir", str(out_dir),
                    "--TagRemovePreprocessor.enabled=True",
                    '--TagRemovePreprocessor.remove_input_tags=["remove_input"]',
                    str(tmp_path),
                ],
                capture_output=True,
                text=True,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        result = subprocess.run(
            [
                "jupyter", "nbconvert", "--to", "markdown",
                "--output-dir", str(out_dir),
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

    # Copy serial notebook as static asset for download
    asset_path = DOCS_DIR / "assets" / "notebooks" / rel
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nb_path, asset_path)

    # Copy parallel notebook as static asset too (if present)
    if parallel_rel:
        parallel_asset = DOCS_DIR / "assets" / "notebooks" / parallel_rel
        shutil.copy2(parallel_path, parallel_asset)

    _post_process(md_path, nb_rel=rel, parallel_nb_rel=parallel_rel)

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
# Notebook merging: serial + parallel → tabbed synthetic notebook
# ---------------------------------------------------------------------------


def _get_tags(cell: dict) -> list[str]:
    return cell.get("metadata", {}).get("tags", [])


def _get_tab_tag(cell: dict) -> str | None:
    for tag in _get_tags(cell):
        if tag.startswith("tab:"):
            return tag
    return None


def _extract_text_outputs(cell) -> str:
    """Return all text/stream outputs from a cell as a single string."""
    parts = []
    for output in cell.get("outputs", []):
        otype = output.get("output_type", "")
        if otype == "stream":
            parts.append("".join(output.get("text", [])))
        elif otype in ("execute_result", "display_data"):
            text = output.get("data", {}).get("text/plain", "")
            if isinstance(text, list):
                text = "".join(text)
            if text:
                parts.append(text)
    return "".join(parts).rstrip()


def _has_image_outputs(cell) -> bool:
    for output in cell.get("outputs", []):
        if any(k.startswith("image/") for k in output.get("data", {})):
            return True
    return False


def _associate_parallel_only_cells(parallel_cells: list) -> list:
    """Scan the parallel notebook and cluster parallel-only cells with their
    nearest tab cell so they can be embedded inside the Parallel tab block.

    Returns a list of events:
      ('shared', cell)
      ('tab', tab_id, cell, before_po, after_po)
        where before_po / after_po are parallel-only cells immediately
        preceding / following this tab cell (with no intervening shared cell).
      ('standalone-po', cell)
        parallel-only cells that are not adjacent to any tab cell.
    """
    n = len(parallel_cells)
    is_tab = [bool(_get_tab_tag(c)) for c in parallel_cells]
    is_po = ["parallel-only" in _get_tags(c) for c in parallel_cells]

    events: list = []
    i = 0
    while i < n:
        if is_po[i]:
            # Buffer parallel-only cells; attach to the next/prev tab
            events.append(("_pending_po", parallel_cells[i]))
            i += 1
        elif is_tab[i]:
            # Pop pending parallel-only cells as before_po
            before_po = [e[1] for e in events if e[0] == "_pending_po"]
            events = [e for e in events if e[0] != "_pending_po"]

            # Collect parallel-only cells immediately after this tab
            after_po = []
            j = i + 1
            while j < n and is_po[j]:
                after_po.append(parallel_cells[j])
                j += 1

            events.append(("tab", _get_tab_tag(parallel_cells[i]), parallel_cells[i], before_po, after_po))
            i = j  # skip the consumed after_po cells
        else:
            # Shared cell — flush any pending parallel-only cells as standalone
            for e in events:
                if e[0] == "_pending_po":
                    events_copy = [ev for ev in events if ev[0] != "_pending_po"]
                    # replace inline
                    break
            standalone = [("standalone-po", e[1]) for e in events if e[0] == "_pending_po"]
            events = [e for e in events if e[0] != "_pending_po"] + standalone
            events.append(("shared", parallel_cells[i]))
            i += 1

    # Flush any trailing pending-po as standalone
    result = []
    for e in events:
        if e[0] == "_pending_po":
            result.append(("standalone-po", e[1]))
        else:
            result.append(e)
    return result


def _make_tab_markdown_cell(
    serial_cell,
    parallel_cell,
    before_po: list | None = None,
    after_po: list | None = None,
) -> nbformat.NotebookNode:
    """Return a markdown cell with Serial/Parallel content tabs.

    parallel-only cells adjacent to this tab are embedded inside the Parallel
    tab block so they appear/disappear automatically with tab switching.
    Text outputs from the serial cell are embedded inside the Serial tab.
    Image outputs are handled by a separate remove_input cell.
    """
    serial_src = "".join(serial_cell["source"])
    parallel_src = "".join(parallel_cell["source"])
    serial_text_out = _extract_text_outputs(serial_cell)

    def indent(s: str) -> str:
        return "\n".join(f"    {line}" for line in s.splitlines())

    def po_admonition(po_cell) -> str:
        src = "".join(po_cell["source"])
        if po_cell["cell_type"] == "code":
            return f'!!! note "Parallel"\n\n    ```python\n{indent(src)}\n    ```'
        return f'!!! note "Parallel"\n\n{indent(src)}'

    def embed_in_tab(block: str) -> str:
        """Indent a block by 4 spaces for placement inside a tab."""
        return "    " + block.replace("\n", "\n    ").rstrip()

    # --- Serial tab ---
    serial_tab = '=== "Serial"\n\n    ```python\n' + indent(serial_src) + "\n    ```"
    if serial_text_out:
        serial_tab += "\n\n    ```\n" + indent(serial_text_out) + "\n    ```"

    # --- Parallel tab ---
    parallel_parts = ['=== "Parallel"\n']
    for po in (before_po or []):
        parallel_parts.append("\n" + embed_in_tab(po_admonition(po)) + "\n")
    parallel_parts.append("\n    ```python\n" + indent(parallel_src) + "\n    ```")
    for po in (after_po or []):
        parallel_parts.append("\n\n" + embed_in_tab(po_admonition(po)))

    parallel_tab = "".join(parallel_parts)

    return nbformat.v4.new_markdown_cell(source=serial_tab + "\n\n" + parallel_tab)


def _make_outputs_only_cell(serial_cell) -> nbformat.NotebookNode:
    """Return a code cell carrying only the serial cell's image outputs.

    Tagged remove_input so nbconvert hides the empty source block.
    Text outputs are embedded in the tab markdown cell instead.
    """
    cell = nbformat.v4.new_code_cell(source="")
    cell["outputs"] = [
        o for o in serial_cell.get("outputs", [])
        if any(k.startswith("image/") for k in o.get("data", {}))
        or o.get("output_type") == "display_data"
    ]
    cell["metadata"]["tags"] = ["remove_input"]
    return cell


def _make_standalone_parallel_only_cell(cell) -> nbformat.NotebookNode:
    """Wrap a parallel-only cell that has no adjacent tab in a note admonition."""
    src = "".join(cell["source"])

    def indent(s: str) -> str:
        return "\n".join(f"    {line}" for line in s.splitlines())

    if cell["cell_type"] == "code":
        content = f'!!! note "Parallel"\n\n    ```python\n{indent(src)}\n    ```'
    else:
        content = f'!!! note "Parallel"\n\n{indent(src)}'

    return nbformat.v4.new_markdown_cell(source=content)


def _merge_notebooks(
    serial_nb: nbformat.NotebookNode, parallel_nb: nbformat.NotebookNode
) -> nbformat.NotebookNode:
    """Build a synthetic notebook merging serial and parallel variants.

    - Variant cells (tab:id in both) → Serial/Parallel content tabs.
      Adjacent parallel-only cells are embedded inside the Parallel tab block
      so they appear/disappear with tab switching (no JS needed).
    - Standalone parallel-only cells (no adjacent tab) → !!! note admonition.
    - Shared cells → taken from the serial notebook (preserving its outputs).
    """
    serial_cells = serial_nb["cells"]
    parallel_cells = parallel_nb["cells"]

    serial_by_tag = {_get_tab_tag(c): c for c in serial_cells if _get_tab_tag(c)}
    serial_shared = [
        c for c in serial_cells
        if not _get_tab_tag(c) and "parallel-only" not in _get_tags(c)
    ]
    serial_shared_idx = 0

    synthetic: list[nbformat.NotebookNode] = []
    for event in _associate_parallel_only_cells(parallel_cells):
        kind = event[0]
        if kind == "shared":
            if serial_shared_idx < len(serial_shared):
                synthetic.append(serial_shared[serial_shared_idx])
                serial_shared_idx += 1
            else:
                synthetic.append(event[1])
        elif kind == "tab":
            _, tab_id, parallel_cell, before_po, after_po = event
            serial_cell = serial_by_tag.get(tab_id)
            if serial_cell:
                synthetic.append(_make_tab_markdown_cell(serial_cell, parallel_cell, before_po, after_po))
                if _has_image_outputs(serial_cell):
                    synthetic.append(_make_outputs_only_cell(serial_cell))
            else:
                synthetic.append(parallel_cell)
        elif kind == "standalone-po":
            synthetic.append(_make_standalone_parallel_only_cell(event[1]))

    merged = nbformat.from_dict(dict(serial_nb))
    merged["cells"] = synthetic
    return merged


# ---------------------------------------------------------------------------
# Post-processing: badges, collapse directives, tags
# ---------------------------------------------------------------------------


def _post_process(
    md_path: Path, nb_rel: Path, parallel_nb_rel: Path | None = None
) -> None:
    content = md_path.read_text()
    content, frontmatter = _extract_tags(content)
    content = _apply_cell_directives(content)
    content = _prepend_header(content, nb_rel, frontmatter, parallel_nb_rel)
    md_path.write_text(content)


def _prepend_header(
    content: str,
    nb_rel: Path,
    frontmatter: str = "",
    parallel_nb_rel: Path | None = None,
) -> str:
    """Prepend YAML frontmatter (if present), then Colab badge and download button(s)."""
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

    if parallel_nb_rel:
        parallel_download_path = f"/assets/notebooks/{parallel_nb_rel.as_posix()}"
        download_btns = (
            f'<a href="{download_path}" download="{nb_rel.name}" class="nb-download-btn">'
            f"{download_icon} Serial"
            "</a>"
            f'<a href="{parallel_download_path}" download="{parallel_nb_rel.name}" class="nb-download-btn">'
            f"{download_icon} Parallel"
            "</a>"
        )
    else:
        download_btns = (
            f'<a href="{download_path}" download="{nb_rel.name}" class="nb-download-btn">'
            f"{download_icon} Download"
            "</a>"
        )

    header = (
        '<div class="nb-header">'
        f'<a href="{colab_url}" target="_blank">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>'
        "</a>"
        f"{download_btns}"
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
