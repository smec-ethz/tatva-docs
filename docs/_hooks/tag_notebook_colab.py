import os
import glob
import json
import logging
import re

def on_page_markdown(markdown, page, config, files):
    if not page.file.src_uri.endswith(".ipynb"):
        return markdown

    # 1. COLAB LINK LOGIC
    tag = os.environ.get("LIB_TAG", "main")
    base_url = f"https://colab.research.google.com/github/smec-ethz/tatva-docs/blob/{tag}/docs/"
    colab_url = f"{base_url}{page.file.src_uri}"
    badge_md = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})\n\n"
    markdown = badge_md + markdown

    # 2. SEPARATE CODE AND OUTPUT LOGIC
    # This regex captures the code block (Group 3) and the following block (Group 4)
    # The [directives] are in Group 1, and the Title is in Group 2.
    pattern = r"```python\s+# \[(.*?)\]\s*(.*?)\n(.*?)\n```(?:\s*\n(.*?)(?=\n\n|(?:\n```)|$)|)"


    def transform_cell(match):
        directives = match.group(1).lower()
        title = match.group(2).strip() or "Code"
        code_body = match.group(3)
        output_body = match.group(4) or ""

        # --- 1. HANDLE OUTPUT HIDING ---
        # If the directive contains 'output: hide', we simply wipe the output_body.
        if "output: hide" in directives:
            output_body = ""

        # --- 2. COLLAPSE LOGIC ---

        # Case A: Collapse Code ONLY
        # (The output, if it exists, stays visible outside the box)
        if "collapse: code" in directives:
            indented_code = "\n".join([f"    {line}" for line in code_body.splitlines()])
            code_block = f'??? example "{title}"\n    ```python\n{indented_code}\n    ```'

            # We return the collapsed code block followed by the (possibly empty) output
            return f"{code_block}\n\n{output_body}"

        # Case B: Collapse Everything (Code + Output)
        # Note: This puts output INSIDE the box.
        # WARNING: This might break interactive PyVista plots if they are inside the indentation.
        if "collapse: all" in directives:
            combined = f"```python\n{code_body}\n```"
            if output_body:
                combined += f"\n\n{output_body}"

            indented_all = "\n".join([f"    {line}" for line in combined.splitlines()])
            return f'??? example "{title}"\n{indented_all}'

        # --- 3. DEFAULT (No Collapse) ---
        # If no collapse directive, we just return the clean code (without the magic comment)
        # and the output (if it wasn't hidden in Step 1).
        return f"```python\n{code_body}\n```\n\n{output_body}"
