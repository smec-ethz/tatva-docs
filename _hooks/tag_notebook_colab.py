import os
import re


def on_page_markdown(markdown, page, config, files):
    if not page.file.src_uri.endswith(".ipynb"):
        return markdown

    # --- 1. COLAB LINK LOGIC ---
    tag = os.environ.get("LIB_TAG", "main")
    clean_path = page.file.src_uri.lstrip("/")
    base_url = f"https://colab.research.google.com/github/smec-ethz/tatva-docs/blob/{tag}/docs/"
    colab_url = f"{base_url}{clean_path}"

    badge_html = f'<a href="{colab_url}" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>\n\n'
    markdown = badge_html + markdown

    # --- 2. ROBUST REGEX ---
    # This regex looks for the Python block and optionally the content following it.
    # We use a non-greedy match for the output (.*?) until we hit a double newline or next code block.
    pattern = r"```python\s+# \[(.*?)\]\s*(.*?)\n(.*?)\n```(\n.*?)(?=\n```python|\Z)"

    def transform_cell(match):
        directives = match.group(1).lower()
        title = match.group(2).strip() or "Code"
        code_body = match.group(3)
        raw_output = match.group(4) or ""

        # Clean up the output (remove leading newlines for consistent indentation)
        output_body = raw_output.strip()

        # Helper to indent text block for admonitions
        def indent(text):
            return "\n".join([f"    {line}" for line in text.splitlines()])

        # --- A. PREPARE THE CODE BLOCK ---
        final_code_block = ""

        if "collapse: code" in directives or "collapse: all" in directives:
            # Wrap code in 'example' (Purple)
            indented_code = indent(code_body)
            final_code_block = (
                f'??? example "{title}"\n    ```python\n{indented_code}\n    ```'
            )
        else:
            # Keep code standard
            final_code_block = f"```python\n{code_body}\n```"

        # --- B. PREPARE THE OUTPUT BLOCK ---
        final_output_block = ""

        if output_body:
            if "output: hide" in directives or "collapse: all" in directives:
                # Wrap output in 'quote' (Gray/Neutral) to distinguish from code
                # You can change 'quote' to 'note' (Blue) or 'warning' (Orange)
                indented_output = indent(output_body)
                final_output_block = f'??? info "Output"\n{indented_output}'
            else:
                # Keep output visible and standard
                final_output_block = output_body

        # --- C. RETURN WITH SAFETY PADDING ---
        # The \n\n at the end is crucial to prevent the "next cell merge" bug
        return f"{final_code_block}\n\n{final_output_block}\n\n"

    # Use re.DOTALL so (.) matches newlines
    markdown = re.sub(pattern, transform_cell, markdown, flags=re.DOTALL)

    return markdown
