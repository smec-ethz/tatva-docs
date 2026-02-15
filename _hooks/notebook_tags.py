import logging
import re

log = logging.getLogger("mkdocs")


def on_page_markdown(markdown, page, config, files):
    # 1. Only run on notebooks
    if not page.file.src_uri.endswith(".ipynb"):
        return markdown

    # 2. Define the Regex Pattern
    # Matches: "# tags: [tag1, tag2]" at the start of a line
    # Group 1 captures the content inside the brackets: "tag1, tag2"
    tag_pattern = r"^#\s*tags:\s*\[(.*?)\]"

    # 3. Search for the pattern
    match = re.search(tag_pattern, markdown, flags=re.MULTILINE | re.IGNORECASE)

    if match:
        # Extract the raw string of tags (e.g., "fracture, 3d, tutorial")
        raw_tags = match.group(1)

        # Split by comma and clean up whitespace
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

        if tags:
            # 4. Generate the HTML Badges
            tags_html = '<div class="nb-tags-container">'
            for tag in tags:
                tags_html += f'<span class="nb-tag">{tag}</span>'
            tags_html += "</div>\n\n"

            # 5. Remove the original "# tags: [...]" line from the markdown
            # We replace the entire matched line with an empty string
            markdown = markdown.replace(match.group(0), "", 1)

            # 6. Prepend the tags HTML to the top of the file
            return tags_html + markdown

    # If no tags found, return original markdown
    return markdown
