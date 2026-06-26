import os
import re
import urllib.request
import urllib.error
import json
from pathlib import Path

# Repo to fetch releases from
REPO = "smec-ethz/tatva"
OUTPUT_FILE = Path(__file__).parent / "docs" / "changelog.md"

def fetch_releases():
    url = f"https://api.github.com/repos/{REPO}/releases"
    req = urllib.request.Request(url)
    
    # Use GitHub token if available (e.g. in GitHub Actions)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"token {token}")
    
    req.add_header("User-Agent", "tatva-docs-builder")
    
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"Error fetching releases from GitHub: {e}")
        if e.code == 403:
            print("API rate limit exceeded or forbidden. Please set GITHUB_TOKEN environment variable.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def dedent_lines(lines):
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return lines
    
    # Calculate leading spaces for each non-empty line
    indents = []
    for line in non_empty_lines:
        indent = 0
        for char in line:
            if char == ' ':
                indent += 1
            else:
                break
        indents.append(indent)
        
    min_indent = min(indents)
    if min_indent > 0:
        # Dedent each line by min_indent spaces
        dedented = []
        for line in lines:
            if line.startswith(' ' * min_indent):
                dedented.append(line[min_indent:])
            else:
                dedented.append(line.lstrip(' '))
        return dedented
    return lines

def main():
    print("Fetching release notes from GitHub...")
    releases = fetch_releases()
    
    if not releases:
        # If fetching fails, we can either keep existing changelog or write a placeholder
        if OUTPUT_FILE.exists():
            print("Using existing changelog.md due to fetch error.")
            return
        releases = []

    markdown_content = []
    markdown_content.append("---")
    markdown_content.append("title: Changelog")
    markdown_content.append("hide:")
    markdown_content.append("  - navigation")
    markdown_content.append("---")
    markdown_content.append("")
    markdown_content.append("# Changelog")
    markdown_content.append("")
    markdown_content.append("All releases and changes for the `tatva` library, pulled directly from GitHub.")
    markdown_content.append("")
    
    if not releases:
        markdown_content.append("No releases found or unable to fetch release notes from GitHub.")
    else:
        for release in releases:
            name = (release.get("name") or release.get("tag_name")).lstrip('v')
            tag = release.get("tag_name")
            date = release.get("published_at", "")[:10]  # Get YYYY-MM-DD
            body = release.get("body", "")
            html_url = release.get("html_url")
            
            markdown_content.append(f"## [{name}]({html_url}) ({date})")
            markdown_content.append("")
            
            # Format markdown body (ensure proper header levels for material theme)
            # Remove duplicate version headers and demote other headers by 1 level
            version_clean = tag.lstrip('v')
            version_header_pattern = re.compile(rf"^\s*#+\s+\[?v?{re.escape(version_clean)}\]?")
            
            lines = body.splitlines()
            formatted_lines = []
            in_code_block = False
            code_block_lines = []
            code_block_marker = ""
            for line in lines:
                if re.match(r"^\s*```", line):
                    if not in_code_block:
                        in_code_block = True
                        code_block_marker = line.strip()
                        code_block_lines = []
                    else:
                        in_code_block = False
                        dedented = dedent_lines(code_block_lines)
                        formatted_lines.append(code_block_marker)
                        for cb_line in dedented:
                            formatted_lines.append(cb_line.rstrip())
                        formatted_lines.append(line.strip())
                    continue
                    
                if in_code_block:
                    code_block_lines.append(line)
                    continue

                header_match = re.match(r"^\s*(#+)\s+(.*)$", line)
                if header_match:
                    hashes = header_match.group(1)
                    content = header_match.group(2)
                    
                    # If it's a version header, skip it to avoid duplicate headings
                    if version_header_pattern.match(line):
                        continue
                        
                    # Force H4 (####) for standard section headings, otherwise demote by 1 level
                    content_clean = content.strip().lower().replace("’", "'")
                    forced_h4_keywords = [
                        "what's changed",
                        "contributors",
                        "breaking changes",
                        "features",
                        "bug fixes",
                        "performance improvements",
                        "refactoring",
                        "documentation",
                        "dependency updates",
                        "dependencies",
                        "chore",
                    ]
                    if any(kw in content_clean for kw in forced_h4_keywords):
                        new_level = 4
                    else:
                        new_level = min(len(hashes) + 1, 6)
                    formatted_lines.append("#" * new_level + " " + content)
                else:
                    formatted_lines.append(line)
            
            markdown_content.append("\n".join(formatted_lines))
            markdown_content.append("\n")
            
    OUTPUT_FILE.write_text("\n".join(markdown_content), encoding="utf-8")
    print(f"Changelog successfully generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
