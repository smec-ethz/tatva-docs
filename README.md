# tatva-docs

Documentation source for the [tatva](https://github.com/smec-ethz/tatva) library.

Built with [Zensical](https://zensical.com) from Markdown sources. Notebooks are authored as `.ipynb` files and converted to Markdown offline before building.

## Repository layout

```
notebooks/          # Jupyter notebooks (source of truth — edit these)
  examples/
  external_solvers/
  operator.ipynb
  ...
docs/               # Zensical docs_dir (do not edit generated files)
  examples/
    linear_elasticity.md          # generated from notebook
    linear_elasticity_files/      # extracted matplotlib images
  assets/
    notebooks/                    # .ipynb copies served for download
  index.md                        # hand-written
  getting_started.md              # hand-written
  ...
mkdocs.yml          # site configuration (nav, theme, plugins)
convert_notebooks.py  # offline notebook → markdown conversion script
pyproject.toml
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/smec-ethz/tatva-docs.git
cd tatva-docs
uv sync
```

## Building the site locally

```bash
uv run zensical serve
```

The site is served at `http://localhost:8000` with live reload.

## Working with notebooks

Notebooks live in `notebooks/` and are **not** executed by the build. After editing or re-running a notebook, convert it to Markdown manually:

```bash
# Convert all notebooks
uv run convert_notebooks.py

# Preview what will be converted without doing anything
uv run convert_notebooks.py --check
```

This will:
1. Run `nbconvert` on each `.ipynb` in `notebooks/`, writing `.md` + image files into the matching path under `docs/`
2. Copy each `.ipynb` to `docs/assets/notebooks/` so it can be downloaded from the site
3. Inject the Colab badge, download button, and any collapse/tag directives into the Markdown

**Commit both the notebook and the generated files:**

```bash
git add notebooks/examples/my_example.ipynb
git add docs/examples/my_example.md
git add docs/examples/my_example_files/
git add docs/assets/notebooks/examples/my_example.ipynb
git commit -m "feat: add my_example notebook"
```

## Adding a new notebook

1. Create the notebook under `notebooks/`, mirroring the `docs/` subdirectory you want it to appear in:
   ```
   notebooks/examples/my_example.ipynb
   ```

2. Add an entry to the `nav` in `mkdocs.yml` pointing to the **`.md`** path:
   ```yaml
   nav:
     - Examples:
         - My Example: "examples/my_example.md"
   ```

3. Run the converter and commit:
   ```bash
   uv run convert_notebooks.py
   git add notebooks/examples/my_example.ipynb \
           docs/examples/my_example.md \
           docs/examples/my_example_files/ \
           docs/assets/notebooks/examples/my_example.ipynb
   git commit -m "feat: add my_example"
   ```

## Notebook authoring conventions

### Collapsible code cells

Add a directive comment as the first line of a cell to control how it renders on the site:

```python
# [collapse: code] Setup
import matplotlib.pyplot as plt
...
```

| Directive | Effect |
|---|---|
| `# [collapse: code] Title` | Code block collapsed, output visible |
| `# [collapse: all] Title` | Code and output both collapsed |
| `# [output: hide] Title` | Code visible, output collapsed |

### Page tags

Add a `# tags: [...]` comment anywhere in a code cell to render tag badges below the page title:

```python
# tags: [elasticity, tutorial, 3d]
```

### Colab badge and download button

These are injected automatically during conversion — no action needed.

To control which GitHub branch the Colab link points to, set the `LIB_TAG` environment variable before converting:

```bash
LIB_TAG=v0.7.1 uv run convert_notebooks.py
```

Defaults to `main`.

## CI

The CI pipeline only builds the site — it does not execute notebooks or run `convert_notebooks.py`. Generated `.md` files must be committed to the repository.
