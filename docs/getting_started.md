---
hide:
  - navigation
  - toc
---

# Getting Started


## Installation

Install the current release from PyPI:

```bash
pip install tatva
```

For development work, clone the repository and install it in editable mode (use your preferred virtual environment tool such as `uv` or `venv`):

```bash
git clone https://github.com/smec-ethz/tatva.git
cd tatva
pip install -e .
```

## Quick Example

Create a mesh, pick an element type, and let Operator perform the heavy lifting with JAX arrays:

```bash
import jax.numpy as jnp
from tatva.element import Tri3
from tatva.mesh import Mesh
from tatva.operator import Operator

coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
elements = jnp.array([[0, 1, 2], [0, 2, 3]])

mesh = Mesh(coords, elements)

op = Operator(mesh, Tri3())
nodal_values = jnp.arange(coords.shape[0], dtype=jnp.float64)

# Integrate a nodal field over the mesh
total = op.integrate(nodal_values)

# Evaluate gradients at all quadrature points
gradients = op.grad(nodal_values)
```

## License

`tatva` is distributed under the GNU Lesser General Public License v3.0 or later. See `COPYING` and `COPYING.LESSER` for the complete terms. © 2025 ETH Zurich (SMEC).
