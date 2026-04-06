<div class="nb-header"><a href="https://colab.research.google.com/github/smec-ethz/tatva-docs/blob/main/notebooks/sparse.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><a href="/assets/notebooks/sparse.ipynb" download="sparse.ipynb" class="nb-download-btn"><svg class="nb-download-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 16l-6-6 1.41-1.41L11 13.17V4h2v9.17l3.59-3.58L18 11l-6 6z"/><path d="M5 18h14v2H5z"/></svg> Download</a></div>

# Building Stiffness Matrix



In finite element analysis, the tangent stiffness matrix (Hessian) $\mathbf{K}$ is typically **sparse**. A node only interacts with its immediate neighbors, meaning most entries in $\mathbf{K}$ are zero. However, standard automatic differentiation (AD) in JAX (`jax.jacfwd` or `jax.jacrev`) is unaware of this sparsity. It attempts to recover the full dense matrix by evaluating the Jacobian-Vector Product (JVP) once for every degree of freedom.

For a mesh with $N$ degrees of freedom:

* **Naive AD Cost:** $N \times t_{\text{residual}}$ (Prohibitive for large $N$)
* **Memory:** $O(N^2)$ (Explodes quickly)

We need a way to compute **only the non-zero entries** of $\mathbf{K}$ efficiently, ideally in constant time with respect to the mesh size.

!!! note "Two ways to build **K** "

    `tatva` provides 2 ways to build a computationally efficient stiffness matrix using the energy functional.

    - **Matrix-free operator** using Jacobian-vector product.
    - **Sparse stiffness matrix** using sparse differentiation.


Below we demonstrate the two approaches, we use a pseudo energy functional that is defined for a given mesh.



??? example "Colab Setup (Install Dependencies)"
    ```python
    
    # Only run this if we are in Google Colab
    if "google.colab" in str(get_ipython()):
        print("Installing dependencies from pyproject.toml...")
        # This installs the repo itself (and its dependencies)
        !apt-get install gmsh 
        !apt-get install -qq xvfb libgl1-mesa-glx
        !pip install pyvista -qq
        !pip install -q "git+https://github.com/smec-ethz/tatva-docs.git"    
        print("Installation complete!")
    ```




```python
import jax

jax.config.update("jax_enable_x64", True)  # use double-precision

import jax.numpy as jnp
import scipy.sparse as sp
from tatva import sparse, Mesh

mesh = Mesh.unit_square(n_x=1, n_y=1, type="triangle", dim=2)
n_dofs_per_node = 2
n_dofs = mesh.coords.shape[0] * n_dofs_per_node


def energy_fn(u, delta):
    # Placeholder for the actual energy function
    return jnp.sum(u**2) + delta * jnp.sum(u)


delta_current = 1.0  # Example parameter

```

## Matrix-free Operator

Since we have the energy functional, we can compute the Jacobian-vector product (JVP) $\mathbf{K}\mathbf{v}$ directly without ever forming $\mathbf{K}$. We can use [`jax.jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html) to compute the Jacobian-vector product.




```python
def jacobian_vector_product(u, v, delta):
    """
    Computes (Hessian of Energy at u) * v. It is equivalent to: jvp( jacrev(energy)(u), v ).
    Args:
        u: Current solution (shape: [n_dofs])
        v: Vector to multiply with the Hessian (shape: [n_dofs])
        delta: Additional parameter for the energy function
    Returns:
        The product of the Hessian of the energy function at u with the vector v (shape: [n_dofs])
    """
    return jax.jvp(jax.jacrev(energy_fn), (u, delta), (v, delta))[1]


delta_f = jacobian_vector_product(
    u=jnp.zeros(n_dofs), v=jnp.ones(n_dofs), delta=delta_current
)

# or can be passed directly to iterative solvers like jax.scipy.sparse.linalg.cg
```

Some of the examples that use **Matrix-Free Operators** are

- [Linear elasticity](examples/linear_elasticity.md)
- [Contact between deformable bodies](examples/contact_3d.md)
- [Fracture using Cohesive Traction Law](examples/fracture_quasistatic_3d.md)
- [Multiphysical Fracture using Phasefield](examples/thermal_shock_fracture.imd)
- [Linear Elasticity using Matrix-Free PETSc](external_solvers/linear_elasticity_petsc4py_mf.md)

## Sparse Differentiation

[`tatva.sparse`](api/tatva.sparse.md#sparse) provides a sparse differentiation engine that reduces the cost from $O(N)$ to $O(c)$, where $c$ is the "chromatic number" of the mesh (typically small and constant, e.g., ~10-20 for 2D meshes).

The process has three steps:

-  **Sparsity Pattern:** Identify the non-zero structure.
-  **Coloring:** Group non-interacting DOFs.
-  **Differentiation:** Compute the matrix in batches.



### Sparsity Pattern

First, we analyze the mesh connectivity to determine which DOFs interact. [`create_sparsity_pattern`](api/tatva.sparse.md#tatva.sparse.create_sparsity_pattern) returns the indices of the non-zero entries.



```python
from tatva.sparse import create_sparsity_pattern

sparsity_pattern = create_sparsity_pattern(mesh, n_dofs_per_node=n_dofs_per_node)

print(f"Sparsity: {sparsity_pattern.nnz} non-zeros")
```

    Sparsity: 56 non-zeros


!!! info "Sparsity Pattern"

    In `tatva` we provide a few functionalities to generate sparsity patetrn for some specific problem. 
    
    - for a single physical field problem, [`sparse.create_sparsity_pattern`](api/tatva.sparse.md#tatva.sparse.create_sparsity_pattern)
    - for KKT problems, [`sparse.create_sparsity_pattern_KKT`](api/tatva.sparse.md#tatva.sparse.create_sparsity_pattern_KKT)
    - for reduced system via condensation  [`sparse.reduce_sparsity_pattern`](api/tatva.sparse.md#tatva.sparse.reduce_sparsity_pattern)
    - for periodic problem [`sparse.create_sparsity_pattern_master_slave`](api/tatva.sparse.md#tatva.sparse.create_sparsity_pattern_master_slave)

### Graph Coloring

We partition the degrees of freedom into independent sets (colors). Two DOFs share the same color **only if** they do not share an edge in the sparsity graph (Distance-1) and do not share a common neighbor (Distance-2). This ensures that when we perturb all DOFs of "Color A" simultaneously, their contributions to the Hessian do not overlap.

The `sparse` module offers a type named [`ColoredMatrix`](api/tatva.sparse.md#sparse.ColoredMatrix), which is a CSR style sparse matrix with colors. 
To generate the `ColoredMatrix` from a scipy csr matrix, $i.e.$ the `sparsity_pattern` created before, use `ColoredMatrix.from_csr(...)`.
If you don't provide colors, they are computed automatically from the given sparse matrix.

!!! info "Coloring Algorithm"
    
    We use the `tatva_coloring` library to generate colors from a sparsity pattern. The implemented coloring algorithm there is a naive greedy-algorithm. However, one can easily use other coloring libraries and provide the colors in `ColoredMatrix.from_csr(..., colors=colors)`. For example, [pysparsematrixcolorings](https://github.com/gdalle/pysparsematrixcolorings) which offers more advanced coloring algorithms which are more efficient because they lead to less colors.



```python
from tatva.sparse import ColoredMatrix

colored_matrix = ColoredMatrix.from_csr(sparsity_pattern)

print(f"Number of colors required: {jnp.max(colored_matrix.colors) + 1}")
```

    Number of colors required: 8



### Sparse Differentiation

Finally, we use [`sparse.jacfwd`](api/tatva.sparse.md#tatva.sparse.jacfwd), which
differentiates a functional into the sparse structure defined by `ColoredMatrix`. It
returns a function that returns a new instance of type `ColoredMatrix`. This function
automatically:

-  Perturbs the input $\boldsymbol{u}$ using the color groups.
-  Evaluates the gradient efficiently (Batched JVPs).
-  Reconstructs the values into the correct sparse matrix locations.


!!! Info

    We use our own implementation of sparse differentiation to make it scalable for large problems. But one can use libraries such as [sparsejac](https://github.com/mfschubert/sparsejac) which has been an source of inspiration for our own implementation.


```python
from tatva.sparse import jacfwd

gradient_fn = jax.jacrev(energy_fn, argnums=0)  # Gradient with respect to u

# differentiate the residual using the ColoredMatrix
K_sparse_fn = jacfwd(
    gradient_fn,
    colored_matrix,
    color_batch_size=10,  # Batch size for evaluating the element routine
)

u_current = jnp.zeros(n_dofs)  # Example input
K_sparse = K_sparse_fn(u_current, delta_current)  # K_sparse is of type ColoredMatrix

K_sparse.to_dense()
```




    Array([[2., 0., 0., 0., 0., 0., 0., 0.],
           [0., 2., 0., 0., 0., 0., 0., 0.],
           [0., 0., 2., 0., 0., 0., 0., 0.],
           [0., 0., 0., 2., 0., 0., 0., 0.],
           [0., 0., 0., 0., 2., 0., 0., 0.],
           [0., 0., 0., 0., 0., 2., 0., 0.],
           [0., 0., 0., 0., 0., 0., 2., 0.],
           [0., 0., 0., 0., 0., 0., 0., 2.]], dtype=float64)



!!! Info

    The above function `K_sparse_fn` needs to be created only once given the sparsity pattern is not changing. Once created one can use the generated function within the simulation loop. 
    If the energy function takes additional arguments for example history parameters then the generated `K_sparse_fn` also takes the same arguments. 


### Computing **K** at fixed additional parameters 

Sometimes we need to evaluate $\mathbf{K}$ at fixed values for additional arguments but updated values of $\boldsymbol{u}$. For example, in Newton-Raphson or Staggered solvers. Use `partial` from `functools` to freeze some argument values. For example

```python
from functools import partial

K_sparse_partial = jax.jit(partial(K_sparse_fn, delta=delta_current))

# newton iteration
for i in range(iter):
    ...
    # call K_sparse_partial with just u
    K_sparse = K_sparse_partial(u_new)
    ...

``` 

Some of the examples that use **sparse differentiation** are:

- [Hyperelastic bar in 3D](examples/hyperelastic_3d.md)
- [Periodic Boundary Conditions using Lagrange Multiplier](examples/homogenization_periodic.md)
- [Neural Constitutive Law](examples/ncm_metamaterial.md)
- [Neural Operator Element method](examples/neural_operator_method_3d.md)
- [Linear Elasticity using PETSc](external_solvers/linear_elasticity_petsc4py.md)


!!! tip

    For extremely large problems, even storing the sparse matrix indices might be too memory-intensive. Then one should use **Matrix-free** approach.
    Also, for the problems where finding the sparsity pattern is difficult.


