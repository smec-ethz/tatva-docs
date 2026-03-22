<div class="nb-header"><a href="https://colab.research.google.com/github/smec-ethz/tatva-docs/blob/main/notebooks/compound.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><a href="/assets/notebooks/compound.ipynb" download="compound.ipynb" class="nb-download-btn"><svg class="nb-download-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 16l-6-6 1.41-1.41L11 13.17V4h2v9.17l3.59-3.58L18 11l-6 6z"/><path d="M5 18h14v2H5z"/></svg> Download notebook</a></div>

# Multifield variable 

<div class="nb-clear"></div>


In `tatva` we intend to define a total energy functional that takes a **flat** array $\mathbf{z}$ with all unknown DOFs and returns a scalar.
However, the actual unknowns in $\mathbf{z}$ are most often **nodal fields** with a specific shape, $e.g$ a displacement field `(n_nodes, 2)`.
Furthermore, the DOF array may include **multiple** fields. For example, a different nodal field like temperature `(n_nodes, 1)`, or a set of Lagrange-Multipliers `(n_constraints, n)`.

!!! note "Why pass a flat array $\mathbf{z}$ instead of the shaped fields to the energy function?"

    Passing a single flat array instead of the shaped fields has the advantage that the derivatives obtained through AD have convenient shapes.

    ```python
    residual_fn = jax.jacrev(energy)  # returns a vector of shape (n_dofs,)
    jacobian_fn = jax.jacfwd(residual_fn)  # returns a rank 2 tensor of shape (n_dofs, n_dofs)
    ```


Using `jax.numpy.array`, we can define such multiple field as:

```python
# unpack
u = z[: 2 * n_nodes].reshape(n_nodes, 2)
p = z[2 * n_nodes : 3 * n_nodes].reshape(n_nodes, 1)
```

Then we can manually pack them into a single variable as:

```python
# pack
z = jnp.hstack([u.flatten(), p.flatten()])
```


!!! warning "Problem"
    
    In such a approach we can usually end up writing repetitive pack/unpack code espescially when we ahve multiple DoFs. Furthermore, when we need to read (or update) values at specific locations in $\mathbf{z}$, we have to manually construct the correct indices:

    ```python
    # set u_y = 0 for a set of nodes defined by right_nodes
    z = z.at[right_nodes * 2 + 1].set(0.0)
    ```

    This quickly becomes hard to maintain when the number of fields grows. To ease this handling of multiple fields we provide utility [`Compound`](api/tatva.compound.md#compound)

## Getting started with  `Compound`

Declare a `Compound` subclass for your specific problem. Here, we define `Solution` for two fields, a displacement field with $u_x$ and $u_y$, and a pressure field $p$.


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
import jax.numpy as jnp
from jax import Array
from tatva.compound import Compound, field

n_nodes = 4


class Solution(Compound):
    u = field((n_nodes, 2))
    p = field((n_nodes, 1), default_factory=lambda: jnp.ones((n_nodes, 1)))
```

## Current state

Now, `Solution` strictly defines our problem and we use it to get a structured **view** of the flat array of unknowns.


```python
state0 = Solution()  # default initial state
z0 = state0.arr  # arr -> flat array of all fields = z
print(z0)
```

    [0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1.]


Given a current $\mathbf{z}$, simply give it to the constructor of `Solution`. Each descriptor (the fields) returns a shaped JAX array:


```python
# random example input
z = jnp.concatenate([jnp.arange(n_nodes * 2), jnp.ones(n_nodes)])

state = Solution(z)
state.u
```




    Array([[0., 1.],
           [2., 3.],
           [4., 5.],
           [6., 7.]], dtype=float32)



!!! note "Iterator unpacking"

    Compound classes support iterator unpacking. We can directly unpack the state into its fields like this:
        
    ```python
    u, p = Solution(z)
    ```
    
    This is the preferred way to work with `Solution` in an energy functional:
    ```python
    def total_energy(z: Array) -> Array:
        (u, p) = Solution(z)
        # compute energy from u and p
        E = ...
        return E
    ```

Assignments update the correct slice in `state.arr`.


```python
state.u = jnp.linspace(0, 1, 8).reshape(4, 2)
state.p = 2.0

print("flat array:", state.arr)

```

    flat array: [0.         0.14285715 0.2857143  0.42857146 0.5714286  0.71428573
     0.8571429  1.         2.         2.         2.         2.        ]


## Class-level DOF indexing

`Compound` classes and its `fields` also support indexing to obtain global DOF indices.

- `Solution.u[i]` gives all DOFs of `u` at node `i`
- `Solution.u[:, 0]` gives all DOFs of `u_x`
- `Solution.p[:]` gives all DOFs for `p`


```python
print("all dofs at node 1:", Solution[1])
print("x components of u", Solution.u[:, 0])
```

    all dofs at node 1: [2 3 9]
    x components of u [0 2 4 6]


!!! tip

    This is particularly useful to declare constrained DOFs:

    ```python
    constrained_dofs = jnp.concatenate([Solution.u[top_edge, 1], Solution.u[0, 0]])
    ```
    
    which returns the global indices for $u_y$ at `top_edge` and $u_x$ for node 0.

## Stack compatible fields

!!! note

    By default, fields are packed in declaration order. You can inspect the mapping from fields to slices directly on the class.
    In our example, the components for $u$ have indices $[0, 8]$, and the components of $p$ are at $[8, 12]$.


For better memory locality or convenience, fields can be stacked into one combined block at class definition.

!!! warning

    All stacked fields must share the same base shape on all non-stacked axes.



```python
from tatva.compound import stack_fields


@stack_fields("u", "p")
class StackedSolution(Compound):
    u = field(shape=(n_nodes, 2))
    p = field(shape=(n_nodes, 1))
    alpha = field(shape=(n_nodes, 1))


StackedSolution.u[:], StackedSolution.p[:], StackedSolution.alpha[:]
```




    (Array([ 0,  1,  3,  4,  6,  7,  9, 10], dtype=int32),
     Array([ 2,  5,  8, 11], dtype=int32),
     Array([12, 13, 14, 15], dtype=int32))



As you can see, the flat array is reordered such that $u$ and $p$ components for each node are sequential,
$i.e.$ $z = [u_{x,1}, u_{y,1}, p_{1}, u_{x,2}, u_{y,2}, p_2, ...]$.

## JAX compatibility

`Compound` is registered as a JAX pytree. This means it works with `jit`, `grad`, `vmap`, and friends.



```python
def energy_fn(s: Solution) -> jax.Array:
    return jnp.sum(s.u**2) + 0.1 * jnp.sum(s.p**2)


energy_jit = jax.jit(energy_fn)

value = energy_jit(state)
print("energy:", value)
```

    energy: 4.457143


## Summary

Use `Compound` when your solver expects a flat vector but your model is naturally expressed in multiple shaped fields.

- write energy/residual code with readable field names
- keep a single source of truth for global DOF layout
- stay fully compatible with JAX transformations

