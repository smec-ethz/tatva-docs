<div class="nb-header"><a href="https://colab.research.google.com/github/smec-ethz/tatva-docs/blob/main/notebooks/lifter.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><a href="/assets/notebooks/lifter.ipynb" download="lifter.ipynb" class="nb-download-btn"><svg class="nb-download-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 16l-6-6 1.41-1.41L11 13.17V4h2v9.17l3.59-3.58L18 11l-6 6z"/><path d="M5 18h14v2H5z"/></svg> Download</a></div>

# Applying Constraints or BCs

<div class="nb-clear"></div>



In constrained problems, we often solve only for **free** DOFs and reconstruct the full vector afterwards.
Typical constraints include:

- Dirichlet conditions (fixed values)
- Periodic constraints (slave DOFs copied from master DOFs)

Since `tatva` uses `jax.numpy.array`, we can manually manage index sets and repeat lifting logic across the codebase.


A manual implementation usually looks like this:

```python
u_full = u_full.at[free_dofs].set(u_reduced)
u_full = u_full.at[dirichlet_dofs].set(dirichlet_values)
u_full = u_full.at[periodic_dofs].set(u_full[periodic_master_dofs])
```


!!! warning "Problem"

    The manual implementation of index sets is easy to get wrong, especially when constraint sets evolve. Therefore, we provide a utility [`Lifter`](api/tatva.lifter.md#lifter) to ease the managing of indexes.


## Getting started with  `Lifter`

Use [`Lifter`](api/tatva.lifter.md#lifter) to define constraints once and map consistently between:

- reduced vectors (free DOFs only)
- full vectors (all DOFs)

A lifter takes a size, and then any number of `Constraints`. The library currently
includes two constraint types: `Fixed` and `Periodic`, but you are encouraged to add your own constraint types.


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
from tatva.lifter import Fixed, Lifter, Periodic

n_dofs = 8

dirichlet = Fixed(
    dofs=jnp.array([0, 7]),
    values=jnp.array([0.0, 0.0]),  # default is zeros, so this line is optional
)
periodic = Periodic(
    dofs=jnp.array([4, 6]),
    master_dofs=jnp.array([1, 3]),
)

lifter = Lifter(n_dofs, dirichlet, periodic)
```

## Lifting (free -> full)

Use [`lift_from_zeros`](api/tatva.lifter.md#tatva.lifter.Lifter.lift_from_zeros) when you want the full vector reconstructed from a zero base state.



```python
u_reduced = jnp.array([10.0, 20.0, 30.0, 40.0])
u_full = lifter.lift_from_zeros(u_reduced)

print("u_reduced:", u_reduced)
print("u_full:", u_full)

```

    u_reduced: [10. 20. 30. 40.]
    u_full: [ 0. 10. 20. 30. 10. 40. 30.  0.]


Use [`lift`](api/tatva.lifter.md#tatva.lifter.Lifter.lift) when you want to start from a previous iterate and only overwrite free DOFs before applying constraints.



```python
u_prev = jnp.linspace(-1.0, 1.0, n_dofs)
u_full_from_prev = lifter.lift(u_reduced, u_prev)

print("previous full state:", u_prev)
print("lifted full state:", u_full_from_prev)

```

    previous full state: [-1.         -0.71428573 -0.42857143 -0.14285709  0.1428572   0.42857146
      0.71428585  1.        ]
    lifted full state: [ 0. 10. 20. 30. 10. 40. 30.  0.]


!!! tip "Usage in total energy function"

    In the most general case, consider `lifter` as a static object (don't pass it as an argument to functions) and use it to lift the solution.

    ```python
    def total_energy(z_full: Array) -> Array:
        (u,) = Solution(z_full)  # See the guide for 'Compound'
        E = ...  # compute energy from u
        return E
        
    def total_energy_free(z_free: Array) -> Array:
        z_full = lifter.lift_from_zeros(z_free)
        return total_energy(z_full)
        
    residual_fn = jax.jacrev(total_energy_free)
    ```
    
    or do it inline:
    ```python
    residual_fn = jax.jacrev(lambda z_free: total_energy(lifter.lift_from_zeros(z_free)))
    ```

## Reducing (full -> free)

[`reduce`](api/tatva.lifter.md#tatva.lifter.Lifter.reduce) extracts the reduced vector from any full vector.



```python
u_reduced_back = lifter.reduce(u_full)
print("reduced from full:", u_reduced_back)

```

    reduced from full: [10. 20. 30. 40.]


!!! note

    The round-trip `reduce(lift_from_zeros(u_reduced))` returns `u_reduced`.


## Constraints

Some constraints are static (like the ones above).
Other constraints are dynamic, $e.g.$ a Dirichlet BC to apply a displacement controlled loading. 
To offer this, we have [`RuntimeValue`](api/tatva.lifter.md#tatva.lifter.RuntimeValue).
`Fixed` supports RuntimeValue out of the box. You can add RuntimeValue to your own constraints (see [Custom constraints](#custom-constraints))

Each `RuntimeValue` has a key (and optionally a default value). These values can be changed after with:
```python
lifter = lifter.at['top'].set(...)  # to change one
lifter = lifter.with_values({"top": ..., "other": ...})  # to set many at once
```


### Fixed

This one is to fix DOFs to some value. Often zero. The signature is `Fixed(dofs, values)`.
You can pass a `RuntimeValue` as a value to make it dynamic!


```python
from tatva.lifter import RuntimeValue

dirichlet = Fixed(jnp.array([0, 7]))
dirichlet_load = Fixed(jnp.array([5]), RuntimeValue("top"))  # we displace "top"

lifter = Lifter(n_dofs, dirichlet, dirichlet_load)
```

And then change it with:


```python
lifter = lifter.at["top"].set(0.1)  # Set the value of the "top" load to 0.1

lifter.lift_from_zeros(
    jnp.zeros(lifter.size_reduced)
)  # see that the dof 5 is set to 0.1 due to the dirichlet load
```




    Array([0. , 0. , 0. , 0. , 0. , 0.1, 0. , 0. ], dtype=float32)



### Periodic

This one is to define a periodic map between dofs. The signature is `Periodic(slave_dofs, master_dofs)`.


```python
periodic = Periodic(jnp.array([4, 6]), jnp.array([1, 3]))
lifter = Lifter(n_dofs, periodic)

u_free = jnp.zeros(lifter.size_reduced)
u_free = u_free.at[1].set(1.0).at[3].set(1.0)  # Set the master dofs to 1.0
lifter.lift_from_zeros(u_free)  # See that the periodic dofs are also set to 1.0
```




    Array([0., 1., 0., 1., 1., 0., 1., 0.], dtype=float32)



### Custom constraints

To create a custom constraint, inherit from `lifter.Constraint`.
In your constraint, you **need to** set the attribute `dofs`, and define the `apply_lift` method:

- the constraint must set `self.dofs` (=the constrained dofs)
- `apply_lift(self, u_full: Array) -> Array`

This method should set the constrained dofs in the solution array.

!!! note "Example: Rigid body MPC"

    This constraint will set the constrained dofs of the nodes in `disk_nodes` based on the disk center dofs.


```python
from tatva.mesh import Mesh
from tatva.compound import Compound, field
from tatva.lifter import Constraint, RuntimeValue
from jax import Array

mesh: Mesh
n_nodes = 10  # Example number of nodes in the mesh


class Solution(Compound):
    u = field(shape=(n_nodes, 2))  # Displacement field for all nodes
    u_disk = field(shape=(1, 3))  # (ux_c, uy_c, theta_c) for the disk


class RigidBody(Constraint):
    def __init__(
        self, disk_center: Array, disk_nodes: Array, ux_disk: float | RuntimeValue
    ):
        # We want to be able to set the disks center x-disp at runtime, so we use a RuntimeValue for that
        self.ux_disk = ux_disk

        self.disk_nodes = disk_nodes
        self.disk_center = disk_center
        self.dofs = jnp.concatenate(
            [
                Solution.u[disk_nodes, :].flatten(),
                # dirichlet bc for center x and y displacements:
                Solution.u_disk[0, :2].flatten(),
            ]
        )

    def apply_lift(self, u_full: Array) -> Array:
        """Apply the rigid body constraint by setting the displacements at the disk nodes
        based on the center displacements and rotation.
        """
        sol = Solution(u_full)
        (u, u_disk) = sol
        _, uy_c, theta_c = u_disk[0]
        # use _resolve_runtime to get the value of ux_c at runtime:
        ux_c = self._resolve_runtime(self.ux_disk)

        # Set the center x-displacement
        sol.u_disk = u_disk.at[0, 0].set(ux_c)

        # Compute the rigid body displacements for the disk nodes based on the center
        # displacements and rotation
        vecs = mesh.coords[self.disk_nodes] - self.disk_center
        cos, sin = jnp.cos(theta_c), jnp.sin(theta_c)
        rotation_matrix = jnp.array([[cos, -sin], [sin, cos]])
        rotated_vecs = vecs @ rotation_matrix.T
        u_rotation = rotated_vecs - vecs
        u_rigid = jnp.stack([ux_c, uy_c]) + u_rotation

        # Set the displacements at the disk nodes
        sol.u = u.at[self.disk_nodes, :].set(u_rigid)

        return sol.arr
```

As you can see, we have made `self.ux_disk` possibly a `RuntimeValue`. With this setup, you will be able to change the disk position directly from the lifter itself.


```python
lifter = Lifter(
    Solution.size,
    RigidBody(
        disk_center=jnp.array([0.5, 0.5]),
        disk_nodes=jnp.array([2, 3, 4]),
        ux_disk=RuntimeValue("ux_disk", default=0.0),
    ),
)

lifter = lifter.at["ux_disk"].set(0.1)  # Set the center x-displacement to 0.1
```

## JAX compatibility

`Lifter` can be used inside JAX-jitted functions.

You have three options:

- static `lifter` object -> not an argument to the function
    ```python
    @jax.jit
    def total_energy(u_free: jax.Array, u_top: float) -> jax.Array:
        lifter = lifter.at['top'].set(u_top)
        u_f = lifter.lift_from_zeros(u_free)
        return jnp.sum(u_f**2)
    ```
- static argument
    ```python
    @partial(jax.jit, static_argnames=["lifter"])
    def total_energy(u_free: jax.Array, u_top: float, lifter: Lifter) -> jax.Array:
        lifter = lifter.at['top'].set(u_top)
        u_f = lifter.lift_from_zeros(u_free)
        return jnp.sum(u_f**2)
    ```
- traced/dynamic argument
    ```python
    @jax.jit
    def total_energy(u_free: jax.Array, lifter: Lifter) -> jax.Array:
        u_f = lifter.lift_from_zeros(u_free)
        return jnp.sum(u_f**2)
    ```
    and call the function with an _updated_ lifter:
    ```python
    total_energy(u_free, lifter.at['top'].set(u_top))
    ```

!!! note

    If you pass a lifter as a dynamic argument to a jitted function, only the
    *runtime_values* will be traced. All other elements of the lifter, are handled as
    static values (aux_data of a custom pytree).


## Summary

Use `Lifter` when solving constrained systems in reduced coordinates.

- centralizes DOF bookkeeping for constraints
- provides explicit `lift` / `reduce` operations
- keeps reduced-space code clean and JAX-friendly

