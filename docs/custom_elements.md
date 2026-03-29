# Building Custom Elements


Tatva is designed to be extensible. While it includes a library of standard elements, the `element.Element` base class allows you to define any interpolation scheme or physics-specific element you need. By leveraging automatic differentiation, you only need to define local relationships; Tatva handles the global assembly, sparsity, and differentiation.


## The `element.Element` Architecture

Every custom element in Tatva must implement or inherit three core components that define how fields (like displacement or temperature) move from nodes to integration points:

1.  **Quadrature**: `quad_points` and `quad_weights` define where the element is sampled.
2.  **Interpolation**: The logic that transforms discrete nodal values into a continuous field.
3.  **Jacobian/Mapping**: The relationship between the reference geometry ($\xi$) and physical space ($x$).



## Two Patterns of Interpolation

In Tatva, elements generally follow one of two patterns depending on whether the interpolation depends solely on the natural coordinates or requires physical geometric data.

### Standard Shape Function Interpolation
Standard elements (like **Tri6** or **Line2In3D**) use predefined shape functions $N(\xi)$. These functions depend only on the natural coordinate and the nodal values.

* **How it works**: The field at any point is $u(\xi) = \sum N_i(\xi) \cdot u_i$.
* **Gradients**: The gradient is computed by dividing the natural derivative by the Jacobian determinant.
* **Example**: A [`Line2In3D`](examples/timoshenko_2d_beam.ipynb) element uses simple linear shape functions to interpolate values along a line in space.

### Coordinate-Aware & Local Basis Interpolation
More complex elements, such as the **Hermite Beam**, require knowledge of the physical geometry (like the element length $L$ or a local rotation matrix) *before* they can interpolate.

* **How it works**: These elements override the `interpolate` and `gradient` methods. They use `nodal_coords` to compute local parameters (like a beam's length) to transform global DOFs into a local frame.
* **Automatic Differentiation**: Instead of manual shape function derivatives, these elements often use `jax.grad` to compute slopes and curvatures directly from the interpolation logic.
* **Example**: A [`HermiteBeamElement2D`](examples/euler_bernoulli_2d_beam.ipynb) uses the distance between `nodal_coords` to define cubic basis functions for $C^1$ continuity.


## Implementation Template

Below is a generic template for a custom element. You can either provide standard shape functions or completely override the interpolation logic for complex physics.

```python
from tatva import element
import jax.numpy as jnp
import equinox as eqx

class MyCustomElement(element.Element):
    # Define Quadrature (e.g., 1-point Gauss)
    def _default_quadrature(self):
        quad_points: jnp.ndarray = jnp.array([0.0])
        quad_weights: jnp.ndarray = jnp.array([2.0])
        return quad_points, quad_weights
 
    # Standard Shape Functions
    def shape_function(self, xi):
        # Linear: [N1, N2]
        return jnp.array([0.5 * (1 - xi), 0.5 * (1 + xi)])

    def shape_function_derivative(self, xi):
        # [dN1/dxi, dN2/dxi]
        return jnp.array([-0.5, 0.5])

    # Custom Interpolation Override 
    def interpolate(self, xi, nodal_values, nodal_coords):
        # Example: Using nodal_coords to find physical length
        L = jnp.linalg.norm(nodal_coords[1] - nodal_coords[0])
        
        # Define custom logic or call an AD-compatible interpolator
        N = self.shape_function(xi)
        return N @ nodal_values
```