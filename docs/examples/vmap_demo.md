---
tags:
  - demo
---

<div class="nb-header"><a href="https://colab.research.google.com/github/smec-ethz/tatva-docs/blob/main/notebooks/examples/vmap_demo.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><a href="/assets/notebooks/examples/vmap_demo.ipynb" download="vmap_demo.ipynb" class="nb-download-btn"><svg class="nb-download-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 16l-6-6 1.41-1.41L11 13.17V4h2v9.17l3.59-3.58L18 11l-6 6z"/><path d="M5 18h14v2H5z"/></svg> Serial</a><a href="/assets/notebooks/examples/vmap_demo_parallel.ipynb" download="vmap_demo_parallel.ipynb" class="nb-download-btn"><svg class="nb-download-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 16l-6-6 1.41-1.41L11 13.17V4h2v9.17l3.59-3.58L18 11l-6 6z"/><path d="M5 18h14v2H5z"/></svg> Parallel</a></div>

# Serial vs Parallel: vmap Demo

This example computes per-element operations on a large array. The serial version uses Python loops; the parallel version uses `jax.vmap`.


```
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
```

## Data setup


```
key = jax.random.PRNGKey(0)
x = jax.random.uniform(key, shape=(1024,), minval=0.0, maxval=1.0)
y = jax.random.uniform(key, shape=(1024,), minval=0.0, maxval=1.0)
```

## Forward pass


```
def forward(xi, yi):
    return jnp.sqrt(xi) + jnp.sin(yi)
```

=== "Serial"

    ```python
    result = jnp.array([forward(xi, yi) for xi, yi in zip(x, y)])
    print(f"result[:4] = {result[:4]}")
    ```

=== "Parallel"

    !!! note "Parallel"
    
        ```python
        forward_vmap = jax.vmap(forward)
        ```

    ```python
    result = forward_vmap(x, y)
    print(f"result[:4] = {result[:4]}")
    ```

## Backward pass


```
def backward(xi, yi):
    return jax.grad(lambda a: forward(a, yi))(xi)
```

=== "Serial"

    ```python
    grads = jnp.array([backward(xi, yi) for xi, yi in zip(x, y)])
    print(f"grads[:4] = {grads[:4]}")
    ```

=== "Parallel"

    !!! note "Parallel"
    
        ```python
        backward_vmap = jax.vmap(backward)
        ```

    ```python
    grads = backward_vmap(x, y)
    print(f"grads[:4] = {grads[:4]}")
    ```

## Summary

Both forward and backward passes produce arrays of the same shape as the input.
