import jax.numpy as jnp

def save_metrics_callback(metrics, step=0, prefix="debug"):
    """Save metrics to disk via callback"""
    filename = f"{prefix}_step_{step}"
    # jnp.save(filename, metrics)
    # print(f"Saved metrics to {filename}")
