import jax
import jax.numpy as jnp
import pytest
from .rope_simple import apply_rope


class TestApplyRope:
    def test_basic_functionality(self):
        """Test basic rope application with simple inputs."""
        batch_size, seq_len, head_dim = 2, 4, 8

        # Create test input
        input_tensor = jnp.ones((batch_size, seq_len, head_dim))
        positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

        # Apply rope
        result = apply_rope(input_tensor, positions)

        # Check output shape
        assert result.shape == input_tensor.shape

        # Check that result is not identical to input (rope should modify values)
        assert not jnp.allclose(result, input_tensor)

    def test_different_head_dimensions(self):
        """Test rope with different head dimensions."""
        for head_dim in [4, 8, 16, 32, 64]:
            batch_size, seq_len = 1, 3

            input_tensor = jnp.ones((batch_size, seq_len, head_dim))
            positions = jnp.arange(seq_len)[None, :]

            result = apply_rope(input_tensor, positions)
            assert result.shape == (batch_size, seq_len, head_dim)

    def test_custom_base(self):
        """Test rope with custom base parameter."""
        batch_size, seq_len, head_dim = 1, 2, 4

        input_tensor = jnp.ones((batch_size, seq_len, head_dim))
        positions = jnp.arange(seq_len)[None, :]

        # Test with different bases
        result_default = apply_rope(input_tensor, positions, base=10000)
        result_custom = apply_rope(input_tensor, positions, base=5000)

        # Results should be different with different bases
        assert not jnp.allclose(result_default, result_custom)

    def test_zero_positions(self):
        """Test rope with zero positions."""
        batch_size, seq_len, head_dim = 1, 3, 4

        input_tensor = jnp.array(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]
        )
        positions = jnp.zeros((batch_size, seq_len))

        result = apply_rope(input_tensor, positions)

        print(result)

        # With zero positions, rope should return the original input
        assert jnp.allclose(result, input_tensor)

    def test_position_independence(self):
        """Test that rope is applied independently to each position."""
        batch_size, seq_len, head_dim = 1, 2, 4

        input_tensor = jnp.ones((batch_size, seq_len, head_dim))

        # Apply rope to individual positions
        pos_0 = jnp.array([[0, 100]])  # Different positions
        pos_1 = jnp.array([[0, 101]])

        result_0 = apply_rope(input_tensor, pos_0)
        result_1 = apply_rope(input_tensor, pos_1)

        # First position should be the same, second should be different
        assert jnp.allclose(result_0[0, 0], result_1[0, 0])
        assert not jnp.allclose(result_0[0, 1], result_1[0, 1])

    def test_batch_processing(self):
        """Test that rope works correctly with batched inputs."""
        batch_size, seq_len, head_dim = 3, 2, 4

        # Create different inputs for each batch
        input_tensor = (
            jnp.arange(batch_size * seq_len * head_dim)
            .reshape(batch_size, seq_len, head_dim)
            .astype(jnp.float32)
        )

        # Same positions for all batches
        positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

        result = apply_rope(input_tensor, positions)

        assert result.shape == input_tensor.shape

        # Check that each batch is processed independently
        # by comparing with single batch processing
        for i in range(batch_size):
            single_batch_input = input_tensor[i : i + 1]
            single_batch_pos = positions[i : i + 1]
            single_result = apply_rope(single_batch_input, single_batch_pos)

            assert jnp.allclose(result[i : i + 1], single_result)

    def test_odd_head_dim_error(self):
        """Test that odd head dimensions raise appropriate errors."""
        batch_size, seq_len = 1, 2
        # Test with odd head dimension
        head_dim = 3

        input_tensor = jnp.ones((batch_size, seq_len, head_dim))
        positions = jnp.arange(seq_len)[None, :]

        with pytest.raises(AssertionError):
            result = apply_rope(input_tensor, positions)
