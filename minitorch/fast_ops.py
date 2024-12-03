from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            out_storage, out_shape, out_strides = out.tuple()
            f(out_storage, tuple(out_shape), out_strides, *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            out_storage, out_shape, out_strides = out.tuple()
            f(out_storage, tuple(out_shape), out_strides, *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            out_storage, out_shape, out_strides = out.tuple()
            f(out_storage, tuple(out_shape), out_strides, *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        out_storage, out_shape, out_strides = out.tuple()
        tensor_matrix_multiply(
            out_storage, tuple(out_shape), out_strides, *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        for out_idx in np.ndindex(out_shape):
            in_idx = [0 for _, _ in enumerate(in_shape)]
            broadcast_index(
                big_index=out_idx,
                big_shape=out_shape,
                shape=in_shape,
                out_index=in_idx,
            )
            out_pos = index_to_position(index=out_idx, strides=out_strides)
            in_pos = index_to_position(index=in_idx, strides=in_strides)
            out[out_pos] = fn(in_storage[in_pos])

    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        for out_idx in np.ndindex(out_shape):
            a_idx = [0 for _, _ in enumerate(a_shape)]
            b_idx = [0 for _, _ in enumerate(b_shape)]
            broadcast_index(
                big_index=out_idx,
                big_shape=out_shape,
                shape=a_shape,
                out_index=a_idx,
            )
            broadcast_index(
                big_index=out_idx,
                big_shape=out_shape,
                shape=b_shape,
                out_index=b_idx,
            )
            out_pos = index_to_position(index=out_idx, strides=out_strides)
            a_pos = index_to_position(index=a_idx, strides=a_strides)
            b_pos = index_to_position(index=b_idx, strides=b_strides)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        for out_idx in np.ndindex(out_shape):
            out_pos = index_to_position(index=out_idx, strides=out_strides)
            a_idx = list(out_idx)
            cache = out[out_pos]
            for i in range(a_shape[reduce_dim]):
                a_idx[reduce_dim] = i
                a_pos = index_to_position(index=a_idx, strides=a_strides)
                cache = fn(cache, a_storage[a_pos])
            out[out_pos] = cache

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    # TODO: Implement for Task 3.2.
    hidden = a_shape[-1]
    out_shape_prefix = out_shape[:-2]
    a_shape_prefix = a_shape[:-2]
    b_shape_prefix = b_shape[:-2]
    for o in np.ndindex(out_shape):
        out_idx = list(o)
        a_idx = [0 for _ in range(len(a_shape) - 2)]
        b_idx = [0 for _ in range(len(b_shape) - 2)]
        broadcast_index(
            big_index=out_idx[:-2],
            big_shape=out_shape_prefix,
            shape=a_shape_prefix,
            out_index=a_idx,
        )
        broadcast_index(
            big_index=out_idx[:-2],
            big_shape=out_shape_prefix,
            shape=b_shape_prefix,
            out_index=b_idx,
        )
        a_idx += [out_idx[-2], -1]
        b_idx += [-1, out_idx[-1]]
        s = 0.0
        for h in range(hidden):
            a_idx[-1] = h
            b_idx[-2] = h
            a_pos = index_to_position(index=a_idx, strides=a_strides)
            b_pos = index_to_position(index=b_idx, strides=b_strides)
            s += a_storage[a_pos] * b_storage[b_pos]
        out_pos = index_to_position(index=out_idx, strides=out_strides)
        out[out_pos] = s

tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
