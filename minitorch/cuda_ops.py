# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:  # noqa: ANN003
    """CUDA version of numba.jit.

    This function takes a function `fn` and returns a CUDA compiled version of it.
    The returned function is a Numba CUDA kernel and can be launched on the GPU.

    Args:
    ----
        fn: function to be compiled.
        **kwargs: Additional keyword arguments to pass to numba.jit.

    Returns:
    -------
        A CUDA compiled function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:  # noqa: ANN001, ANN003
    """Numba jit function.

    This function takes a function `fn` and returns a numba.jitted version of it.
    The returned function is a Numba function and can be called on the host.

    Args:
    ----
        fn: function to be compiled.
        **kwargs: Additional keyword arguments to pass to numba.jit.

    Returns:
    -------
        A numba.jitted function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """CUDA-based higher-order tensor reduce function.

        This function returns a callable that reduces a tensor along the specified dimension using a given binary function. The reduction is performed in parallel on the GPU using CUDA.

        Args:
        ----
            fn: A binary function that takes two floats and returns a float. This function is used to reduce the tensor along the specified dimension.
            start: A float representing the initial value to start the reduction from. Default is 0.0.

        Returns:
        -------
            A callable function that takes a tensor and an integer representing the dimension to reduce. It returns a new tensor where the specified dimension is reduced using the given function.

        Note:
        ----
            The reduction is performed using CUDA, leveraging parallel execution for efficient computation.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index, in_strides)
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Load data into shared memory
    cache[pos] = a[i] if i < size else 0.0
    cuda.syncthreads()

    # Perform reduction in shared memory
    step = 1
    while step < BLOCK_DIM:
        if pos % (2 * step) == 0 and pos + step < BLOCK_DIM:
            cache[pos] += cache[pos + step]
        step *= 2
        cuda.syncthreads()

    # Write the result back to the global memory
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Higher-order tensor sum function.

    This function computes the sum of all values in a tensor.

    Args:
    ----
        a: The tensor to sum.

    Returns:
    -------
        A tensor with a single value: the sum of all values in `a`.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        # BLOCK_DIM = 1024
        # cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # pos = cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            # Reduce over dimension
            # reduce_index = cuda.local.array(MAX_DIMS, numba.int32)
            for j in range(a_shape[reduce_dim]):
                out_index[reduce_dim] = j
                in_pos = index_to_position(out_index, a_strides)
                reduce_value = fn(reduce_value, a_storage[in_pos])

            out[out_pos] = reduce_value

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    # intialize the output element to zero in global memory
    if i < size and j < size:
        out_pos = i * size + j  # find linear index
        out[out_pos] = 0.0

    # Load global memory contents for a and b into shared memory
    for k in range(size):
        if i < size and k < size:
            a_shared[i, k] = a[
                i * size + k
            ]  # load current element of a into shared memory
        if k < size and j < size:
            b_shared[k, j] = b[
                k * size + j
            ]  # load current element of b into shared memory
    # sync threads to make sure all data is loaded
    cuda.syncthreads()

    # keep track of dot product
    result = 0.0
    # compute dot product for current element in out
    for k in range(size):
        result += a_shared[i, k] * b_shared[k, j]

    # Write the result back to global memory
    if i < size and j < size:
        out_pos = i * size + j  # find linear index
        out[out_pos] = result  # write result


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Higher-order tensor matrix multiply function.

    This function computes the matrix multiplication of two tensors with shape
    (size, size).

    Args:
    ----
        a (Tensor): The first tensor to multiply.
        b (Tensor): The second tensor to multiply.

    Returns:
    -------
        A tensor with shape (size, size) that is the result of multiplying
        `a` and `b`.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]

    # Variable to keep track of the current element's dot product result
    value = 0.0
    # loop over tiles in shared dimension for a this is a_shape[-1], for b this is b_shape[-2]
    for tile in range((a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM):
        # if tile a is in bounds, load into shared memory
        if i < a_shape[-2] and tile * BLOCK_DIM + pj < a_shape[-1]:
            # loading into shared memroy for a
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride  # batch offset
                + i * a_strides[-2]  # row offset
                + (tile * BLOCK_DIM + pj) * a_strides[-1]  # column offset
            ]
        # out of bounds, load 0
        else:
            a_shared[pi, pj] = 0.0
        # if tile b is in bounds, load into shared memory
        if tile * BLOCK_DIM + pi < b_shape[-2] and j < b_shape[-1]:
            # loading into shared memory for b
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride  # batch offset
                + (tile * BLOCK_DIM + pi) * b_strides[-2]  # row offset
                + j * b_strides[-1]  # column offset
            ]
        # out of bounds, load 0
        else:
            b_shared[pi, pj] = 0.0
        # Synchronize all threads to ensure shared memory is fully loaded before proceeding
        cuda.syncthreads()
        # dot product calculation for current tile (32 wide inner product)
        for k in range(BLOCK_DIM):
            value += a_shared[pi, k] * b_shared[k, pj]
        # Synchronize all threads to prevent potential race conditions/data issues
        cuda.syncthreads()
    # check if within bounds before writing to global memory
    if i < out_shape[-2] and j < out_shape[-1]:
        # compute position using batch, row, column offsets
        out_pos = batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
        # store computed value in the output tensor
        out[out_pos] = value


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
