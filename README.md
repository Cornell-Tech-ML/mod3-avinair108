# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


Output from running parallel_check.py:

# MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (163) 
-------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                              | 
        out: Storage,                                                                      | 
        out_shape: Shape,                                                                  | 
        out_strides: Strides,                                                              | 
        in_storage: Storage,                                                               | 
        in_shape: Shape,                                                                   | 
        in_strides: Strides,                                                               | 
    ) -> None:                                                                             | 
        """NUMBA higher-order tensor map function. See `tensor_ops.py` for description.    | 
                                                                                           | 
        Optimizations:                                                                     | 
                                                                                           | 
        * Main loop in parallel                                                            | 
        * All indices use numpy buffers                                                    | 
        * When `out` and `in` are stride-aligned, avoid indexing                           | 
                                                                                           | 
        Args:                                                                              | 
        ----                                                                               | 
            out (Storage): storage for `out` tensor                                        | 
            out_shape (Shape): shape for `out` tensor                                      | 
            out_strides (Strides): strides for `out` tensor                                | 
            in_storage (Storage): storage for `in` tensor                                  | 
            in_shape (Shape): shape for `in` tensor                                        | 
            in_strides (Strides): strides for `in` tensor                                  | 
                                                                                           | 
        Returns:                                                                           | 
        -------                                                                            | 
            None : Fills in `out`                                                          | 
                                                                                           | 
        """                                                                                | 
        if np.array_equal(out_shape, in_shape) and np.array_equal(                         | 
            out_strides, in_strides                                                        | 
        ):                                                                                 | 
            for i in prange(len(out)):-----------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                                 | 
        else:                                                                              | 
            for i in prange(len(out)):-----------------------------------------------------| #3
                out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)----------------------| #0
                in_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)-----------------------| #1
                to_index(i, out_shape, out_index)                                          | 
                broadcast_index(out_index, out_shape, in_shape, in_index)                  | 
                                                                                           | 
                out_pos = index_to_position(out_index, out_strides)                        | 
                in_pos = index_to_position(in_index, in_strides)                           | 
                                                                                           | 
                out[out_pos] = fn(in_storage[in_pos])                                      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)


 
Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (200) is
 hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (201) is
 hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: in_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (236)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (236) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        """Numba higher-order tensor zip function.                         | 
                                                                           | 
        Optimizations:                                                     | 
                                                                           | 
        * Main loop in parallel                                            | 
        * All indices use numpy buffers                                    | 
        * When `out`, `a`, `b` are stride-aligned, avoid indexing          | 
                                                                           | 
        """                                                                | 
        if (                                                               | 
            np.array_equal(out_shape, a_shape)                             | 
            and np.array_equal(out_shape, b_shape)                         | 
            and np.array_equal(out_strides, a_strides)                     | 
            and np.array_equal(out_strides, b_strides)                     | 
        ):                                                                 | 
            for i in prange(len(out)):-------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                    | 
        else:                                                              | 
            for i in prange(len(out)):-------------------------------------| #8
                out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)------| #4
                a_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------| #5
                b_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------| #6
                                                                           | 
                to_index(i, out_shape, out_index)                          | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                                                                           | 
                out_pos = index_to_position(out_index, out_strides)        | 
                a_pos = index_to_position(a_index, a_strides)              | 
                b_pos = index_to_position(b_index, b_strides)              | 
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)


 
Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (266) is
 hoisted out of the parallel loop labelled #8 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (267) is
 hoisted out of the parallel loop labelled #8 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (268) is
 hoisted out of the parallel loop labelled #8 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: b_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (303)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (303) 
---------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                       | 
        out: Storage,                                                                  | 
        out_shape: Shape,                                                              | 
        out_strides: Strides,                                                          | 
        a_storage: Storage,                                                            | 
        a_shape: Shape,                                                                | 
        a_strides: Strides,                                                            | 
        reduce_dim: int,                                                               | 
    ) -> None:                                                                         | 
        # for i in prange(len(out)):                                                   | 
        #     out_index = np.zeros(out_shape, dtype=np.int32)                          | 
        #     a_index = np.zeros(a_shape, dtype=np.int32)                              | 
        #     to_index(i, out_shape, out_index)                                        | 
        #     out_pos = index_to_position(out_index, out_strides)                      | 
        #     result = out[out_pos]                                                    | 
        #     for j in range(a_shape[reduce_dim]):                                     | 
        #         np.copyto(a_index, out_index)                                        | 
        #         # Update the input index along the reduction dimension               | 
        #         a_index[reduce_dim] = j                                              | 
        #         # Convert the input index to a flat position in the input storage    | 
        #         a_pos = index_to_position(a_index, a_strides)                        | 
        #         # Apply the reduction function                                       | 
        #         result = fn(result, a_storage[a_pos])                                | 
                                                                                       | 
        #     out[out_pos] = result                                                    | 
        for i in prange(len(out)):-----------------------------------------------------| #10
            out_index: Index = np.zeros(MAX_DIMS, np.int32)----------------------------| #9
            reduce_size = a_shape[reduce_dim]                                          | 
            # Get the corresponding multidimensional index for output                  | 
            to_index(i, out_shape, out_index)                                          | 
            pos = index_to_position(out_index, out_strides)                            | 
                                                                                       | 
            # Iterate over the reduce dimension and apply the reduction function       | 
            for j in range(reduce_size):                                               | 
                out_index[reduce_dim] = j                                              | 
                out[pos] = fn(                                                         | 
                    out[pos], a_storage[index_to_position(out_index, a_strides)]       | 
                )                                                                      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)


 
Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (329) is
 hoisted out of the parallel loop labelled #10 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (345)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/avinair/Desktop/miniTorch4/mod3-avinair108/minitorch/fast_ops.py (345) 
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                | 
    out: Storage,                                                                           | 
    out_shape: Shape,                                                                       | 
    out_strides: Strides,                                                                   | 
    a_storage: Storage,                                                                     | 
    a_shape: Shape,                                                                         | 
    a_strides: Strides,                                                                     | 
    b_storage: Storage,                                                                     | 
    b_shape: Shape,                                                                         | 
    b_strides: Strides,                                                                     | 
) -> None:                                                                                  | 
    """NUMBA tensor matrix multiply function.                                               | 
                                                                                            | 
    Should work for any tensor shapes that broadcast as long as                             | 
                                                                                            | 
    ```                                                                                     | 
    assert a_shape[-1] == b_shape[-2]                                                       | 
    ```                                                                                     | 
                                                                                            | 
    Optimizations:                                                                          | 
                                                                                            | 
    * Outer loop in parallel                                                                | 
    * No index buffers or function calls                                                    | 
    * Inner loop should have no global writes, 1 multiply.                                  | 
                                                                                            | 
                                                                                            | 
    Args:                                                                                   | 
    ----                                                                                    | 
        out (Storage): storage for `out` tensor                                             | 
        out_shape (Shape): shape for `out` tensor                                           | 
        out_strides (Strides): strides for `out` tensor                                     | 
        a_storage (Storage): storage for `a` tensor                                         | 
        a_shape (Shape): shape for `a` tensor                                               | 
        a_strides (Strides): strides for `a` tensor                                         | 
        b_storage (Storage): storage for `b` tensor                                         | 
        b_shape (Shape): shape for `b` tensor                                               | 
        b_strides (Strides): strides for `b` tensor                                         | 
                                                                                            | 
    Returns:                                                                                | 
    -------                                                                                 | 
        None : Fills in `out`                                                               | 
                                                                                            | 
    """                                                                                     | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  | 
                                                                                            | 
    batch_size = out_shape[0]                                                               | 
    out_rows = out_shape[1]                                                                 | 
    out_cols = out_shape[2]                                                                 | 
    common_dim = a_shape[-1]                                                                | 
                                                                                            | 
    for batch in prange(batch_size):--------------------------------------------------------| #11
        for i in range(out_rows):  # Rows in the output tensor                              | 
            for j in range(out_cols):  # Columns in the output tensor                       | 
                # Initialize the accumulator for the dot product                            | 
                acc = 0.0                                                                   | 
                for k in range(common_dim):  # Shared dimension                             | 
                    a_idx = batch * a_batch_stride + i * a_strides[1] + k * a_strides[2]    | 
                    b_idx = batch * b_batch_stride + k * b_strides[1] + j * b_strides[2]    | 
                    acc += a_storage[a_idx] * b_storage[b_idx]                              | 
                # Write the accumulated value to the output tensor                          | 
                out_idx = (                                                                 | 
                    batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]        | 
                )                                                                           | 
                out[out_idx] = acc                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
