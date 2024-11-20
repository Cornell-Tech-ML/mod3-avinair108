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
<pre>
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
</pre>
