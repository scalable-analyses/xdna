.. _ch:xdna1:

XDNA1 Kernel
============
Tiled tensor contractions use kernels that operate on subtensors (tiles) as their core building block.
Accelerating these kernels is crucial to the overall performance of tensor workloads.
As discussed in the chapter :ref:`ch:isa`, the XDNA cores' floating-point throughput is driven by VMAC.F.
On XDNA1, the highest-throughput VMAC.F operation performs a BF16 4×8×4 matrix multiplication.
On XDNA2, the highest-throughput VMAC.F computes a BFP16 8×8×8 matrix multiplication.

The goal of this chapter is to develop a high-performance tensor contraction kernel for XDNA1.
We pursue this goal through three sub-objectives:

1. Maximize the rate at which VMAC.F operations are issued. In the best case, a VMAC.F operation is issued every clock cycle.
2. Hide all other operations, e.g., loads and stores, and pointer arithmetic, behind computations.
3. Minimize bank conflicts.

.. note::

   We discuss the design and implementation of a representative best-case kernel.
   Many other variants of this kernel are required for implementing a flexible and high-performance tensor compiler.
   We plan to extend the *Hello XDNA* website with generalizations of the kernels presented in this chapter and the :ref:`ch:xdna2-kernel` chapter in the future, e.g., through just-in-time code generation.

Data Layout
-----------

.. figure:: ./figures/bf16_vmac_data.svg
   :name: tensor_kernels:reg_data_layout
   :width: 35 %
   :align: center

   Register data layout for the BF16 4×8×4 ``VMAC.F (<BMLd>|<BMHd>), (<BMLm>|<BMHm>), <Xr>, <Xs>, <Rn>`` operation computing ``[m,k],[k,n]->[m,n]`` with ``|m|=4``, ``|k|=8``, and ``|n|=4``.

:numref:`tensor_kernels:reg_data_layout` illustrates the register data layout required for the BF16 4×8×4 VMAC.F operation.
The operation multiplies a BF16 M×K matrix in one ``X`` register with a BF16 K×N matrix in another ``X`` register and adds the result to an FP32 M×N matrix in an accumulation register.
All matrices are stored in row-major order.
Note that this is equivalent to a column-major matrix-matrix multiplication if we exchange the two operands.

We can also write the operation as an einsum, assuming that all tensors are in row-major order: ``[m,k],[k,n]->[m,n]``.
The einsum notation becomes helpful when considering the more complex data layout of the entire kernel.

.. figure:: ./figures/xdna1_data_and_registers.svg
   :name: tensor_kernels:l1_data_layout
   :width: 100 %
   :align: center

   Data layout of the tensor contraction kernel in scratchpad memory (L1) of an executing compute tile. The kernel computes ``[m1,k1,m0,k0],[k1,n1,k0,n0]->[m1,n1,m0,n0]``, where the dimension sizes ``|m0|=4``, ``|k0|=8``, and ``|n0|=4`` are fixed. The first computation of a 2×4 block of output tiles is highlighted in darker colors.

:numref:`tensor_kernels:l1_data_layout` illustrates the data layout of the entire tensor contraction kernel.
It covers the two input tensors ``in0`` and ``in1``, as well as the output tensor ``out``.
The three tensors are tiled based on the requirements of the BF16 VMAC.F instruction.
In detail, ``in0`` has tiles of size M₀×K₀=4×8, ``in1`` of size K₀×N₀=8×4, and ``out`` of size M₀×N₀=4×4.
The tensor contraction kernel operates on three additional dimensions of type M, K and N.
The tiles are stored in row-major order; accordingly, ``in0`` uses M₁×K₁×M₀×K₀, while ``in1`` uses K₁×N₁×K₀×N₀, and ``out`` uses M₁×N₁×M₀×N₀.

As before, we can write the operation compactly as an einsum, assuming that all tensors are stored in row-major order: ``[m1,k1,m0,k0],[k1,n1,k0,n0]->[m1,n1,m0,n0]``.
The tile size requirement means that ``|m0|=4``, ``|k0|=8``, and ``|n0|=4``.
We discuss limitations on ``m1``, ``k1`` and ``n1`` in the following sections.

Design Decisions
----------------
We have already discussed the data layout of our tensor contraction kernel.
For this, we identified requirements on the tiling and identified an einsum that summarizes the contraction computed by the targeted kernel.
The dimensions ``m0``, ``k0`` and ``n0`` are consumed by the VMAC.F operation, while the handling of dimensions ``m1``, ``k1`` and ``n1`` is still unspecified.

Before introducing design decisions for the kernel, we recapitulate key hardware properties that have to be considered in the kernel design:

1. Loading a 64-byte accumulation register requires two 32-byte loads (VLDA).
   A load has a 7-cycle latency.
   An instruction can contain a single load or store operation accessing the accumulation registers.
2. VMAC.F reads the accumulation register in its third cycle (forwarding).
3. Combining properties 1 and 2, we can issue a dependent VMAC.F operation at the earliest in the sixth cycle of the second 32-byte load.
   In other words, the second load and the dependent VMAC.F have to be five cycles apart.
4. A VMAC.F operation has a latency of six cycles.

.. only:: comment

   ???

   5. TODO: "Only one operation per instruction can read from or write to a memory area."

We make the following *design decisions* in our kernel:

.. _xdna1-kernel:design-output-stationary:

Output Stationary
  Load each output tensor value to the register file exactly once.
  This means that an output value is kept in the respective accumulation register until all updates have been applied through VMAC.F operations.
  Storing intermediate values and loading them back into the registers is challenging because loads and stores must be in different instructions to avoid bank conflicts.

Register Blocking
  ``|m1|`` and ``|n1|`` must be multiples of the register-blocking size.
  Our example kernel uses a 2×4 register blocking scheme for the output tensor.
  This means that we use eight accumulation registers to hold the values of the eight output tiles as shown in :numref:`tensor_kernels:l1_data_layout`.

  Due to the 2×4 blocking, each loaded ``in0`` tile is used in four VMAC.F operations, and each ``in1`` tile is used in two operations.
  This reuse is required to hide the register data transfer behind computation.
  For example, we could not achieve this with a 2×2 blocking.

.. _xdna1-kernel:design-linear-contraction:

Linear Contraction Dimension
  The ``k1`` dimension is handled with linear code without loop structures.
  This allows for different combinations of operations per block and is necessary for register preloading.

.. _xdna1-kernel:design-single-loop:

Single Hardware Loop
  The ``m1`` and ``n1`` dimensions are represented by a single hardware loop.
  The first and last 2×4 blocks are computed outside of this loop, forming a *warm-up phase* and *cool-down phase*.

Double Buffering: Accumulation Registers
  A 2×4 block requires eight out of sixteen available accumulation registers.
  We alternate registers ``BML0``--``BML3`` and ``BMH0``--``BMH3`` with ``BML4``--``BML7`` and ``BMH4``--``BMH7`` to realize a double buffering scheme.
  This means that while updating the tiles in one half of the accumulation registers, we load the next 2×4 block into the other half.

Double Buffering: Vector Registers
  We also use double buffering for the registers holding tiles of ``in0``.
  In particular, we alternate ``X0`` and ``X1`` with ``X2`` and ``X3``.

Implementation
--------------

This section discusses the :download:`implementation <../../src/tensor_kernel_32x32x32_bf16_bf16_fp32.s>` of a representative XDNA1 tensor contraction kernel.
The kernel computes the einsum ``[m1,k1,m0,k0],[k1,n1,k0,n0]->[m1,n1,m0,n0]`` with dimension sizes ``|m0|=4``, ``|k0|=8``, ``|n0|=4``, ``|m1|=8``, ``|k1|=4``, and ``|n1|=8``.
It contains three parts: a warm-up phase, a hardware loop, and a cool-down phase.

Warm-Up Phase
^^^^^^^^^^^^^

.. _xdna1-kernel:lst-warm-up:

.. literalinclude:: ../../src/tensor_kernel_32x32x32_bf16_bf16_fp32.s
   :caption: Warm-up phase (lines 7-62) of the :download:`XDNA1 kernel <../../src/tensor_kernel_32x32x32_bf16_bf16_fp32.s>`.
   :language: asm
   :linenos:
   :lineno-match:
   :lines: 7-62

:numref:`xdna1-kernel:lst-warm-up` shows the warm-up phase of the kernel.
Load unit A loads data into the accumulation registers.
The sixteen 32-byte VLDA operations in lines 7--22 load the first 2×4 block of output tiles into the accumulation registers ``BML0``--``BML3`` and ``BMH0``--``BMH3``.
Additionally, in lines 7--18, load unit B is used to load the first two input tiles of ``in0`` into vector registers ``X0`` and ``X1``, as well as the first four input tiles of ``in1`` into ``X4``--``X7``.
The register mapping is also illustrated in :numref:`tensor_kernels:l1_data_layout`.
The first update of the 2×4 block is performed by the VMAC.F operations in lines 17--31.

In lines 15--19, the warm-up phase initializes the general-purpose registers ``R3``--``R7``.
These are used throughout the kernel and their values copied to modifier registers for subsequent updates of addresses in pointer registers.

We also see that load unit A is used to load the next 2×4 block of output tiles to accumulation registers ``BML4``--``BML7`` and ``BMH4``--``BMH7`` (lines 27--30, 37--40, 47--50, and 57--60).

The first instruction block (lines 7--31) contains eight VMAC.F operations and 17 NOPV operations, thus leaving the vector unit partially unutilized.
Every instruction in the following three eight-instruction blocks contains a VMAC.F operation, meaning that the BF16 matrix multiplication unit of the core is fully utilized.
In summary, the warm-up phase has a total of 49 instructions, out of which 32 contain VMAC.F operations.

.. _xdna1:imp-hardware-loop:

Hardware Loop
^^^^^^^^^^^^^

We must perform the loop setup at least 64 bytes before the loop's start address.
The first and last instructions in the loop must be 16-byte aligned.
Additionally, the last instruction covered by a loop must have a size of 16 bytes.
An instruction that contains operations for all functional units is 16 bytes wide.
A NOP instruction is only two bytes wide.

.. _xdna1:lst-hardware-loop-setup:

.. literalinclude:: ../../src/tensor_kernel_32x32x32_bf16_bf16_fp32.s
   :caption: Hardware loop setup (lines 29-31: chars 104+) of the :download:`XDNA1 kernel <../../src/tensor_kernel_32x32x32_bf16_bf16_fp32.s>`.
   :language: asm
   :linenos:
   :lineno-match:
   :lines: 29-31
   :dedent: 103

:numref:`xdna1:lst-hardware-loop-setup` shows the operations configuring the hardware loop.
``movxm ls, #.l_start`` copies the address of the first loop instruction into the loop start register.
The operation ``movxm le, #.l_end`` copies the address of the last loop instruction into the loop end register.
``movxm lc, #3`` copies the value 3 into the loop counter register.

.. _xdna1:lst-hardware-loop-body:

.. literalinclude:: ../../src/tensor_kernel_32x32x32_bf16_bf16_fp32.s
   :caption: Body of the loop (lines 64-148) in the :download:`XDNA1 kernel <../../src/tensor_kernel_32x32x32_bf16_bf16_fp32.s>`.
   :language: asm
   :linenos:
   :lineno-match:
   :lines: 64-148

:numref:`xdna1:lst-hardware-loop-body` shows the body of the tensor contraction kernel's loop.
In the first half of the body (lines 67--104), the values in accumulation registers with indices 4--7 are updated by the VMAC.F operations.
When entering the loop body, accumulation registers 0--3 hold the results of the preceding 2×4 block of output tiles.
During execution, these are written to scratchpad memory (L1) using VST operations.
Simultaneously, load unit A transfers the next 2×4 block's tiles into registers 0--3.

The second half of the loop body (lines 108--144) computes the pre-loaded 2×4 block and updates the output tiles in registers 0--3.
At the same time, the data of the now preceding 2×4 block, computed in the first half of the loop body, is written to memory, while the next block is loaded to registers 4--7.

Considering the XDNA1 vector unit, we see that every instruction in the loop body contains a VMAC.F operation.
Therefore, the unit is fully utilized and all of the 64 instructions in the loop body perform a BF16 4×8×4 matrix multiplication.

Cool-down Phase
^^^^^^^^^^^^^^^

.. _xdna1-kernel:lst-cool-down:

.. literalinclude:: ../../src/tensor_kernel_32x32x32_bf16_bf16_fp32.s
   :caption: Cool-down phase (lines 150-205) of the :download:`XDNA1 kernel <../../src/tensor_kernel_32x32x32_bf16_bf16_fp32.s>`.
   :language: asm
   :linenos:
   :lineno-match:
   :lines: 150-205

The cool-down phase is shown in :numref:`xdna1-kernel:lst-cool-down`.
It differs from the loop body in two key ways.
First, no preloading of the next 2×4 block of output tiles is required.
Second, most of the VST operations writing the last block in accumulation registers 4--7 are exposed, meaning they cannot be hidden behind VMAC.F operations.
In line 200, the ``ret lr`` operation is issued, which has six-cycle latency.

The cool-down phase contains a total of 47 instructions out of which 32 contain VMAC.F operations.

Kernel Efficiency
-----------------

Our XDNA1 tensor contraction kernel has the following utilization of the vector unit in the three parts:

* Warm-up phase: 32 out of 49 instructions contain VMAC.F operations.
* Hardware loop: All 64 instructions in the loop body contain VMAC.F operations. The loop executes three times, yielding 192 instructions containing VMAC.F operations.
* Cool-down phase: 32 out of 47 instructions contain VMAC.F operations.

In summary, the kernel consists of 288 instructions out of which 256 contain VMAC.F operations.
This leads to a theoretical utilization of 89%.
In other words, a compute tile running at 1.8 GHz would execute 1.6×10⁹ BF16 4×8×4 operations per second.
This is equivalent to a theoretical floating-point throughput of 410 BF16 GFLOPS.

We implemented a benchmark in which the tensor contraction kernel is called repeatedly in a loop on the NPU.
Benchmarking the kernel on an XDNA1 NPU (AMD Ryzen 7 8700G), we achieved 398 BF16 GFLOPS.
The benchmarking code is available from our `xdna <https://github.com/scalable-analyses/xdna>`__ repository.
To run the benchmark, execute the following commands:

.. code-block:: bash

   git clone https://github.com/scalable-analyses/xdna
   cd xdna
   make run

.. note::
   The installation of the MLIR-AIE compiler aiecc and Peano is documented in the `mlir-aie <https://github.com/Xilinx/mlir-aie>`__ repository.
   The Makefile assumes that the environment variable ``PEANO_INSTALL_DIR`` contains the path to Peano and that ``aiecc.py`` is available in the path.
   Use ``xrt-smi configure --pmode turbo`` to set the NPU clock to its maximum frequency.

