.. _ch:isa:

Instruction Set Architecture
============================
As of 2026, the XDNA official documentation focuses mainly on the higher-level software stack and provides limited technical details.
However, by combining information from several publicly available sources with microbenchmarks, we can construct an accurate picture of the hardware.

XDNA1 and XDNA2 are AMD's consumer variants of the AIE-ML (AIE2) and AIE-ML v2 (AIE2p) architectures.
Public documentation for the AIE-ML and AIE-ML v2 architectures is available as part of AMD's Versal AI Edge Series.
Architecturally, XDNA1 closely resembles `AIE-ML <https://docs.amd.com/r/en-US/am020-versal-aie-ml>`__, whereas XDNA2 diverges more from `AIE-ML v2 <https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2>`__.
Nonetheless, the AIE-ML and AIE-ML v2 architecture manuals are helpful for understanding the features of the microarchitectures.

As stated in an `AMD paper <https://ieeexplore.ieee.org/document/10592049>`__, XDNA cores use a very-long instruction word (VLIW) instruction set architecture (ISA) that supports single-instruction multiple-data (SIMD) operations for fixed- and floating-point arithmetic.
Although the ISA itself is not publicly documented, the open-source, LLVM-based compiler framework `Peano
<https://github.com/Xilinx/llvm-aie>`__ implements it.
Specifically, this framework can compile the `AIE-API <https://download.amd.com/docnav/aiengine/xilinx2025_2/aiengine_api/aie_api/doc/index.html>`__, which contains intrinsic functions for programming the compute-tile cores.
Peano emits assembly code for intrinsic kernels compiled from C++.
This allows us to infer the ISA from the generated code.
Furthermore, microbenchmarking operations within instruction words allows us to determine their hardware execution properties.
Operation latencies are particularly important because they are crucial for scheduling operations in VLIW code.

.. _isa:sec-fp-mat_ops:

Floating-Point Matrix Operations
--------------------------------
*BF16* is a popular ML data format `introduced <https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus>`__ by Google in 2018.
A BF16 number has a sign bit, eight exponent bits, and seven mantissa bits.
Thus, we obtain the same dynamic range as FP32, which has eight exponent bits as well.
*BFP16* is a block-floating-point format, which means that multiple values share the same exponent.
Specifically, BFP16 uses eight bits for the exponent and groups eight values into a block.
Each of the entries in the block has a sign and seven mantissa bits.
This results in a total of 8 + 8 * (1 + 7) = 72 bits, or 9 bytes, for all eight values.
Details on the format are available in the `AIE-API <https://download.amd.com/docnav/aiengine/xilinx2025_2/aiengine_api/aie_api/doc/group__group__basic__types.html>`__, where it has the name *bfp16ebs8*.

.. container:: table-with-notes

   .. list-table:: Excerpt of the floating-point matrix-multiplication modes listed in the `AIE-API <https://download.amd.com/docnav/aiengine/xilinx2025_2/aiengine_api/aie_api/doc/group__group__mmul.html#group_mmul_page_supported_regular_modes>`__.
      :name: isa:fp-modes
      :header-rows: 1
      :align: center
      :widths: 20 40 40 40

      * - Arch.
        - bfloat16 × bfloat16
        - float × float\ :sup:`d`
        - bfp16 × bfp16

      * - AIE-ML/XDNA1
        - | 4×8×4
          | 8×8×4\ :sup:`a`
          | 4×16×4\ :sup:`ab`
          | 8×8×8\ :sup:`ab`
        - | 4×8×4
          | 4×1×4\ :sup:`b`
          | 4×1×8\ :sup:`ab`
        -

      * - XDNA2
        - | 8×8×4\ :sup:`ab`
          | 4×8×8\ :sup:`abc`
          | 4×8×4\ :sup:`ab`
          | 8×8×8\ :sup:`e`
          | 8×1×8\ :sup:`b`
        - 4×8×4\ :sup:`ab`
        - | 8×8×8
          | 8×8×16\ :sup:`ab`

   | :sup:`a` Emulated using multiple intrinsic calls.
   | :sup:`b` Require additional data manipulation.
   | :sup:`c` 32b * 16b multiplications are emulated on AIE-ML/XDNA1, XDNA2, and AIE-MLv2.
   | :sup:`d` float multiplications are emulated on AIE-ML/XDNA1, XDNA2, and AIE-MLv2 using native bfloat16 multiplications.
   | :sup:`e` Mode available through block-floating-point emulation to increase throughput at the cost of accuracy. Enabled by defining ``AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`` at compile time.


The AIE-API `lists <https://download.amd.com/docnav/aiengine/xilinx2025_2/aiengine_api/aie_api/doc/group__group__mmul.html#group_mmul_page_supported_regular_modes>`__ supported matrix-multiplication modes.
:numref:`isa:fp-modes` provides an excerpt of the modes relevant for our targeted tensor workloads.
We observe that the BF16 4×8×4 mode runs natively on XDNA1, while the BFP16 8×8×8 mode runs natively on XDNA2.
All other modes are emulated in software.

XDNA1
-----

Inferring the ISA
^^^^^^^^^^^^^^^^^
This section illustrates the process of inferring the XDNA1 ISA using AIE-API intrinsics and Peano.
We use the BF16 4×8×4 matrix-matrix multiplication mode, which runs natively on XDNA1, as an example.
The notation specifies an M×K×N matrix-matrix multiplication C+=AB, where all matrices are in row-major order:

M
  Appears in A (rows) and C (rows).
  In the example M=4.
K
  Appears in A (columns) and B (rows).
  K is the contraction dimension.
  In the example K=8.
N
  Appears in B (columns) and C (columns).
  In the example N=4.

.. _isa:bf16_mac_aie_api:

.. literalinclude:: ../../src/bf16_mac.cpp
   :caption: :download:`AIE-API kernel <../../src/bf16_mac.cpp>` that uses the BF16 4×8×4 matrix-multiplication mode.
   :language: cpp
   :linenos:

:numref:`isa:bf16_mac_aie_api` shows a simple AIE-API kernel that performs a BF16 4×8×4 operation in line 25 when instantiated with the template parameters from line 38.
The target matrix multiplication mode is set up in line 9.
It operates on BF16 inputs and accumulates into FP32 as specified by ``accfloat``.
Lines 12--14 declare the BF16 4×8 matrix ``mat_in0``, the 8×4 BF16 matrix ``mat_in1``, and the 4×4 FP32 matrix ``mat_out``.
Next, the ``load_v`` intrinsics in lines 17--19 load data from memory into registers.
Line 22 declares that ``mat_out`` should be used for accumulation of the matrix multiplication in line 25.
The last intrinsic function in line 28 stores the data in ``mat_out`` to memory at address ``ptr_out``.

Using the Peano compiler, we can compile the AIE-API kernel and obtain the generated assembly code:

.. code:: sh

   clang++ -O2 -std=c++20 --target=aie2-none-unknown-elf \
           -I aie_api/include -S bf16_mac.cpp -o bf16_mac.s

.. note::

   The installation of Peano is documented in the `mlir-aie <https://github.com/Xilinx/mlir-aie>`__ repository.
   Additionally, the AIE-API header-only `library <https://github.com/Xilinx/aie_api.git>`__ is required to compile the kernel (``-I aie_api/include``).

.. _isa:bf16_mac_asm:

.. literalinclude:: ../../src/bf16_mac.s
   :caption: :download:`Assembly code <../../src/bf16_mac.s>` obtained from the compiled AIE-API kernel.
   :language: asm
   :lines: 6-26
   :lineno-start: 6
   :linenos:

:numref:`isa:bf16_mac_asm` shows the relevant part of the assembly code generated from the AIE-API kernel in :numref:`isa:bf16_mac_aie_api`.
The label ``bf16_mac`` in line 6 marks the entry point of the function.
Each of lines 8--26 represents a VLIW instruction, potentially consisting of multiple operations separated by semicolons.
For example, the instruction ``vlda wl2, [p1, #0]; paddb [p0], #32; padds [p3], #32`` in line 10 consists of three operations that are not ``nop``.
The function parameters ``ptr_in0``, ``ptr_in1``, and ``ptr_out`` are passed via the pointer registers ``P0``, ``P1``, and ``P2``.

We give high-level descriptions of the operations in each instruction:

**Line 8** -- ``nopb; nopa; nops; nopx; mov p3, p0; nopv``

* ``mov p3, p0``: Copy the value in pointer register ``P0`` to ``P3``.

**Line 9** -- ``vldb wl0, [p0, #0]; mov p0, p1``

* ``vldb wl0, [p0, #0]``: Load 32 bytes (16 BF16 values) from the address in pointer register ``P0`` into the vector register ``WL0``.
  The lower 32 bytes of the 64-byte register ``X0`` overlap with ``WL0``.

* ``mov p0, p1``: Copy the value in pointer register ``P1`` to ``P0``.

**Line 10** -- ``vlda wl2, [p1, #0]; paddb [p0], #32; padds [p3], #32``

* ``vlda wl2, [p1, #0]``: Load 32 bytes (16 BF16 values) from the address in pointer register ``P1`` into the vector register ``WL2``.
* ``paddb [p0], #32``: Increment the value in pointer register ``P0`` by 32.
* ``padds [p3], #32``: Increment the value in pointer register ``P3`` by 32.

**Line 11** -- ``vlda wh0, [p3, #0]; vldb wh2, [p0, #0]``

* ``vlda wh0, [p3, #0]``: Load 32 bytes (16 BF16 values) from the address in pointer register ``P3`` into the vector register ``WH0``.
* ``vldb wh2, [p0, #0]``: Load 32 bytes (16 BF16 values) from the address in pointer register ``P0`` into the vector register ``WH2``.

**Line 12** -- ``vlda amhh0, [p2, #32]``

* ``vlda amhh0, [p2, #32]``: Load 32 bytes (8 FP32 values) from the address in pointer register ``P2`` with an offset of 32 into accumulation register ``AMHH0``. The upper 32 bytes of the accumulation register ``BMH0`` overlap with ``AMHH0``.

**Line 13** -- ``vlda amhl0, [p2, #0]``

* ``vlda amhl0, [p2, #0]``: Load 32 bytes (8 FP32 values) from the address in pointer register ``P2`` into accumulation register ``AMHL0``. The lower 32 bytes of the accumulation register ``BMH0`` overlap with ``AMHL0``.

**Line 17** -- ``mova r0, #28``

* ``mova r0, #28``: Write the value 28 into the 4-byte general purpose register ``R0``.

**Line 18** -- ``vmac.f bmh0, bmh0, x0, x2, r0``

* ``vmac.f bmh0, bmh0, x0, x2, r0``: Perform a matrix-matrix multiplication where the matrices in the vector register ``X0`` and ``X2`` are multiplied.
  The FP32 values of the matrix in ``BMH0`` (second occurrence in the operation) are added to the multiplication result.
  After adding, the final result is stored in ``BMH0`` (first occurrence in the operation).

  Register ``R0`` is used as a configuration register, where the value ``28`` corresponds to the BF16 4×8×4 matrix-matrix multiplication.

**Line 21** -- ``ret lr``

* ``ret lr``: Write the address in the link register to the program counter.
  This operation has a 6-cycle latency, meaning the next 5 instructions are executed during the delay slots.

**Line 24** -- ``vst amhh0, [p2, #32]``

* ``vst amhh0, [p2, #32]``: Store the 8 FP32 values from ``AMHH0`` to the address in register ``P2`` with an offset of 32.

**Line 25** -- ``vst amhl0, [p2, #0]``

* ``vst amhl0, [p2, #0]``: Store the 8 FP32 values from ``AMHL0`` to the address in register ``P2``.

Some of the instruction properties can be seen directly in the assembly code.
In particular, the XDNA cores lack stall logic and execute instructions in order.
This means that the compiler inserts no-operation (``nop``) instructions to resolve dependencies.

Consider, for example, the instruction in line 18 that exclusively contains the ``vmac.f`` matrix-multiplication operation.
The two stores in lines 24 and 25 write the result of the multiplication to memory and therefore depend on the completion of ``vmac.f``.
We see that the compiler schedules the first store in line 24 after six cycles.
This indicates that the ``vmac.f`` operation has a latency of 6 cycles.

The details on Peano's scheduling rules for the used ``vmac.f bmh0, bmh0, x0, x2, r0`` operation are defined in the AIEngine Scheduling Definitions for `AIE2 <https://github.com/Xilinx/llvm-aie/blob/70ac20b17d30b489a8265a10d1c65a5bf8788c15/llvm/lib/Target/AIE/AIE2Schedule.td#L974-L975>`__:

.. code-block:: none

   InstrItinData<II_VMACf, [InstrStage<1, [R_RV_PORT]>, EmptyCycles<4>, InstrStage<1, [CM_WA_PORT]>],
                 [6,3,1,1,1,/*srFPFlags*/7, /*crFPMask*/7]>,

We see that the compiler assumes a 6 cycle latency for the instruction.
More precisely, the write-back to the accumulation register ``bmh0`` completes after 6 cycles, whereas the accumulator is read in the third cycle.
This means that our kernels can exploit forwarding and only have to ensure that preceding write-backs are completed before the third cycle of the operation.
The latencies of the BFP16 8×8×8 operation mentioned in :ref:`isa:sec-fp-mat_ops` are part of the `AIE2p <https://github.com/Xilinx/llvm-aie/blob/70ac20b17d30b489a8265a10d1c65a5bf8788c15/llvm/lib/Target/AIE/aie2p/AIE2PGenSchedule.td>`__ scheduling definitions.

In general, we can follow a similar analysis for all operations.
If we are uncertain about a particular latency, we can increase or decrease the distances in the assembly code and run a small test to check if the change was valid.

Summary
^^^^^^^
We observed no differences between the register files of XDNA1 and `AIE-ML <https://docs.amd.com/r/en-US/am020-versal-aie-ml/Register-Files>`__.
The process of determining the available operations together with their respective latencies is tedious.
We have done this for the XDNA1 operations that are required for our target tensor contraction workload.
The obtained ISA is provided in :numref:`isa:xdna1-isa`.

.. container:: table-with-notes

   .. table:: XDNA1 operations that are required for our target tensor contraction workload. The operations are sorted by groups and provided together with their latencies.
      :name: isa:xdna1-isa
      :align: center
      :widths: 70 5 25

      +---------------------------------------------------------------+------+----------------------------+
      | Operation                                                     | Lat. | Notes                      |
      +===============================================================+======+============================+
      | **nop** - no-operation                                                                            |
      +---------------------------------------------------------------+------+----------------------------+
      | ``NOP``                                                       | \-   | do nothing                 |
      +---------------------------------------------------------------+------+----------------------------+
      | ``NOP(V|A|B|S|X|M|XM)``                                       | \-   | do nothing in unit         |
      +---------------------------------------------------------------+------+----------------------------+
      | **mov - move**                                                                                    |
      +---------------------------------------------------------------+------+----------------------------+
      | ``MOV <Rd>, #<imm10>``                                        | 1    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | ``MOV <Rd>, <Rm>``                                            | 1    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | ``MOV(A|X) <Rd>, #<imm8>``                                    | 1    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | ``MOV(A|X) <Rd>, <Rm>``                                       | 1    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | ``MOVXM <Rd>, #<imm32>``                                      | 1    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | **ld - load** - 4 byte load                                                                       |
      +---------------------------------------------------------------+------+----------------------------+
      | ``LD(A|B) <Rd>, [<Pn>], #<imm8>``                             | 6    | post-index                 |
      +---------------------------------------------------------------+------+----------------------------+
      | ``LD(A|B) <Rd>, [<Pn>], <Mm>``                                | 6    | post-index                 |
      +---------------------------------------------------------------+------+----------------------------+
      | ``LD(A|B) <Rd>, [<Pn>, #<imm8>]``                             | 6    | offset                     |
      +---------------------------------------------------------------+------+----------------------------+
      | ``LD(A|B) <Rd>, [<Pn>, <DJm>]``                               | 6    | offset                     |
      +---------------------------------------------------------------+------+----------------------------+
      | **vld - vector load** - 32 byte loads                                                             |
      +---------------------------------------------------------------+------+----------------------------+
      | ``VLD(A|B) (<WLd>|<WHd>), [<Pn>], #<imm8>``                   | 7    | post-index\ :sup:`a`       |
      +---------------------------------------------------------------+------+----------------------------+
      | ``VLDA (<AMLLd>|<AMLHd>|<AMHLd>|<AMHHd>), [<Pn>], #<imm8>``   | 7    | post-index\ :sup:`a`       |
      +---------------------------------------------------------------+------+----------------------------+
      | ``VLDA.CONV.FP32.BF16 (<BMLd>|<BMHd>), [<Pn>], #<imm8>``      | 7    | post-index\ :sup:`a`       |
      +---------------------------------------------------------------+------+----------------------------+
      | **st - store** - 4 byte store                                                                     |
      +---------------------------------------------------------------+------+----------------------------+
      | ``ST <Rd>, [<Pn>], #<imm8>``                                  | 6    | post-index\ :sup:`a`       |
      +---------------------------------------------------------------+------+----------------------------+
      | **vst - vector store** - 32 byte store                                                            |
      +---------------------------------------------------------------+------+----------------------------+
      | ``VST (<WLd>|<WHd>), [<Pn>], #<imm8>``                        | 2    | post-index\ :sup:`a`       |
      +---------------------------------------------------------------+------+----------------------------+
      | ``VST (<AMLLd>|<AMLHd>|<AMHLd>|<AMHHd>), [<Pn>], #<imm8>``    | 2    | post-index\ :sup:`a`       |
      +---------------------------------------------------------------+------+----------------------------+
      | ``VST.CONV.BF16.FP32 (<BMLd>|<BMHd>), [<Pn>], #<imm8>``       | 2    | post-index\ :sup:`a`       |
      +---------------------------------------------------------------+------+----------------------------+
      | **vshuffle - vector shuffle**                                                                     |
      +---------------------------------------------------------------+------+----------------------------+
      | ``VSHUFFLE (<Xd>|<BMLd>|<BMHd>),  <Xr>, <Xs>, <Rn>``          | 2    | conf. register\ :sup:`b`   |
      +---------------------------------------------------------------+------+----------------------------+
      | **add - addition**                                                                                |
      +---------------------------------------------------------------+------+----------------------------+
      | ``ADD <Rd>, <Rm>, #<imm7>``                                   | 1    | exp. leading bit\ :sup:`c` |
      +---------------------------------------------------------------+------+----------------------------+
      | ``ADD <Rd>, <Rm>, <Rn>``                                      | 1    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | **padd - pointer addition**                                                                       |
      +---------------------------------------------------------------+------+----------------------------+
      | ``PADD(A|B|S) [<Pd>], <Mn>``                                  | 1    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | ``PADD(A|S) [<Pd>], #<imm11>``                                | 1    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | ``PADDB [<Pd>], #<imm10>``                                    | 1    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | **mul - multiplication**                                                                          |
      +---------------------------------------------------------------+------+----------------------------+
      | ``MUL <Rd>, <Rm>, <Rn>``                                      | 2    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | **vmac - vector multiply accumulate**                                                             |
      +---------------------------------------------------------------+------+----------------------------+
      | ``VMAC.F (<BMLd>|<BMHd>), (<BMLm>|<BMHm>), <Xr>, <Xs>, <Rn>`` | 6    | lfw, conf. reg.\ :sup:`d`  |
      +---------------------------------------------------------------+------+----------------------------+
      | **comparisons**                                                                                   |
      +---------------------------------------------------------------+------+----------------------------+
      | ``(GT|LT|GE|LE){U} <Rd>, <Rm>, <Rn>``                         | 1    |  ``U`` for unsigned        |
      +---------------------------------------------------------------+------+----------------------------+
      | ``SEL.(EQZ|NEZ) <Rd>, <Rm>, <Rn>, <R27>``                     | 1    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | **j - jump**                                                                                      |
      +---------------------------------------------------------------+------+----------------------------+
      | ``J #<label>``                                                | 6    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | ``J(Z|NZ) <Rd>, #<label>``                                    | 6    |                            |
      +---------------------------------------------------------------+------+----------------------------+
      | ``RET <LR>``                                                  | 6    |                            |
      +---------------------------------------------------------------+------+----------------------------+

   | :sup:`a` same addressing modes as ld
   | :sup:`b` ``<Rn>`` as configuration register (value 28 for ``4x8 -> 8x4``, value 29 for ``8x4 -> 4x8``)
   | :sup:`c` Leading bit is expanded.
   | :sup:`d` ``(<BMLm>|<BMHm>)`` are read in third cycle. ``<Rn>`` as configuration register (value 28 for ``4x8x4-bfloat16``)

XDNA2
-----

.. _xdna2:infer-isa:

Inferring the ISA
^^^^^^^^^^^^^^^^^
The AIE-API can emulate BF16 8×8×8 matrix multiplications through BFP16 8×8×8 operations but does not expose a direct BFP16 8×8×8 intrinsic.

.. _isa:bfp16_emu_mac_aie_api:

.. literalinclude:: ../../src/bfp16_emu_mac.cpp
   :caption: :download:`AIE-API kernel <../../src/bfp16_emu_mac.cpp>` that can be compiled to use BFP16 8×8×8 matrix-matrix multiplications.
   :language: cpp
   :linenos:

:numref:`isa:bfp16_emu_mac_aie_api` shows an AIE-API kernel that performs an 8×8×8 BF16 matrix multiplication.
The template is identical to the XDNA1 version in :numref:`isa:bf16_mac_aie_api` but initialized with the appropriate dimension sizes in line 38.

Defining the directive ``AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`` instructs the AIE-API to emulate the matrix multiplication in the kernel with BFP16 operations:

.. code:: sh

   clang++ -O2 -std=c++20 --target=aie2p-none-unknown-elf \
           -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 \
           -I aie_api/include -S bfp16_emu_mac.cpp -o bfp16_emu_mac.s

.. _isa:bfp16_mac_asm:

.. literalinclude:: ../../src/bfp16_emu_mac.s
   :caption: :download:`Assembly code <../../src/bfp16_emu_mac.s>` obtained from the AIE-API kernel when compiled with the defined directive ``AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16``.
   :language: asm
   :lines: 6-39
   :lineno-match:
   :linenos:

The relevant part of the generated assembly code is shown in :numref:`isa:bfp16_mac_asm`.
Line 29 contains the instruction ``vmac.f dm0, dm0, ex0, ex2, r0``, which performs the BFP16 matrix multiplication.

Since the intrinsics assume pointers to BF16 inputs and FP32 output, the compiler issues respective conversion operations to BFP16.
We were not able to write an AIE-API kernel that reads BFP16 values directly from L1 scratchpad memory.
Instead, we examined the `Peano test code <https://github.com/Xilinx/llvm-aie/blob/70ac20b17d30b489a8265a10d1c65a5bf8788c15/llvm/test/CodeGen/AIE/aie2p/end-to-end/conv2d_bfp16_kernel_red.ll#L49-L62>`__ to identify the necessary operations for loading BFP16 data into the ``EX`` registers.

.. code-block:: asm
   :caption: Relevant operations for loading an ``EX`` register in `Peano test code <https://github.com/Xilinx/llvm-aie/blob/70ac20b17d30b489a8265a10d1c65a5bf8788c15/llvm/test/CodeGen/AIE/aie2p/end-to-end/conv2d_bfp16_kernel_red.ll#L49-L62>`__.
   :name: isa:load_ex_register
   :lineno-start: 49
   :linenos:

   movx r24, #0
   vlda.fill.512 [p0, lf0, r24]
   // no instruction in line
   vlda.pop.576 ex5, [p0, lf0, r24]








   // no instruction in line
   // no instruction in line
   vmac.f dm2, dm2, ex11, ex5, r0

The relevant part is shown in :numref:`isa:load_ex_register`.
The required 576-bit load for an ``EX`` register is split into multiple operations.
Line 49 initializes ``R24`` to 0.
``R24`` is used as an offset and to track the pipeline fill stage.
``vlda.fill.512 [p0, lf0, r24]`` loads 512 bits into load file 0 and increments the value in ``R24`` by 64 (the number of loaded bytes).
``vlda.pop.576 ex5, [p0, lf0, r24]`` loads 512 bits using the value in ``R24`` as an address offset.
Depending on the value in ``R24``, the pipeline consumes the 64 bytes already in LF0 and 8 additional bytes from the subsequent fetch to assemble the 576-bit ``EX5`` register.
The operation increments ``P0`` by 72 and decrements ``R24`` by 8 (number of bytes drained from the fill stage).
The ``EX5`` register is used after eight instructions, indicating an eight-cycle latency.

Summary
^^^^^^^

XDNA2 has five 2048-bit accumulator registers (``DM0``--``DM4``) and their 1024-bit/512-bit views in the ISA.
Compared to `AIE-ML v2 <https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/Accumulator-Registers>`__,  this represents a 1.6× reduction in the architecturally visible accumulator count.

:numref:`isa:xdna2-exp-regs`, :numref:`isa:xdna2-vec-with-exp-regs`, :numref:`isa:xdna2-ex0-data-layout`, and :numref:`isa:xdna2-load-store-files` summarize additional information on registers that are relevant for BFP16 operations.
The inferred ISA is provided in :numref:`isa:xdna2-isa`.

.. table:: Exponent Registers.
    :width: 70%
    :name: isa:xdna2-exp-regs
    :align: center

    +------------------------+-----------------------------+
    | .. centered:: 32-bit   |  .. centered:: 64-bit       |
    +========================+=============================+
    | .. centered:: ``EL0``  |  .. centered:: ``E0``       |
    +------------------------+                             |
    | .. centered:: ``EH0``  |                             |
    +------------------------+-----------------------------+
    | .. centered:: ...      | .. centered:: ...           |
    +------------------------+                             |
    | .. centered:: ...      |                             |
    +------------------------+-----------------------------+
    | .. centered:: ``EL11`` | .. centered:: ``E11``       |
    +------------------------+                             |
    | .. centered:: ``EH11`` |                             |
    +------------------------+-----------------------------+

.. table:: Vector Registers with Exponent Registers.
    :width: 70%
    :widths: 60 40
    :name: isa:xdna2-vec-with-exp-regs
    :align: center

    +----------------------------------+-------------------------+
    | .. centered:: 8-byte  + 64-byte  |  .. centered:: 72-byte  |
    +==================================+=========================+
    | .. centered::  ``E0`` + ``X0``   |  .. centered:: ``EX0``  |
    +----------------------------------+-------------------------+
    | .. centered:: ...                |  .. centered:: ...      |
    +----------------------------------+-------------------------+
    | .. centered::  ``E11`` + ``X11`` |  .. centered:: ``EX11`` |
    +----------------------------------+-------------------------+



.. table:: The data layout of ``EX0`` is as follows:
    ``EX0`` overlaps with ``E0`` and ``X0`` in an interleaved manner.
    The first byte of ``EX0`` overlaps with the first byte of ``E0``.
    The next eight bytes of ``EX0`` overlap with the first eight bytes of ``X0``.
    This pattern repeats.
    :width: 70 %
    :widths: 50 50
    :name: isa:xdna2-ex0-data-layout
    :align: center

    +----------------------------------------------+------------------------------+
    | .. centered:: 1-byte    +  8-byte            | .. centered::  9-byte        |
    +==============================================+==============================+
    | .. centered:: ``E0[0]`` + ``X0[ 0: 7]``      | .. centered:: ``EX0[ 0: 8]`` |
    +----------------------------------------------+------------------------------+
    | .. centered:: ``E0[1]`` + ``X0[ 8:15]``      | .. centered:: ``EX0[ 9:17]`` |
    +----------------------------------------------+------------------------------+
    | .. centered:: ``E0[2]`` + ``X0[16:23]``      | .. centered:: ``EX0[18:26]`` |
    +----------------------------------------------+------------------------------+
    | .. centered:: ``E0[3]`` + ``X0[24:31]``      | .. centered:: ``EX0[27:35]`` |
    +----------------------------------------------+------------------------------+
    | .. centered:: ``E0[4]`` + ``X0[32:39]``      | .. centered:: ``EX0[36:44]`` |
    +----------------------------------------------+------------------------------+
    | .. centered:: ``E0[5]`` + ``X0[40:47]``      | .. centered:: ``EX0[45:53]`` |
    +----------------------------------------------+------------------------------+
    | .. centered:: ``E0[6]`` + ``X0[48:55]``      | .. centered:: ``EX0[54:62]`` |
    +----------------------------------------------+------------------------------+
    | .. centered:: ``E0[7]`` + ``X0[56:63]``      | .. centered:: ``EX0[63:71]`` |
    +----------------------------------------------+------------------------------+

.. table:: Load and store files.
    :width: 70%
    :name: isa:xdna2-load-store-files
    :align: center

    +------------------------+-----------------------------+
    | .. centered:: 512-bit  |  .. centered:: 1024-bit     |
    +========================+=============================+
    | .. centered:: ``LFL0`` |  .. centered:: ``LF0``      |
    +------------------------+                             |
    | .. centered:: ``LFH0`` |                             |
    +------------------------+-----------------------------+
    | .. centered:: ``LFL1`` |  .. centered:: ``LF1``      |
    +------------------------+                             |
    | .. centered:: ``LFH1`` |                             |
    +------------------------+-----------------------------+
    | .. centered:: ``STL``  |  .. centered:: ``ST``       |
    +------------------------+                             |
    | .. centered:: ``STH``  |                             |
    +------------------------+-----------------------------+

.. container:: table-with-notes

   .. table:: XDNA2 operations that are required for our target tensor contraction workload. The operations are sorted by groups and provided together with their latencies.
      :name: isa:xdna2-isa
      :align: center
      :widths: 70 5 25

      +---------------------------------------------------------------+------+----------------------------------+
      | Operation                                                     | Lat. | Notes                            |
      +===============================================================+======+==================================+
      | **nop** - no-operation                                                                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``NOP``                                                       | \-   | do nothing                       |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``NOP(V|A|B|S|X|M|XM)``                                       | \-   | do nothing in unit               |
      +---------------------------------------------------------------+------+----------------------------------+
      | **mov - move**                                                                                          |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``MOV <Rd>, #<imm10>``                                        | 1    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``MOV <Rd>, <Rm>``                                            | 1    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``MOV(A|X) <Rd>, #<imm8>``                                    | 1    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``MOV(A|X) <Rd>, <Rm>``                                       | 1    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``MOVXM <Rd>, #<imm32>``                                      | 1    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VMOV <Xd>, <Xm>``                                           | 2    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VMOV (<BMLLd>|<BMLHd>|<BMHLd>|<BMHHd>), <Xm>``              | 2    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | **ld - load** - 4 byte load                                                                             |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``LD(A|B) <Rd>, [<Pn>], #<imm8>``                             | 6    | post-index                       |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``LD(A|B) <Rd>, [<Pn>], <Mm>``                                | 6    | post-index                       |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``LD(A|B) <Rd>, [<Pn>, #<imm8>]``                             | 6    | offset                           |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``LD(A|B) <Rd>, [<Pn>, <DJm>]``                               | 6    | offset                           |
      +---------------------------------------------------------------+------+----------------------------------+
      | **vld - vector load** - 64 byte loads                                                                   |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VLD(A|B) <Xd>, [<Pn>], #<imm8>``                            | 7    | post-index\ :sup:`a`             |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VLDA (<BMLLd>|<BMLHd>|<BMHLd>|<BMHHd>), [<Pn>], #<imm8>``   | 7    | post-index\ :sup:`a`             |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VLDA.CONV.FP32.BF16 (<CMLd>|<CMHd>), [<Pn>], #<imm8>``      | 7    | post-index\ :sup:`a`             |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VLDA.FILL.512 [P0, LF0, R24]``                              | \-   |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VLDB.FILL.512 [P1, LF1, R25]``                              | \-   |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VLDA.POP.576 <EXd>, [P0, LF0, R24]``                        | 8    | no std. vld follow\ :sup:`b`     |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VLDB.POP.576 <EXd>, [P1, LF1, R25]``                        | 8    | no std. vld follow\ :sup:`b`     |
      +---------------------------------------------------------------+------+----------------------------------+
      | **st - store** - 4 byte store                                                                           |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``ST <Rd>, [<Pn>], #<imm8>``                                  | 6    | post-index\ :sup:`a`             |
      +---------------------------------------------------------------+------+----------------------------------+
      | **vst - vector store** - 64 byte store                                                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VST (<WLd>|<WHd>), [<Pn>], #<imm8>``                        | 2    | post-index\ :sup:`a`             |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VST (<BMLLd>|<BMLHd>|<BMHLd>|<BMHHd>), [<Pn>], #<imm8>``    | 2    | post-index\ :sup:`a`             |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VST.CONV.BF16.FP32 (<CMLd>|<CMHd>), [<Pn>], #<imm8>``       | 2    | post-index\ :sup:`a`             |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VST.PUSH.576.CONV.BFP16EBS8.FP32 <DMd>, [P2, SF, R26]``     | \-   |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VST.FLUSH.512.CONV [P2, SF, R26]``                          | 2    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | **add - addition**                                                                                      |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``ADD <Rd>, <Rm>, #<imm7>``                                   | 1    | exp. leading bit\ :sup:`c`       |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``ADD <Rd>, <Rm>, <Rn>``                                      | 1    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | **padd - pointer addition**                                                                             |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``PADD(A|B|S) [<Pd>], <Mn>``                                  | 1    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``PADD(A|S) [<Pd>], #<imm11>``                                | 1    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``PADDB [<Pd>], #<imm10>``                                    | 1    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | **mul - multiplication**                                                                                |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``MUL <Rd>, <Rm>, <Rn>``                                      | 2    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | **vmac - vector multiply accumulate**                                                                   |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``VMAC.F <DMd>, <DMm>, <EXr>, <EXs>, <Rn>``                   | 6    | lfw, conf. reg.\ :sup:`d`        |
      +---------------------------------------------------------------+------+----------------------------------+
      | **comparisons**                                                                                         |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``(GT|LT|GE|LE){U} <Rd>, <Rm>, <Rn>``                         | 1    |  ``U`` for unsigned              |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``SEL.(EQZ|NEZ) <Rd>, <Rm>, <Rn>, <R27>``                     | 1    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | **j - jump**                                                                                            |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``J #<label>``                                                | 6    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``J(Z|NZ) <Rd>, #<label>``                                    | 6    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+
      | ``RET <LR>``                                                  | 6    |                                  |
      +---------------------------------------------------------------+------+----------------------------------+

   | :sup:`a` same addressing modes as ld
   | :sup:`b` Cannot be followed by a standard (non-pipeline) load operation.
   | :sup:`c` Leading bit is expanded.
   | :sup:`d` ``<DMm>`` is read in fourth cycle. ``<Rn>`` as configuration register (value 780 for ``8x8x8-bfp16``)