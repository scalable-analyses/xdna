Overview
========

In early 2023, AMD unveiled the Phoenix system-on-a-chip (SoC) under the product name `Ryzen 7040 Series <https://web.archive.org/web/20260121164508/https://ir.amd.com/news-events/press-releases/detail/1111/amdextends-its-leadership-with-the-introduction-of-its-broadest-portfolio-of-high-performance-pc-products-for-mobile-and-desktop>`__.
Phoenix is AMD's first SoC combining an x86-64 CPU with a neural processing unit (NPU).
In June 2024, AMD introduced the second generation of the NPU as part of the Strix Point SoC, branded as the `Ryzen AI 300 Series <https://web.archive.org/web/20260121164532/https://www.amd.com/en/newsroom/press-releases/2024-6-2-amd-unveils-next-gen-zen-5-ryzen-processors-to-p.html>`__.
The NPU in Phoenix uses the XDNA1 microarchitecture, while Strix Point uses XDNA2.
Strix Halo (`Ryzen AI Max <https://web.archive.org/web/20260121164553/https://ir.amd.com/news-events/press-releases/detail/1232/amd-announces-expanded-consumer-and-commercial-ai-pc-portfolio-at-ces>`__) was announced in January 2025.
It features an XDNA2 NPU and represents, as of 2026, the highest-performing iteration of AMD's SoCs for the AI PC market.

This website documents the key features of the XDNA microarchitectures and their utilization in high-performance tensor contractions.
We are actively working on the descriptions and examples; therefore, some sections remain under development and may contain errors.
The currently available chapters cover background information, microbenchmarks, and assembly kernels.

XDNA
----
XDNA1 and XDNA2 differ significantly from conventional CPU or GPU architectures.
While some of the individual architectural properties can also be found elsewhere, it is the combination of features that makes the microarchitectures unique.
XDNA1 and XDNA2 expose their design explicitly to software, containing minimal logic to abstract away hardware complexity.
With good software utilization, this yields more efficient hardware because more die area can be allocated to computational throughput.
From a software perspective, this approach also improves microarchitectural predictability by reducing hardware-assisted abstractions.
The trade-off of the XDNA approach is the required integration of low-level hardware properties into the software stack.

The most important hardware properties of XDNA1 and XDNA2 are the following:

Spatial Dataflow Architecture
  The XDNA microarchitectures consist of tiles arranged in a two-dimensional grid.
  There are three types of tiles: compute tiles, memory tiles, and shim tiles.
  Only compute tiles have cores and can perform floating- and fixed-point operations.
  The compute and memory tiles have scratchpad memory, while shim tiles can access main memory.
  The tiles are connected via a network-on-chip (NoC) that is configured by switch boxes.
  Each tile has direct memory access, which moves data through the NoC between tiles.
  The strided data layout of a tensor can be changed when communicating from one tile to another.

Very Long Instruction Word
  The compute tile cores are programmed using a very long instruction word (VLIW) instruction set architecture.
  Each VLIW instruction can contain up to six operations, one for each of the cores' six function units.
  The cores issue an instruction in every clock cycle and do not contain any stall logic.
  Data dependencies between instructions must be resolved in software.

Matrix Instructions
  Each compute tile core has a vector unit that can perform matrix multiplications.
  On XDNA1, the best-performing floating-point operation is a *BF16* matrix multiplication with an execution throughput of 256 floating-point operations (FLOPs) per clock cycle.
  On XDNA2, the best-performing operation is a *BFP16* matrix multiplication with an execution throughput of 1024 FLOPs per clock cycle.

Block Floating Point (BFP) Formats
  XDNA2 supports the *BFP16* data type.
  Compared to BF16, BFP16 reduces the data size per value by 1.8×.
  It uses a common 8-bit exponent for every group of eight values, each of which has an individual sign bit and a 7-bit mantissa.
  XDNA2 also supports converting an 8×8 FP32 matrix to an 8×8 BFP16 matrix in a single operation.

Pointers
--------

The source code of the examples and this website are available on `GitHub <https://github.com/scalable-analyses/xdna>`__.
For questions or feedback, join our `#scalable:uni-jena.de <https://matrix.to/#/#scalable:uni-jena.de>`__ Matrix room or submit an `issue <https://github.com/scalable-analyses/xdna/issues>`__.

Links to external resources:

* AMD
   * `AI Engine Kernel and Graph Programming Guide (UG1079) <https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding>`__
   * `IRON API and MLIR-based AI Engine Toolchain <https://github.com/Xilinx/mlir-aie>`__
   * `Micro 2025, AMD Versal AI Edge Series Gen 2 <https://www.computer.org/csdl/magazine/mi/2025/03/10926865/2558g2yWH5e>`__
   * `HC2024, AMD Versal AI Edge Series Gen 2 for Vision and Automotive <https://hc2024.hotchips.org/assets/program/conference/day2/21_HC2024.AMD.KnoppChu.final.pdf>`__
   * `HC33, Xilinx Edge Processors <https://hc33.hotchips.org/assets/program/conference/day2/XilinxHotChips2021-v0.90-final.pdf>`__
* Chips and Cheese
   * `AMD’s Chiplet APU: An Overview of Strix Halo <https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix>`__
   * `Evaluating the Infinity Cache in AMD Strix Halo <https://chipsandcheese.com/p/evaluating-the-infinity-cache-in>`__
   * `Strix Halo’s Memory Subsystem: Tackling iGPU Challenges <https://chipsandcheese.com/p/strix-halos-memory-subsystem-tackling>`__
* Gentoo Linux
   * `User:Lockal/AMDXDNA <https://wiki.gentoo.org/wiki/User:Lockal/AMDXDNA>`__

.. toctree::
   :maxdepth: 1
   :hidden:

   isa
   xdna1_kernel
   xdna2_kernel
