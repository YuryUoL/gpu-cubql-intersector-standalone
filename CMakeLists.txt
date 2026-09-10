===============================================================================
SYSTEM REQUIREMENTS & TESTED ENVIRONMENT
===============================================================================

1. SOFTWARE REQUIREMENTS
-------------------------------------------------------------------------------
* Operating System : Linux (Fedora / Ubuntu recommended)
* CUDA Toolkit     : >= 13.0 (Required for C++17/20 host-device __int128 support
                     and sm_120 architecture support; older versions will produce 
                     compilation errors with __int128 types)
* Host Compiler    : GCC >= 11.0 or Clang >= 13.0 (C++17 standard support required)
* Build System     : CMake >= 3.22
* GPU Compute      : Compute Capability 7.5+ (Ampere/Ada/Blackwell recommended)

Libraries & Dependencies:
* NVIDIA cuBQL     : Built-in / Submodule
* CGAL             : >= 5.6 (EPECK Exact Predicates Inexact Constructions Kernel)
* Intel TBB        : Required for parallel CPU fallback filtering
* Polyscope        : Optional (Required for GUI interactive visualizer)


2. HARDWARE REQUIREMENTS
-------------------------------------------------------------------------------
* GPU              : NVIDIA Dedicated GPU with CUDA support
* Video Memory     : >= 4 GB VRAM (8+ GB recommended for large meshes like Thingi10K)
* System RAM       : >= 16 GB RAM
* Storage          : High-speed NVMe SSD recommended for fast mesh IO (.off / .obj)


3. TESTED SYSTEM SPECIFICATIONS (Reference Environment)
-------------------------------------------------------------------------------
The standalone engine and benchmarks were evaluated on the following setup:

* OS               : Fedora Linux 42 (Workstation Edition) (x86_64)
* Kernel           : Linux 6.19.14-108.fc42.x86_64
* CPU              : AMD Ryzen 9 3950X (16 Cores / 32 Threads)
* GPU              : NVIDIA GeForce RTX 5060 Ti
* Compute Arch     : sm_120
* VRAM             : 16 GB GDDR6 (16311 MiB)
* System RAM       : 80 GB DDR4 (78 GiB)
* NVIDIA Driver    : 580.159.03
* CUDA Toolkit     : 13.0 (V13.0.88)
* Host Compiler    : GCC 15.2.1
* CMake            : 3.31.11


4. KNOWN COMPILATION & SETUP NOTES
-------------------------------------------------------------------------------
* __int128 Errors : Ensure CUDA 13.0+ is active in your PATH/NVCC environment.
                     Older CUDA toolkits trigger host/device header mismatch 
                     errors when parsing 128-bit integer definitions in C++17 mode.
* CMake Selection  : Point CMAKE_CUDA_COMPILER explicitly if multiple CUDA 
                     versions are installed:
                     $ cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc ..
===============================================================================
