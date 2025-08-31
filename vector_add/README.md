# GPU Vector Addition

## Overview

This project demonstrates **vector addition on NVIDIA GPUs** using CUDA with different parallelization strategies:

1. **Single-thread kernel** – one thread processes the entire vector.
2. **Single-block multi-thread kernel** – multiple threads in a single block process the vector.
3. **Multi-block multi-thread kernel** – multiple threads across blocks fully utilize the GPU.

It also measures **GPU memory usage** and **kernel execution time**.

---

## Environment

* **GPU:** NVIDIA GeForce RTX 5080
* **CUDA Toolkit:** release 12.9, V12.9.41
* **Compiler:** gcc 13.3.0 (Ubuntu 24.04)
* **Nsight Systems:** 2025.1.3.140-251335620677v0
* **Nsight Compute:** Version 2025.2.0.0 (public-release build 35613519)

---

## Features

* Three CUDA kernels for vector addition.
* Verification of results on host.
* GPU memory usage reporting.
* Kernel execution time measurement.
* Compile-time selection of kernel via macros:
  `KERNEL_SINGLE`, `KERNEL_MULTITHREAD`, `KERNEL_MULTIBLOCK`.

---

## Usage

```bash
bash run.sh
```

---

## Performance Example

| Kernel Type                      | Execution Time (ms) |
| -------------------------------- | ------------------- |
| Single-thread kernel             | 5873.41             |
| Single-block multi-thread kernel | 135.32              |
| Multi-block multi-thread kernel  | 1.66                |

---

## References

* [CUDA Tutorial](https://cuda-tutorial.readthedocs.io/en/latest/) – official CUDA learning resources
* [ChatGPT](https://www.chatgpt.com) – for guidance and explanations

