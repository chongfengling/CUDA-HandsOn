# Single-thread kernel
nvcc -DKERNEL_SINGLE vector_add_kernels.cu -o vector_add_single

# Single-block multi-thread kernel
nvcc -DKERNEL_MULTITHREAD vector_add_kernels.cu -o vector_add_multithread

# Multi-block multi-thread kernel
nvcc -DKERNEL_MULTIBLOCK vector_add_kernels.cu -o vector_add_multiblock

# 1. Single-thread kernel
nsys profile -o report_single ./vector_add_single

# 2. Single-block multi-thread kernel
nsys profile -o report_multithread ./vector_add_multithread

# 3. Multi-block multi-thread kernel
nsys profile -o report_multiblock ./vector_add_multiblock

# 1. Single-thread kernel
nsys stats --format=csv --timeunit=ms report_single.nsys-rep > kernel_summary_single.csv

# 2. Single-block multi-thread kernel
nsys stats --format=csv --timeunit=ms report_multithread.nsys-rep > kernel_summary_multithread.csv

# 3. Multi-block multi-thread kernel
nsys stats --format=csv --timeunit=ms report_multiblock.nsys-rep > kernel_summary_multiblock.csv


# result

# Single-thread kernel PASSED
# Kernel execution time: 5873.414551 ms
# Single-block multi-thread kernel PASSED
# Kernel execution time: 135.324295 ms
# Multi-block multi-thread kernel PASSED
# Kernel execution time: 1.659328 ms
