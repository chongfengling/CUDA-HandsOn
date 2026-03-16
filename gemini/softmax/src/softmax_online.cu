#include "softmax.h"
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ==========================================
// Part 1: 数据结构定义
// ==========================================

// 我们需要同时维护 Max 和 Sum，因为它们是耦合的
struct __align__(8) OnlineStat {
    float max_val;
    float sum;
};

// ==========================================
// Part 2: 核心数学逻辑 (请填充)
// ==========================================

// Helper: 合并两个 Stat (比如: 我手里的 + 邻居传来的)
// 数学提示: 
// new_max = max(a.max, b.max)
// new_sum = a.sum * exp(a.max - new_max) + b.sum * exp(b.max - new_max)
__inline__ __device__ OnlineStat combine_stat(OnlineStat a, OnlineStat b) {
    OnlineStat res;
    
    // TODO: 1. 计算 res.max_val
    res.max_val = fmaxf(a.max_val, b.max_val);

    // TODO: 2. 计算 res.sum (利用 Online Softmax 的修正公式)
    res.sum = a.sum * expf(a.max_val - res.max_val) + b.sum * expf(b.max_val - res.max_val);

    return res;
}

// ==========================================
// Part 3: Warp 级通信 (请填充)
// ==========================================

// Helper: Warp 内归约一个结构体
// 注意: __shfl_down_sync 只能传 32位(float/int)，不能直接传 struct
// 你需要分别 shuffle max_val 和 sum
__inline__ __device__ OnlineStat warpReduceStat(OnlineStat val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        OnlineStat other;
        
        // TODO: 3. 使用 __shfl_down_sync 获取 offset 处的 max_val 和 sum
        other.max_val = __shfl_down_sync(0xffffffff, val.max_val, offset);
        other.sum = __shfl_down_sync(0xffffffff, val.sum, offset);

        // TODO: 4. 调用 combine_stat 合并 val 和 other
        val = combine_stat(val, other);
    }
    return val;
}

// ==========================================
// Part 4: Kernel 实现 (请填充)
// ==========================================

__global__ void softmax_online_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x;
    int laneId = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    
    int row = blockIdx.x;
    float* input_row = input + row * N;
    float* output_row = output + row * N;

    // Shared Memory 用于存放每个 Warp 的归约结果
    // 因为是存放结构体，大小是 8 * sizeof(OnlineStat)
    __shared__ OnlineStat sdata[BLOCK_SIZE / WARP_SIZE];

    // --- Pass 1: Global Reduction (只读一次内存，同时算 Max 和 Sum) ---
    
    // 初始化状态: Max 为负无穷, Sum 为 0
    OnlineStat local_state = {-INFINITY, 0.0f};

    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float x = input_row[i];
        
        // TODO: 5. 实现线程内的 Online 更新逻辑
        // naive: 
        // local_state = combine_stat(local_state, {x, 1.0f});

        // 优化: 手动展开单元素更新，将每步的 2 次 expf 降低为 1 次
        if (x > local_state.max_val) {
            // 发现新的最大值，按新最大值缩放之前的 sum，当前元素的 exp(x-x) 就是 1.0f
            local_state.sum = local_state.sum * expf(local_state.max_val - x) + 1.0f;
            local_state.max_val = x;
        } else {
            // 最大值没变，当前元素按原最大值计算 exp 并累加
            local_state.sum += expf(x - local_state.max_val);
        }
    }

    // --- Pass 1.2: Warp Level Reduction ---
    
    // TODO: 6. 调用 warpReduceStat
    local_state = warpReduceStat(local_state);

    // --- Pass 1.3: Block Level Reduction (Warp 间通信) ---
    
    // 让每个 Warp 的 Leader 把结果写到 Shared Memory
    if (laneId == 0) {
        sdata[warpId] = local_state;
    }
    __syncthreads();

    // 由 Warp 0 聚合所有 Warp 的结果
    OnlineStat row_state = {-INFINITY, 0.0f};
    if (warpId == 0) {
        // 读取 Shared Memory (处理 Padding)
        // 如果 tid < 8, 读 sdata[tid]; 否则初始化为 {-INF, 0}
        OnlineStat val = (tid < (BLOCK_SIZE / WARP_SIZE)) ? sdata[tid] : (OnlineStat){-INFINITY, 0.0f};
        
        // TODO: 7. 再次调用 warpReduceStat 得到最终结果
        row_state = warpReduceStat(val);
    }

    // --- Pass 1.4: Broadcast ---
    // 将最终的 global_max 和 global_sum 广播给所有人
    
    // 这种写法比较 trick，复用 sdata[0] 的空间来广播 float
    // 或者你可以定义 __shared__ float final_max, final_sum;
    if (tid == 0) sdata[0] = row_state;
    __syncthreads();
    
    float global_max = sdata[0].max_val;
    float global_sum = sdata[0].sum;

    // ============================================
    // Pass 2: Normalize (写回结果)
    // ============================================
    
    // 这一步和之前一样，因为必须要写 output，所以必须再遍历一次
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        // TODO: 8. 计算最终结果并写入显存
        output_row[i] = expf(input_row[i] - global_max) / global_sum;
    }
}

void launch_softmax_online(float* input, float* output, int M, int N) {
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    softmax_online_kernel<<<grid, block>>>(input, output, N);
}