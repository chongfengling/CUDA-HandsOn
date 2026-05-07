#pragma once

enum class AddAlgo {
    NAIVE_SINGLE_THREAD,    // One thread for all elements
    SINGLE_BLOCK_MULTI,     // One block, multiple threads
    MULTI_BLOCK_MULTI       // Grid-stride loop (Standard optimized)
};

/**
 * @brief Launch vector addition: C = A + B
 * 
 * @param a    Input vector A (device pointer)
 * @param b    Input vector B (device pointer)
 * @param c    Output vector C (device pointer)
 * @param n    Number of elements
 * @param algo Algorithm version to use
 */
void launch_vector_add(const float* a, const float* b, float* c, int n, AddAlgo algo = AddAlgo::MULTI_BLOCK_MULTI);

// implementation launchers
void launch_vector_add_naive(const float* a, const float* b, float* c, int n);
void launch_vector_add_single_block(const float* a, const float* b, float* c, int n);
void launch_vector_add_multi_block(const float* a, const float* b, float* c, int n);
