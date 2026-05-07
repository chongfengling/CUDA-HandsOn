#include "add.h"
#include <stdexcept>

void launch_vector_add(const float* a, const float* b, float* c, int n, AddAlgo algo) {
    switch(algo) {
        case AddAlgo::NAIVE_SINGLE_THREAD:
            launch_vector_add_naive(a, b, c, n);
            break;
        case AddAlgo::SINGLE_BLOCK_MULTI:
            launch_vector_add_single_block(a, b, c, n);
            break;
        case AddAlgo::MULTI_BLOCK_MULTI:
            launch_vector_add_multi_block(a, b, c, n);
            break;
        default:
            throw std::runtime_error("Unknown add algorithm");
    }
}
