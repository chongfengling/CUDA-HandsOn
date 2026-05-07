#pragma once

void launch_softmax_naive(const float* input, float* output, int M, int N);
void launch_softmax_warp(const float* input, float* output, int M, int N);
void launch_softmax_online(const float* input, float* output, int M, int N);
