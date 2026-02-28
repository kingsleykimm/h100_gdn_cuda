#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <gdn_cuda/error.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace test_utils {

inline std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i + 1 < shape.size())
            ss << ", ";
    }
    ss << ")";
    return ss.str();
}

inline void inspect_tensor(const torch::Tensor& tensor, int n) {
    std::cout << tensor.flatten().slice(0, 0, n) << std::endl;
}

inline void inspect_tensor(const torch::Tensor& tensor, int s, int e) {
    std::cout << tensor.flatten().slice(0, s, e) << std::endl;
}

inline bool check_tensor_close(const torch::Tensor& t1, const torch::Tensor& t2, float atol,
                               float rtol) {
    bool pass = torch::allclose(t1, t2, rtol, atol);
    auto a = t1.flatten();
    auto b = t2.flatten();
    if (!pass) {
        int idx = torch::abs(a - b).argmax().item<int>();
        std::cout << "\033[0;31mTensor mismatch\033[0m\n";
        printf("\033[0;31mShape1: %s, Strides1: %s\033[0m\n",
               shape_to_string(t1.sizes().vec()).c_str(),
               shape_to_string(t1.strides().vec()).c_str());
        printf("\033[0;31mShape2: %s, Strides2: %s\033[0m\n",
               shape_to_string(t2.sizes().vec()).c_str(),
               shape_to_string(t2.strides().vec()).c_str());
        std::cout << "\033[0;31mMax diff: " << torch::abs(a - b).max().item<float>() << "\033[0m\n";
        std::cout << "\033[0;31mAvg diff: " << torch::abs(a - b).mean().item<float>()
                  << "\033[0m\n";
        std::cout << "\033[0;31mAt index " << idx << ": ref=" << a[idx].item<float>()
                  << " got=" << b[idx].item<float>() << "\033[0m\n";
    } else {
        int idx = torch::abs(a - b).argmax().item<int>();
        std::cout << "\033[0;32mTensor passed\033[0m\n";
        printf("\033[0;32mShape1: %s, Strides1: %s\033[0m\n",
               shape_to_string(t1.sizes().vec()).c_str(),
               shape_to_string(t1.strides().vec()).c_str());
        printf("\033[0;32mShape2: %s, Strides2: %s\033[0m\n",
               shape_to_string(t2.sizes().vec()).c_str(),
               shape_to_string(t2.strides().vec()).c_str());
        std::cout << "\033[0;32mMax diff: " << torch::abs(a - b).max().item<float>() << "\033[0m\n";
        std::cout << "\033[0;32mAvg diff: " << torch::abs(a - b).mean().item<float>()
                  << "\033[0m\n";
        std::cout << "\033[0;32mAt index " << idx << ": ref=" << a[idx].item<float>()
                  << " got=" << b[idx].item<float>() << "\033[0m\n";
    }
    return pass;
}

inline double calc_diff(torch::Tensor& ref, torch::Tensor& out) {
    auto r = ref.to(torch::kDouble);
    auto o = out.to(torch::kDouble);
    double denom = (r * r + o * o).sum().item<double>();
    return 1.0 - (2.0 * (r * o).sum().item<double>()) / denom;
}

// Build chunk indices vector from cu_seqlens for varlen kernels.
// Returns interleaved [batch_idx, chunk_idx_within_batch, ...] pairs.
inline std::vector<int> prepare_chunk_indices(const std::vector<int>& cu_seqlens,
                                              int chunk_size = 64) {
    std::vector<int> chunk_indices;
    for (int b = 0; b + 1 < (int)cu_seqlens.size(); ++b) {
        int len = cu_seqlens[b + 1] - cu_seqlens[b];
        int num_chunks = (len + chunk_size - 1) / chunk_size;
        for (int c = 0; c < num_chunks; ++c) {
            chunk_indices.push_back(b);
            chunk_indices.push_back(c);
        }
    }
    return chunk_indices;
}

// Build cumulative chunk counts: cu_chunks[b+1] = cu_chunks[b] + num_chunks_in_seq_b
inline std::vector<int> prepare_cu_chunks(const std::vector<int>& cu_seqlens, int chunk_size = 64) {
    std::vector<int> cu_chunks;
    cu_chunks.push_back(0);
    for (int b = 0; b + 1 < (int)cu_seqlens.size(); ++b) {
        int len = cu_seqlens[b + 1] - cu_seqlens[b];
        int num_chunks = (len + chunk_size - 1) / chunk_size;
        cu_chunks.push_back(cu_chunks.back() + num_chunks);
    }
    return cu_chunks;
}

// Build at::Tensor cu_seqlens on GPU from std::vector<int>
inline torch::Tensor make_cu_seqlens(const std::vector<int>& vec) {
    return torch::tensor(vec, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}

// Build chunk_indices at::Tensor on GPU
inline torch::Tensor make_chunk_indices(const std::vector<int>& cu_seqlens, int chunk_size) {
    auto v = prepare_chunk_indices(cu_seqlens, chunk_size);
    return torch::tensor(v, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}

// Build cu_chunks at::Tensor on GPU
inline torch::Tensor make_cu_chunks(const std::vector<int>& cu_seqlens, int chunk_size) {
    auto v = prepare_cu_chunks(cu_seqlens, chunk_size);
    return torch::tensor(v, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}

}  // namespace test_utils
