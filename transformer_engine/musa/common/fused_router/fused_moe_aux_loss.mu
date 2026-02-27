/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <musa_runtime.h>
#include <transformer_engine/fused_router.h>

#include "../common.h"
#include "../util/logging.h"
#include "../util/musa_runtime.h"
#include "../utils.muh"
#include "common/util/musa_runtime.h"
#include "utils.h"

namespace transformer_engine {

using CompType = float;

template <typename DataType, typename IndexType>
__global__ void fused_moe_aux_loss_forward_reduce_kernel(const DataType* probs,
                                                         const IndexType* tokens_per_expert,
                                                         int num_rows, int num_cols,
                                                         float* partial_buf) {
  int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
  int threads_per_block = blockDim.x * blockDim.y;
  int warp_num = threads_per_block / kThreadsPerWarp;
  int warp_id = linear_tid / kThreadsPerWarp;
  int lane_id = linear_tid % kThreadsPerWarp;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ CompType warp_sums[32];

  CompType thread_sum = 0.0f;
  if (col < num_cols) {
    CompType token_weight = static_cast<CompType>(tokens_per_expert[col]);
    for (int r = row; r < num_rows; r += gridDim.y * blockDim.y) {
      thread_sum += static_cast<CompType>(probs[r * num_cols + col]) * token_weight;
    }
  }

  for (int s = 16; s > 0; s /= 2) {
    thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, s);
  }
  if (lane_id == 0) {
    warp_sums[warp_id] = thread_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    CompType block_sum = lane_id < warp_num ? warp_sums[lane_id] : 0.0f;
    for (int s = 16; s > 0; s /= 2) {
      block_sum += __shfl_xor_sync(0xffffffff, block_sum, s);
    }
    if (lane_id == 0) {
      int block_linear_idx = blockIdx.y * gridDim.x + blockIdx.x;
      partial_buf[block_linear_idx] = static_cast<float>(block_sum);
    }
  }
}

template <typename DataType>
__global__ void fused_moe_aux_loss_forward_finalize_kernel(int total_num_tokens, int num_experts,
                                                           int topk, float coeff, int partial_count,
                                                           const float* partial_buf,
                                                           DataType* aux_loss,
                                                           float* Const_buf) {
  extern __shared__ float shmem[];
  int tid = threadIdx.x;
  float local_sum = 0.0f;
  for (int i = tid; i < partial_count; i += blockDim.x) {
    local_sum += partial_buf[i];
  }
  shmem[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shmem[tid] += shmem[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float tokens = static_cast<float>(total_num_tokens);
    float C_coeff = static_cast<float>(num_experts) * coeff /
                    (static_cast<float>(topk) * tokens * tokens);
    aux_loss[0] = static_cast<DataType>(shmem[0] * C_coeff);
    Const_buf[0] = C_coeff;
  }
}

template <typename DataType, typename IndexType>
void fused_moe_aux_loss_forward_kernel_launcher(const DataType* probs,
                                                const IndexType* tokens_per_expert,
                                                int total_num_tokens, int num_experts, int num_rows,
                                                int num_cols, int topk, float coeff,
                                                DataType* aux_loss, float* Const_buf,
                                                musaStream_t stream) {
  constexpr int threads_x = 128;
  constexpr int threads_y = 2;
  dim3 block(threads_x, threads_y);
  int grid_x = (num_cols + threads_x - 1) / threads_x;
  int grid_y = (num_rows + threads_y - 1) / threads_y;
  int max_grid_y = cuda::sm_count() * 8;
  if (max_grid_y < 1) {
    max_grid_y = 1;
  }
  if (grid_y > max_grid_y) {
    grid_y = max_grid_y;
  }
  dim3 grid(grid_x, grid_y);
  int partial_count = grid_x * grid_y;
  constexpr int finalize_block = 256;

  fused_moe_aux_loss_forward_reduce_kernel<DataType, IndexType>
      <<<grid, block, 0, stream>>>(probs, tokens_per_expert, num_rows, num_cols, Const_buf);
  fused_moe_aux_loss_forward_finalize_kernel<DataType>
      <<<1, finalize_block, finalize_block * sizeof(float), stream>>>(
          total_num_tokens, num_experts, topk, coeff, partial_count, Const_buf, aux_loss, Const_buf);
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_moe_aux_loss_forward(const Tensor& probs, const Tensor& tokens_per_expert,
                                int total_num_tokens, int num_experts, int num_rows, int num_cols,
                                int topk, float coeff, Tensor& aux_loss, Tensor& Const_buf,
                                musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      probs.data.dtype, DataType,
      TE_ROUTER_INDEX_TYPE_SWITCH_ALL(
          tokens_per_expert.data.dtype, IndexType,
          fused_moe_aux_loss_forward_kernel_launcher<DataType, IndexType>(
              reinterpret_cast<DataType*>(probs.data.dptr),
              reinterpret_cast<IndexType*>(tokens_per_expert.data.dptr), total_num_tokens,
              num_experts, num_rows, num_cols, topk, coeff,
              reinterpret_cast<DataType*>(aux_loss.data.dptr),
              reinterpret_cast<float*>(Const_buf.data.dptr), stream);););
}

template <typename DataType, typename IndexType>
__global__ void fused_moe_aux_loss_backward_kernel(const float* Const_buf,
                                                   const IndexType* tokens_per_expert, int num_rows,
                                                   int num_cols, DataType* grad_aux_loss,
                                                   DataType* grad_probs) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  float scale = Const_buf[0] * static_cast<float>(grad_aux_loss[0]);

  if (col < num_cols) {
    float value = scale * static_cast<float>(tokens_per_expert[col]);
    for (int r = row; r < num_rows; r += gridDim.y * blockDim.y) {
      grad_probs[r * num_cols + col] = static_cast<DataType>(value);
    }
  }
}

template <typename DataType, typename IndexType>
void fused_moe_aux_loss_backward_kernel_launcher(const float* Const_buf,
                                                 const IndexType* tokens_per_expert, int num_rows,
                                                 int num_cols, DataType* grad_aux_loss,
                                                 DataType* grad_probs, musaStream_t stream) {
  // Meta data for the kernel
  constexpr int threads_x = 128;
  constexpr int threads_y = 2;
  dim3 block(threads_x, threads_y);
  int grid_x = (num_cols + threads_x - 1) / threads_x;
  int grid_y = (num_rows + threads_y - 1) / threads_y;
  int max_grid_y = cuda::sm_count() * 8;
  if (max_grid_y < 1) {
    max_grid_y = 1;
  }
  if (grid_y > max_grid_y) {
    grid_y = max_grid_y;
  }
  dim3 grid(grid_x, grid_y);
  fused_moe_aux_loss_backward_kernel<DataType, IndexType><<<grid, block, 0, stream>>>(
      Const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss, grad_probs);
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_moe_aux_loss_backward(const Tensor& Const_buf, const Tensor& tokens_per_expert,
                                 int num_rows, int num_cols, Tensor& grad_aux_loss,
                                 Tensor& grad_probs, musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_aux_loss.data.dtype, DataType,
      TE_ROUTER_INDEX_TYPE_SWITCH_ALL(
          tokens_per_expert.data.dtype, IndexType,
          fused_moe_aux_loss_backward_kernel_launcher<DataType, IndexType>(
              reinterpret_cast<float*>(Const_buf.data.dptr),
              reinterpret_cast<IndexType*>(tokens_per_expert.data.dptr), num_rows, num_cols,
              reinterpret_cast<DataType*>(grad_aux_loss.data.dptr),
              reinterpret_cast<DataType*>(grad_probs.data.dptr), stream);););
}

}  // namespace transformer_engine

void nvte_fused_moe_aux_loss_forward(const NVTETensor probs, const NVTETensor tokens_per_expert,
                                     int total_num_tokens, int num_experts, int num_rows,
                                     int num_cols, int topk, float coeff, NVTETensor aux_loss,
                                     NVTETensor Const_buf, musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_moe_aux_loss_forward);
  using namespace transformer_engine;
  fused_moe_aux_loss_forward(
      *reinterpret_cast<Tensor*>(probs), *reinterpret_cast<Tensor*>(tokens_per_expert), total_num_tokens,
      num_experts, num_rows, num_cols, topk, coeff, *reinterpret_cast<Tensor*>(aux_loss),
      *reinterpret_cast<Tensor*>(Const_buf), stream);
}

void nvte_fused_moe_aux_loss_backward(const NVTETensor Const_buf,
                                      const NVTETensor tokens_per_expert, int num_rows,
                                      int num_cols, NVTETensor grad_aux_loss, NVTETensor grad_probs,
                                      musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_moe_aux_loss_backward);
  using namespace transformer_engine;
  fused_moe_aux_loss_backward(*reinterpret_cast<Tensor*>(Const_buf),
                              *reinterpret_cast<Tensor*>(tokens_per_expert), num_rows, num_cols,
                              *reinterpret_cast<Tensor*>(grad_aux_loss),
                              *reinterpret_cast<Tensor*>(grad_probs), stream);
}
