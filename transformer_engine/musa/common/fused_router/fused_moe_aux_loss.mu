/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cooperative_groups.h>
#include <musa_runtime.h>
#include <transformer_engine/fused_router.h>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.muh"
#include "common/util/musa_runtime.h"
#include "utils.h"

namespace transformer_engine {

// Using Double to hanld all the calculations
using CompType = double;

template <typename DataType, typename IndexType>
__global__ void fused_moe_aux_loss_forward_kernel(const DataType* probs,
                                                  const IndexType* tokens_per_expert,
                                                  int total_num_tokens, int num_experts,
                                                  int num_rows, int num_cols, int topk, float coeff,
                                                  DataType* aux_loss, float* Const_buf) {
  // Use Only 1 block/1024 threads to avoid the grid sync
  if (blockIdx.x > 0) return;
  int warp_num = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem_aux_loss[];
  CompType* aggregated_probs_per_expert = reinterpret_cast<CompType*>(shmem_aux_loss);

  // Clear the shmem
  for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
    aggregated_probs_per_expert[i] = CompType(0);
  }
  __syncthreads();

  /**
     * Section: Reduce the probs to the aggregated_probs_per_expert
     */
  // Loop: for all positions in each row
  for (int i = lane_id; i < num_cols; i += kThreadsPerWarp) {
    CompType tmp = CompType(0);
    // Loop: for all rows that this warp is responsible for
    for (int j = warp_id; j < num_rows; j += warp_num) {
      tmp += CompType(probs[j * num_cols + i]);
    }
    atomicAdd(&aggregated_probs_per_expert[i], tmp);
  }
  __syncthreads();

  /**
     * Section: aggregated_probs_per_expert * tokens_per_expert
     * In-place update on shmem
     */
  for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
    aggregated_probs_per_expert[i] *= CompType(tokens_per_expert[i]);
  }
  __syncthreads();

  if (warp_id == 0) {
    /**
         * Section: Reduce to get the sum of aggregated_probs_per_expert
         */
    CompType intermediate_result =
        warp_reduce_on_shmem(aggregated_probs_per_expert, num_cols, ReduceFuncType::SUM, lane_id);
    __syncwarp();

    if (lane_id == 0) {
      /**
             * Section: Compute the aux_loss
             */
      float C_coeff = (num_experts * coeff) / topk / total_num_tokens / total_num_tokens;
      aux_loss[0] = static_cast<DataType>(static_cast<double>(intermediate_result) * C_coeff);
      Const_buf[0] = C_coeff;
    }
  }
}

template <typename DataType, typename IndexType>
void fused_moe_aux_loss_forward_kernel_launcher(const DataType* probs,
                                                const IndexType* tokens_per_expert,
                                                int total_num_tokens, int num_experts, int num_rows,
                                                int num_cols, int topk, float coeff,
                                                DataType* aux_loss, float* Const_buf,
                                                musaStream_t stream) {
  size_t smem_size = sizeof(CompType) * num_cols;
  fused_moe_aux_loss_forward_kernel<DataType, IndexType>
      <<<1, 1024, smem_size, stream>>>(probs, tokens_per_expert, total_num_tokens, num_experts,
                                          num_rows, num_cols, topk, coeff, aux_loss, Const_buf);
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
  int global_warp_num = gridDim.x * blockDim.x / kThreadsPerWarp;
  int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;

  // Loop: for all positions in each row
  for (int i = lane_id; i < num_cols; i += kThreadsPerWarp) {
    float C_coeff = Const_buf[0];
    double tokens_per_expert_i = static_cast<double>(tokens_per_expert[i]);
    double grad_aux_loss_value = static_cast<double>(grad_aux_loss[0]);
    // Loop: for all rows
    for (int j = global_warp_id; j < num_rows; j += global_warp_num) {
      grad_probs[j * num_cols + i] = C_coeff * tokens_per_expert_i * grad_aux_loss_value;
    }
  }
}

template <typename DataType, typename IndexType>
void fused_moe_aux_loss_backward_kernel_launcher(const float* Const_buf,
                                                 const IndexType* tokens_per_expert, int num_rows,
                                                 int num_cols, DataType* grad_aux_loss,
                                                 DataType* grad_probs, musaStream_t stream) {
  // Meta data for the kernel
  int block_size = 256;
  int grid_size = (num_rows + block_size - 1) / block_size;
  fused_moe_aux_loss_backward_kernel<DataType, IndexType><<<grid_size, block_size, 0, stream>>>(
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
