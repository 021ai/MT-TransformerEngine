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
#include "utils.h"

namespace transformer_engine {

constexpr size_t kRouterShmemBudgetBytes = 48 * 1024;

inline int select_block_size_by_shmem(size_t per_token_shmem_bytes) {
  int block_size = 256;
  size_t shmem = per_token_shmem_bytes * static_cast<size_t>(block_size / kThreadsPerWarp);
  if (shmem <= kRouterShmemBudgetBytes) return block_size;
  block_size = 128;
  shmem = per_token_shmem_bytes * static_cast<size_t>(block_size / kThreadsPerWarp);
  if (shmem <= kRouterShmemBudgetBytes) return block_size;
  return 64;
}

template <typename DataType, bool kSoftmax, bool kScoresShared>
__global__ void fused_score_for_moe_aux_loss_forward_kernel(const DataType *logits, int num_tokens,
                                                            int num_experts, int topk,
                                                            DataType *scores, bool *routing_map,
                                                            DataType *intermediate_output) {
  /***
     * Section: Global Variables/Addresses init
     * - Assume the sizeof(DataType) >= sizeof(int),
     *   So DataType address is assigned firstly to avoid the alignment issue
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  // Used variables/addresses init
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem_scores_for_aux_loss[];
  DataType *logits_buf = reinterpret_cast<DataType *>(shmem_scores_for_aux_loss);
  int *topk_indices_buf =
      reinterpret_cast<int *>(logits_buf + num_experts * num_token_per_block);
  // The address of buffers on the current warp
  DataType *local_logits = logits_buf + warp_id * num_experts;
  int *topk_indices = topk_indices_buf + warp_id * topk;

  /***
     * Section: Main Loop
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    // Each warp is responsible for one token
    if (token_offset_cur_warp >= num_tokens) break;

    /***
         * Section: Init buffer
         * - Clear the global buffer which will accept the result of this round
         * - Clear/Init the shmem buffer used by current warp this round
         * - Load the logits to shmem
         */
    int pos_offset = token_offset_cur_warp * num_experts;
    // Load the logits to shmem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_logits[i] = logits[pos_offset + i];
    }
    __syncwarp();

    /***
         * Section: Preprocess
         * Possible preprocess the scores before the topk operation
         * - Pre-softmax
         * - Sigmoid
         * - Sigmoid post-processing when topk > 1
         * This is in-place scores update
         */
    if (kSoftmax) {
      // Apply softmax to the logits before the topk
      apply_softmax_on_float(local_logits, num_experts, lane_id);
      // Save the softmax output for backward
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
    } else {
      // Apply sigmoid to the logits
      apply_sigmoid_on_float(local_logits, num_experts, lane_id);
      __syncwarp();
      // Save the sigmoid output for backward
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
      if (topk > 1) {
        float sum_logits = static_cast<float>(
            warp_reduce_on_shmem(local_logits, num_experts, ReduceFuncType::SUM, lane_id));
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          local_logits[i] = static_cast<DataType>(static_cast<float>(local_logits[i]) /
                                                  (sum_logits + epsilon));
        }
      }
      __syncwarp();
    }

    // Write scores only when it doesn't share the same buffer with intermediate_output.
    if (!kScoresShared) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        scores[pos_offset + i] = local_logits[i];
      }
    }

    /***
         * Section: Topk
         * Get the topk indices
         */
    naive_topk_indices_inplace(local_logits, num_experts, topk, topk_indices, lane_id);

    // Write the routing_map to the output tensor
    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
    }
  }
}

template <typename DataType>
void fused_score_for_moe_aux_loss_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, int score_function,
    DataType *scores, bool *routing_map, DataType *intermediate_output, musaStream_t stream) {
  const size_t output_elements = static_cast<size_t>(num_tokens) * num_experts;
  NVTE_CHECK_CUDA(musaMemsetAsync(routing_map, 0, output_elements * sizeof(bool), stream));

  // Meta data for the kernel
  size_t per_token_shmem_bytes = num_experts * sizeof(DataType) + topk * sizeof(int);
  int block_size = select_block_size_by_shmem(per_token_shmem_bytes);
  size_t num_token_per_block = block_size / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t max_grid_size = static_cast<size_t>(cuda::sm_count()) * 8;
  if (max_grid_size < 1) max_grid_size = 1;
  if (grid_size > max_grid_size) grid_size = max_grid_size;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(DataType)  // logits
                              + topk * num_token_per_block * sizeof(int);           // topk_indices
  bool scores_shared = (scores == intermediate_output);
  if (score_function == 1) {
    if (scores_shared) {
      fused_score_for_moe_aux_loss_forward_kernel<DataType, true, true>
          <<<grid_size, block_size, shared_memory_size, stream>>>(
              logits, num_tokens, num_experts, topk, scores, routing_map, intermediate_output);
    } else {
      fused_score_for_moe_aux_loss_forward_kernel<DataType, true, false>
          <<<grid_size, block_size, shared_memory_size, stream>>>(
              logits, num_tokens, num_experts, topk, scores, routing_map, intermediate_output);
    }
  } else if (score_function == 0) {
    if (scores_shared) {
      fused_score_for_moe_aux_loss_forward_kernel<DataType, false, true>
          <<<grid_size, block_size, shared_memory_size, stream>>>(
              logits, num_tokens, num_experts, topk, scores, routing_map, intermediate_output);
    } else {
      fused_score_for_moe_aux_loss_forward_kernel<DataType, false, false>
          <<<grid_size, block_size, shared_memory_size, stream>>>(
              logits, num_tokens, num_experts, topk, scores, routing_map, intermediate_output);
    }
  } else {
    NVTE_ERROR("Invalid score_function.");
  }
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_score_for_moe_aux_loss_forward(const Tensor &logits, int num_tokens, int num_experts,
                                          int topk, int score_function, Tensor &scores,
                                          Tensor &routing_map, Tensor &intermediate_output,
                                          musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      logits.data.dtype, DataType,
      fused_score_for_moe_aux_loss_forward_kernel_launcher<DataType>(
          reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,
          score_function, reinterpret_cast<DataType *>(scores.data.dptr),
          reinterpret_cast<bool *>(routing_map.data.dptr),
          reinterpret_cast<DataType *>(intermediate_output.data.dptr), stream););
}

template <typename DataType, bool kSoftmax>
__global__ void fused_score_for_moe_aux_loss_backward_kernel(const DataType *intermediate_output,
                                                             const DataType *grad_scores,
                                                             int num_tokens, int num_experts,
                                                             int topk, DataType *grad_logits) {
  /***
     * Section: Global Variables/Addresses init
     * - Assume the sizeof(DataType) >= sizeof(int),
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  // Used variables/addresses init
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];
  DataType *grad_scores_buf = reinterpret_cast<DataType *>(shmem);
  // To store the output of softmax/sigmoid from the fwd
  DataType *act_from_fwd_buf =
      reinterpret_cast<DataType *>(grad_scores_buf + num_experts * num_token_per_block);
  DataType *comp_buf =
      reinterpret_cast<DataType *>(act_from_fwd_buf + num_experts * num_token_per_block);
  // The address of buffers on the current warp
  DataType *local_grad = grad_scores_buf + warp_id * num_experts;
  DataType *local_act_from_fwd = act_from_fwd_buf + warp_id * num_experts;
  DataType *local_comp_buf = comp_buf + warp_id * num_experts;

  /***
     * Section: Main Loop
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    // Each warp is responsible for one token
    if (token_offset_cur_warp >= num_tokens) break;

    /***
         * Section: Init buffer
         * - Clear the global buffer which will accept the result of this round
         * - Clear/Init the shmem buffer used by current warp this round
         * - Load the dgrad/output_from_fwd to shmem
         */
    int pos_offset = token_offset_cur_warp * num_experts;
    // Load the dgrad/output_from_fwd to shmem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_grad[i] = grad_scores[pos_offset + i];
      local_act_from_fwd[i] = intermediate_output[pos_offset + i];
    }
    __syncwarp();

    /***
         * Section: Backward of ops before the topk
         * - Pre-softmax bwd
         * - Sigmoid Post-processing bwd when topk > 1
         * - Sigmoid bwd
         * - Write the grad_logits to the global mem
         */
    if (!kSoftmax) {
      // Sigmoid Post-processing bwd when topk > 1
      if (topk > 1) {
        float sum_fwd_input = static_cast<float>(
            warp_reduce_on_shmem(local_act_from_fwd, num_experts, ReduceFuncType::SUM, lane_id));
        // Put the result of output * grad to the comp_buf
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          local_comp_buf[i] = static_cast<DataType>(static_cast<float>(local_grad[i]) *
                                                    static_cast<float>(local_act_from_fwd[i]));
        }
        __syncwarp();
        float sum_Output_x_Grad = static_cast<float>(
            warp_reduce_on_shmem(local_comp_buf, num_experts, ReduceFuncType::SUM, lane_id));
        // In-place update
        float norm_inv = 1.0f / (sum_fwd_input + epsilon);
        float corr = sum_Output_x_Grad * norm_inv * norm_inv;
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          local_grad[i] =
              static_cast<DataType>(static_cast<float>(local_grad[i]) * norm_inv - corr);
        }
      }
      apply_sigmoid_bwd_on_float(local_grad, local_act_from_fwd, num_experts, lane_id);
    } else {
      // Pre-softmax bwd
      apply_softmax_bwd_on_float(local_grad, local_act_from_fwd, local_comp_buf, nullptr,
                                 num_experts, lane_id);
    }
    // Write the grad_logits to the global mem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      grad_logits[pos_offset + i] = local_grad[i];
    }
  }
}

template <typename DataType>
void fused_score_for_moe_aux_loss_backward_kernel_launcher(
    const DataType *intermediate_output, const DataType *grad_scores, int num_tokens,
    int num_experts, int topk, int score_function, DataType *grad_logits, musaStream_t stream) {
  // Meta data for the kernel
  size_t per_token_shmem_bytes = 3 * num_experts * sizeof(DataType);
  int block_size = select_block_size_by_shmem(per_token_shmem_bytes);
  size_t num_token_per_block = block_size / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t max_grid_size = static_cast<size_t>(cuda::sm_count()) * 8;
  if (max_grid_size < 1) max_grid_size = 1;
  if (grid_size > max_grid_size) grid_size = max_grid_size;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(DataType)  // grad_scores
                              +
                              num_experts * num_token_per_block * sizeof(DataType)  // act_from_fwd
                              + num_experts * num_token_per_block * sizeof(DataType);  // comp_buf
  if (score_function == 1) {
    fused_score_for_moe_aux_loss_backward_kernel<DataType, true>
        <<<grid_size, block_size, shared_memory_size, stream>>>(
            intermediate_output, grad_scores, num_tokens, num_experts, topk, grad_logits);
  } else if (score_function == 0) {
    fused_score_for_moe_aux_loss_backward_kernel<DataType, false>
        <<<grid_size, block_size, shared_memory_size, stream>>>(
            intermediate_output, grad_scores, num_tokens, num_experts, topk, grad_logits);
  } else {
    NVTE_ERROR("Invalid score_function.");
  }
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_score_for_moe_aux_loss_backward(const Tensor &intermediate_output,
                                           const Tensor &grad_scores, int num_tokens,
                                           int num_experts, int topk, int score_function,
                                           Tensor &grad_logits, musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_scores.data.dtype, DataType,
      fused_score_for_moe_aux_loss_backward_kernel_launcher<DataType>(
          reinterpret_cast<DataType *>(intermediate_output.data.dptr),
          reinterpret_cast<DataType *>(grad_scores.data.dptr), num_tokens, num_experts, topk,
          score_function, reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
}

}  // namespace transformer_engine

void nvte_fused_score_for_moe_aux_loss_forward(const NVTETensor logits, int num_tokens,
                                               int num_experts, int topk, int score_function,
                                               NVTETensor scores, const NVTETensor routing_map,
                                               const NVTETensor intermediate_output,
                                               musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_score_for_moe_aux_loss_forward);
  using namespace transformer_engine;
  fused_score_for_moe_aux_loss_forward(*reinterpret_cast<Tensor*>(logits), num_tokens, num_experts,
                                       topk, score_function, *reinterpret_cast<Tensor*>(scores),
                                       *reinterpret_cast<Tensor*>(routing_map),
                                       *reinterpret_cast<Tensor*>(intermediate_output), stream);
}

void nvte_fused_score_for_moe_aux_loss_backward(const NVTETensor intermediate_output,
                                                const NVTETensor grad_scores, int num_tokens,
                                                int num_experts, int topk, int score_function,
                                                NVTETensor grad_logits, musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_score_for_moe_aux_loss_backward);
  using namespace transformer_engine;
  fused_score_for_moe_aux_loss_backward(
      *reinterpret_cast<Tensor*>(intermediate_output), *reinterpret_cast<Tensor*>(grad_scores),
      num_tokens, num_experts, topk, score_function, *reinterpret_cast<Tensor*>(grad_logits), stream);
}
