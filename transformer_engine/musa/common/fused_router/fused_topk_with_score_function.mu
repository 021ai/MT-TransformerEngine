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

template <typename DataType, typename BiasType, bool kSoftmax, bool kUsePreSoftmax>
__global__ void fused_topk_with_score_function_forward_kernel(
    const DataType *logits, int num_tokens, int num_experts, int topk, int num_groups,
    int group_topk, float scaling_factor, bool use_double_buffer, const BiasType *expert_bias,
    DataType *probs, bool *routing_map, DataType *intermediate_output) {
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
  extern __shared__ float shmem[];
  int scores_stride = num_experts * num_token_per_block;
  DataType *scores_buf0 = reinterpret_cast<DataType *>(shmem);
  DataType *scores_buf1 = nullptr;
  DataType *topk_scores_buf = nullptr;
  if (use_double_buffer) {
    scores_buf1 = scores_buf0 + scores_stride;
    topk_scores_buf = scores_buf1 + scores_stride;
  } else {
    scores_buf1 = scores_buf0;
    topk_scores_buf = scores_buf0 + scores_stride;
  }
  DataType *group_scores_buf = nullptr, *masked_scores_buf = nullptr;
  int *topk_indices_buf = nullptr;
  if (group_topk > 0) {
    masked_scores_buf = reinterpret_cast<DataType *>(topk_scores_buf + topk * num_token_per_block);
    group_scores_buf =
        reinterpret_cast<DataType *>(masked_scores_buf + num_experts * num_token_per_block);
    topk_indices_buf = reinterpret_cast<int *>(group_scores_buf + num_groups * num_token_per_block);
  } else {
    topk_indices_buf = reinterpret_cast<int *>(topk_scores_buf + topk * num_token_per_block);
  }
  // The address of buffers on the current warp
  DataType *scores = scores_buf0 + warp_id * num_experts;
  DataType *scores_next = scores_buf1 + warp_id * num_experts;
  DataType *topk_scores = topk_scores_buf + warp_id * topk;
  DataType *masked_scores = masked_scores_buf + warp_id * num_experts;
  DataType *group_scores = group_scores_buf + warp_id * num_groups;
  int *topk_indices = topk_indices_buf + warp_id * topk;
  bool has_prefetch = false;

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
    // Load the logits to shmem if not prefetched
    if (!use_double_buffer || !has_prefetch) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        scores[i] = logits[pos_offset + i];
      }
    }
    // If group_topk > 0, init the masked_scores to -inf
    if (group_topk > 0) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        masked_scores[i] = -std::numeric_limits<DataType>::infinity();
      }
    }
    __syncwarp();

    /***
         * Section: Preprocess
         * Possible preprocess the scores before the topk operation
         * - Pre-softmax
         * - Sigmoid
         * - Expert bias
         * This is in-place scores update
         */
    // score_function == 1 means softmax
    if constexpr (kSoftmax && kUsePreSoftmax) {
      // Apply softmax to the logits before the topk
      apply_softmax_on_float(scores, num_experts, lane_id);
      __syncwarp();
      // Save the softmax output for backward
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
    }

    // score_function == 0 means sigmoid
    if constexpr (!kSoftmax) {
      // Apply sigmoid to the logits
      apply_sigmoid_on_float(scores, num_experts, lane_id);
      __syncwarp();
      // Save the sigmoid output for backward
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
    }

    __syncwarp();  //Confirm the scores is written to the softmax/sigmoid output

    // Expert bias is only used at the sigmoid case
    if constexpr (!kSoftmax) {
      if (expert_bias) {
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          scores[i] = static_cast<DataType>(static_cast<float>(scores[i]) +
                                            static_cast<float>(expert_bias[i]));
        }
      }
    }
    __syncwarp();

    /***
         * Section: Topk
         * Get the topk indices
         * - group_topk
         * - naive topk
         * - topk with expert bias
         */
    // Topk on the scores
    // The bias is not empty only happens at the sigmod case
    if (group_topk > 0) {
      int group_size = num_experts / num_groups;
      // Top2
      for (int i = 0; i < num_groups; i++) {
        naive_topk_and_mask(
            /*scores ptr = */ scores + i * group_size,
            /*data size = */ group_size,
            /*topk = */ topk / group_topk,
            /*topk indices ptr = */ topk_indices,
            /*topk scores ptr = */ topk_scores,
            /*lane id = */ lane_id);
        __syncwarp();
        // Compute the group score
        if (lane_id == 0) {
          DataType tmp = 0.0f;
          for (int j = 0; j < topk / group_topk; j++) {
            tmp = tmp + topk_scores[j];
          }
          group_scores[i] = tmp;
        }
        __syncwarp();
      }

      // select the topk groups
      naive_topk_and_mask_inplace(
          /*scores ptr = */ group_scores,
          /*data size = */ num_groups,
          /*topk = */ group_topk,
          /*topk indices ptr = */ topk_indices,
          /*topk scores ptr = */ topk_scores,
          /*lane id = */ lane_id);
      __syncwarp();
      // Copy the unmasked scores to the buffer
      for (int i = 0; i < group_topk; i++) {
        int st = topk_indices[i] * group_size;
        int ed = st + group_size;
        for (int j = st + lane_id; j < ed; j += kThreadsPerWarp) {
          masked_scores[j] = scores[j];
        }
      }
      __syncwarp();
      naive_topk_and_mask_inplace(masked_scores, num_experts, topk, topk_indices, topk_scores,
                                  lane_id);

    } else {
      naive_topk_and_mask_inplace(scores, num_experts, topk, topk_indices, topk_scores, lane_id);
    }
    __syncwarp();

    /***
         * Section: Postprocess
         * Possible postprocess the scores after the topk operation
         * - Revert Expert bias
         * - Softmax
         * - Sigmoid post-processing when topk > 1
         * - Write the result with scaling_factor
         */
    // Revert Expert bias from the topk scores
    if constexpr (!kSoftmax) {
      if (expert_bias) {
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          topk_scores[i] = static_cast<DataType>(
              static_cast<float>(topk_scores[i]) - static_cast<float>(expert_bias[topk_indices[i]]));
        }
      }
    }
    __syncwarp();

    // score_function == 1 means softmax
    if constexpr (kSoftmax && !kUsePreSoftmax) {
      // Apply softmax to the topk logits
      apply_softmax_on_float(topk_scores, topk, lane_id);
      __syncwarp();
      // Save the softmax output for backward
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + topk_indices[i]] = topk_scores[i];
      }
    }

    // score_function == 0 means sigmoid
    if constexpr (!kSoftmax) {
      if (topk > 1) {
        float sum_scores = static_cast<float>(
            warp_reduce_on_shmem(topk_scores, topk, ReduceFuncType::SUM, lane_id));
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          topk_scores[i] = static_cast<DataType>(static_cast<float>(topk_scores[i]) /
                                                 (sum_scores + epsilon));
        }
      }
      __syncwarp();
    }

    // Write the probs/routing_map to the output tensor
    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
      probs[pos_offset + topk_indices[i]] =
          static_cast<DataType>(scaling_factor * static_cast<float>(topk_scores[i]));
    }
    __syncwarp();

    if (use_double_buffer) {
      int token_offset_next_warp = token_offset_cur_warp + gridDim.x * num_token_per_block;
      if (token_offset_next_warp < num_tokens) {
        int pos_offset_next = token_offset_next_warp * num_experts;
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          scores_next[i] = logits[pos_offset_next + i];
        }
        has_prefetch = true;
      } else {
        has_prefetch = false;
      }
      __syncwarp();
      DataType *tmp = scores;
      scores = scores_next;
      scores_next = tmp;
    }
  }
}

template <typename DataType, typename BiasType>
void fused_topk_with_score_function_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    bool use_double_buffer, const BiasType *expert_bias, DataType *probs, bool *routing_map,
    DataType *intermediate_output, musaStream_t stream) {
  const size_t output_elements = static_cast<size_t>(num_tokens) * num_experts;
  NVTE_CHECK_CUDA(musaMemsetAsync(probs, 0, output_elements * sizeof(DataType), stream));
  NVTE_CHECK_CUDA(musaMemsetAsync(routing_map, 0, output_elements * sizeof(bool), stream));

  size_t scores_buffers = use_double_buffer ? 2 : 1;
  size_t per_token_shmem_bytes = scores_buffers * num_experts * sizeof(DataType)  // scores
                                 + topk * sizeof(DataType)                        // topk_scores
                                 + topk * sizeof(int);                            // topk_indices
  if (group_topk > 0) {
    per_token_shmem_bytes += num_groups * sizeof(DataType);   // group_scores
    per_token_shmem_bytes += num_experts * sizeof(DataType);  // masked_scores
  }
  int block_size = select_block_size_by_shmem(per_token_shmem_bytes);
  size_t num_token_per_block = static_cast<size_t>(block_size / kThreadsPerWarp);
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t max_grid_size = static_cast<size_t>(cuda::sm_count()) * 8;
  if (max_grid_size < 1) max_grid_size = 1;
  if (grid_size > max_grid_size) grid_size = max_grid_size;
  size_t shared_memory_size = per_token_shmem_bytes * num_token_per_block;

  if (score_function == 1) {
    if (use_pre_softmax) {
      fused_topk_with_score_function_forward_kernel<DataType, BiasType, true, true>
          <<<grid_size, block_size, shared_memory_size, stream>>>(
              logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
              use_double_buffer, expert_bias, probs, routing_map, intermediate_output);
    } else {
      fused_topk_with_score_function_forward_kernel<DataType, BiasType, true, false>
          <<<grid_size, block_size, shared_memory_size, stream>>>(
              logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
              use_double_buffer, expert_bias, probs, routing_map, intermediate_output);
    }
  } else if (score_function == 0) {
    fused_topk_with_score_function_forward_kernel<DataType, BiasType, false, false>
        <<<grid_size, block_size, shared_memory_size, stream>>>(
            logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,
            use_double_buffer, expert_bias, probs, routing_map, intermediate_output);
  } else {
    NVTE_ERROR("Invalid score_function.");
  }
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_topk_with_score_function_forward(const Tensor logits, int num_tokens, int num_experts,
                                            int topk, bool use_pre_softmax, int num_groups,
                                            int group_topk, float scaling_factor,
                                            int score_function, const Tensor expert_bias,
                                            bool use_double_buffer, Tensor probs, Tensor routing_map,
                                            Tensor intermediate_output, musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      logits.data.dtype, DataType,
      TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
          expert_bias.data.dtype, BiasType,
          fused_topk_with_score_function_forward_kernel_launcher<DataType, BiasType>(
              reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,
              use_pre_softmax, num_groups, group_topk, scaling_factor, score_function,
              use_double_buffer, reinterpret_cast<BiasType *>(expert_bias.data.dptr),
              reinterpret_cast<DataType *>(probs.data.dptr),
              reinterpret_cast<bool *>(routing_map.data.dptr),
              reinterpret_cast<DataType *>(intermediate_output.data.dptr), stream);););
}

template <typename DataType, bool kSoftmax, bool kUsePreSoftmax>
__global__ void fused_topk_with_score_function_backward_kernel(
    // Inputs tensor
    const bool *routing_map, const DataType *intermediate_output, const DataType *grad_probs,
    // Other parameters
    int num_tokens, int num_experts, int topk, float scaling_factor,
    // Output tensor
    DataType *grad_logits) {
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
  DataType *grad_probs_buf = reinterpret_cast<DataType *>(shmem);
  // To store the output of softmax/sigmoid from the fwd
  DataType *act_from_fwd_buf =
      reinterpret_cast<DataType *>(grad_probs_buf + num_experts * num_token_per_block);
  DataType *comp_buf =
      reinterpret_cast<DataType *>(act_from_fwd_buf + num_experts * num_token_per_block);
  // To store the routing_map from the fwd
  bool *routing_map_buf = reinterpret_cast<bool *>(comp_buf + num_experts * num_token_per_block);
  // The address of buffers on the current warp
  DataType *local_grad = grad_probs_buf + warp_id * num_experts;
  DataType *local_act_from_fwd = act_from_fwd_buf + warp_id * num_experts;
  DataType *local_comp_buf = comp_buf + warp_id * num_experts;
  bool *local_routing_map = routing_map_buf + warp_id * num_experts;

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
      local_grad[i] = grad_probs[pos_offset + i];
      local_act_from_fwd[i] = intermediate_output[pos_offset + i];
      local_routing_map[i] = routing_map[pos_offset + i];
    }
    __syncwarp();

    /***
         * Section: Backward of ops after the topk
         * - Backward of the used scaling_factor
         * - Sigmoid Post-processing bwd when topk > 1
         * - Softmax bwd if use_pre_softmax is false
         */
    // Backward of the used scaling_factor
    // In-place update
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (local_routing_map[i]) {
        local_grad[i] = static_cast<DataType>(static_cast<float>(local_grad[i]) * scaling_factor);
      }
    }
    __syncwarp();
    // Sigmoid Post-processing bwd when topk > 1
    if constexpr (!kSoftmax) {
      if (topk > 1) {
      float sum_fwd_input = static_cast<float>(masked_warp_reduce_on_shmem(
          /*data ptr = */ local_act_from_fwd,
          /*mask ptr = */ local_routing_map,
          /*data size = */ num_experts,
          /*reduce func = */ ReduceFuncType::SUM, lane_id));
      // Put the result of output * grad to the comp_buf
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_comp_buf[i] = (local_routing_map[i] ? static_cast<float>(local_grad[i]) *
                                                        static_cast<float>(local_act_from_fwd[i])
                                                  : 0.0f);
      }
      __syncwarp();
      float sum_Output_x_Grad = static_cast<float>(masked_warp_reduce_on_shmem(
          /*data ptr = */ local_comp_buf,
          /*mask ptr = */ local_routing_map,
          /*data size = */ num_experts,
          /*reduce func = */ ReduceFuncType::SUM, lane_id));
      // In-place update
      float norm_inv = 1.0f / (sum_fwd_input + epsilon);
      float corr = sum_Output_x_Grad * norm_inv * norm_inv;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        if (local_routing_map[i]) {
          local_grad[i] =
              static_cast<DataType>(static_cast<float>(local_grad[i]) * norm_inv - corr);
        } else {
          local_grad[i] = 0.0f;
        }
      }
      }
    }
    __syncwarp();
    // Softmax bwd if use_pre_softmax is false
    if constexpr (kSoftmax && !kUsePreSoftmax) {
      apply_softmax_bwd_on_float(local_grad, local_act_from_fwd, local_comp_buf, local_routing_map,
                                 num_experts, lane_id);
      __syncwarp();
    }

    /***
         * Section: Backward of topk
         * mask the unselected position in the grad
         */
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (!local_routing_map[i]) {
        local_grad[i] = 0.0f;
      }
    }
    __syncwarp();

    /***
         * Section: Backward of ops before the topk
         * - Pre-softmax bwd
         * - Sigmoid bwd
         * - Write the grad_logits to the global mem
         */
    // Pre-softmax bwd
    if constexpr (kSoftmax && kUsePreSoftmax) {
      apply_softmax_bwd_on_float(local_grad, local_act_from_fwd, local_comp_buf, nullptr,
                                 num_experts, lane_id);
      __syncwarp();
    }
    // Sigmoid bwd
    if constexpr (!kSoftmax) {
      apply_sigmoid_bwd_on_float(local_grad, local_act_from_fwd, num_experts, lane_id);
      __syncwarp();
    }
    // Write the grad_logits to the global mem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      grad_logits[pos_offset + i] = local_grad[i];
    }
    __syncwarp();
  }
}

template <typename DataType>
void fused_topk_with_score_function_backward_kernel_launcher(
    const bool *routing_map, const DataType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType *grad_logits, musaStream_t stream) {
  // Meta data for the kernel
  size_t per_token_shmem_bytes =
      num_experts * sizeof(DataType)       // grad_probs
      + num_experts * sizeof(DataType)     // act_from_fwd
      + num_experts * sizeof(DataType)     // comp_buf
      + num_experts * sizeof(bool);        // routing_map
  int block_size = select_block_size_by_shmem(per_token_shmem_bytes);
  size_t num_token_per_block = static_cast<size_t>(block_size / kThreadsPerWarp);
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t max_grid_size = static_cast<size_t>(cuda::sm_count()) * 8;
  if (max_grid_size < 1) max_grid_size = 1;
  if (grid_size > max_grid_size) grid_size = max_grid_size;
  size_t shared_memory_size = per_token_shmem_bytes * num_token_per_block;
  if (score_function == 1) {
    if (use_pre_softmax) {
      fused_topk_with_score_function_backward_kernel<DataType, true, true>
          <<<grid_size, block_size, shared_memory_size, stream>>>(
              routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
              scaling_factor, grad_logits);
    } else {
      fused_topk_with_score_function_backward_kernel<DataType, true, false>
          <<<grid_size, block_size, shared_memory_size, stream>>>(
              routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
              scaling_factor, grad_logits);
    }
  } else if (score_function == 0) {
    fused_topk_with_score_function_backward_kernel<DataType, false, false>
        <<<grid_size, block_size, shared_memory_size, stream>>>(
            routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
            scaling_factor, grad_logits);
  } else {
    NVTE_ERROR("Invalid score_function.");
  }
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_topk_with_score_function_backward(const Tensor &routing_map,
                                             const Tensor &intermediate_output,
                                             const Tensor &grad_probs, int num_tokens,
                                             int num_experts, int topk, bool use_pre_softmax,
                                             float scaling_factor, int score_function,
                                             Tensor &grad_logits, musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_logits.data.dtype, DataType,
      fused_topk_with_score_function_backward_kernel_launcher<DataType>(
          reinterpret_cast<bool *>(routing_map.data.dptr),
          reinterpret_cast<DataType *>(intermediate_output.data.dptr),
          reinterpret_cast<DataType *>(grad_probs.data.dptr), num_tokens, num_experts, topk,
          use_pre_softmax, scaling_factor, score_function,
          reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
}

}  // namespace transformer_engine

void nvte_fused_topk_with_score_function_forward(
    const NVTETensor logits, int num_tokens, int num_experts, int topk, int use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    int use_double_buffer,
    const NVTETensor expert_bias, NVTETensor probs, NVTETensor routing_map,
    NVTETensor intermediate_output, musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_forward);
  using namespace transformer_engine;
  fused_topk_with_score_function_forward(
      *reinterpret_cast<Tensor*>(logits), num_tokens, num_experts, topk,
      static_cast<bool>(use_pre_softmax), num_groups, group_topk, scaling_factor, score_function,
      *reinterpret_cast<Tensor*>(expert_bias), static_cast<bool>(use_double_buffer),
      *reinterpret_cast<Tensor*>(probs),
      *reinterpret_cast<Tensor*>(routing_map), *reinterpret_cast<Tensor*>(intermediate_output), stream);
}

void nvte_fused_topk_with_score_function_backward(const NVTETensor routing_map,
                                                  const NVTETensor intermediate_output,
                                                  const NVTETensor grad_probs, int num_tokens,
                                                  int num_experts, int topk, int use_pre_softmax,
                                                  float scaling_factor, int score_function,
                                                  NVTETensor grad_logits, musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_backward);
  using namespace transformer_engine;
  fused_topk_with_score_function_backward(
      *reinterpret_cast<Tensor*>(routing_map), *reinterpret_cast<Tensor*>(intermediate_output),
      *reinterpret_cast<Tensor*>(grad_probs), num_tokens, num_experts, topk,
      static_cast<bool>(use_pre_softmax), scaling_factor, score_function,
      *reinterpret_cast<Tensor*>(grad_logits), stream);
}
