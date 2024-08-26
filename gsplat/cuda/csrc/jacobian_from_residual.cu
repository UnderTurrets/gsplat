//
// Created by cvgluser on 24-8-23.
//
#include "bindings.h"
#include "helpers.cuh"
#include "spherical_harmonics.cuh"
#include "utils.cuh"
#include "types.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Backward Pass
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void jacobian_bwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
    // fwd inputs
    const S *__restrict__ means,    // [N, 3]
    const S *__restrict__ covars,   // [N, 6] optional
    const S *__restrict__ quats,    // [N, 4] optional
    const S *__restrict__ scales,   // [N, 3] optional
    const S *__restrict__ coeffs,     // [N, K, 3]
    const S *__restrict__ opacities,     // [C, N] or [nnz]
    const S *__restrict__ viewmats, // [C, 4, 4]
    const S *__restrict__ Ks,       // [C, 3, 3]
    // 中间变量
    const S *__restrict__ colors,        // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    // const S *__restrict__ depths,        // [C, N]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    // const S *__restrict__ compensations, const S eps2d, // [C, N] optional
    const S *__restrict__ backgrounds,   // [C, COLOR_DIM] or [nnz, COLOR_DIM]
    const bool *__restrict__ masks,      // [C, tile_height, tile_width]
    // 辅助信息
    const uint32_t K, const uint32_t degrees_to_use, const vec3<S> *__restrict__ dirs, // [N, 3]
    const uint32_t image_width, const uint32_t image_height,
    const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const S *__restrict__ render_alphas,  // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    // grad input which is automatically computed by torch
    const S *__restrict__ v_render_colors, // [C, image_height, image_width, COLOR_DIM]
    const S *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    // output
    int32_t *__restrict__ nnz_per_pixel, // [C, image_width * image_height]
    // grad outputs
    S * __restrict__ jacobian_row_indices, // [C, nnz_Jacobian]
    S *__restrict__ jacobian_col_indices, // [C, nnz_Jacobian]
    S *__restrict__ jacobian_values // [C, nnz_Jacobian]

    // S *__restrict__ v_means,   // [N, 3]
    // S *__restrict__ v_covars,  // [N, 6] optional
    // S *__restrict__ v_quats,   // [N, 4] optional
    // S *__restrict__ v_scales,  // [N, 3] optional
    // S *__restrict__ v_viewmats, // [C, 4, 4] optional
    // S *__restrict__ v_colors,            // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    // S *__restrict__ v_opacities          // [C, N] or [nnz]
) {
    auto block = cg::this_thread_block();
    // block.group_index().y表示当前的tile在这张图片上的第几行tile，block.group_index().z表示当前的tile在这张图片上的第几列tile
    // 知道现在处理的是哪一张图片
    uint32_t camera_id = block.group_index().x;
    // 知道现在处理的是这张图片中的哪个tile
    uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    // block.thread_index().x和block.thread_index().y表示当前线程处理的pixel在tile中的坐标
    // i、j表示当前线程处理的pixel在图片中的坐标
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    // 偏移指针到当前图片
    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    v_render_colors += camera_id * image_height * image_width * COLOR_DIM;
    v_render_alphas += camera_id * image_height * image_width;

    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const S px = (S)j + 0.5f;
    const S py = (S)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    // range_start和range_end用来对flatten_ids进行索引，range_end-range_start是这个tile对应的gaussian的数量
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    // this is the T AFTER the last gaussian in this pixel
    S T_final = 1.0f - render_alphas[pix_id];
    S T = T_final;
    // the contribution from gaussians behind the current one
    // 式子（18）
    S buffer[COLOR_DIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    S v_render_c[COLOR_DIM];
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
    }
    const S v_render_a = v_render_alphas[pix_id];

    // 获得分配给这个block的共享内存的指针
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3<S> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]); // [block_size]
    S *rgbs_batch = (S *)&conic_batch[block_size]; // [block_size * COLOR_DIM]

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    // 拿到这个tile里位于最后索引的gaussian的索引id
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    // 当前线程按照深度从大到小遍历range_end-range_start个gaussians，直到done
    for (uint32_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        // resync all threads before writing next batch of shared mem
        block.sync();
        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        // batch_end指向这个batch的最后一个gaussian
        // 这里在准备gaussians参数的时候已经倒序，因为idx = batch_end - tr
        const int32_t batch_end = range_end - 1 - block_size * batch_idx;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            const int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        // 在b=0时，t=batch_end - warp_bin_final
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = false;
            }
            S alpha; S opac; vec2<S> delta; vec3<S> conic; S vis;
            if (valid) {
                conic = conic_batch[t];
                vec3<S> xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                S sigma =
                    0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                    conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            S v_rgb_local[COLOR_DIM] = {0.f};
            S v_opacity_local = 0.f;
            vec3<S> v_conic_local = {0.f, 0.f, 0.f};
            vec2<S> v_xy_local = {0.f, 0.f};
            vec2<S> v_xy_abs_local = {0.f, 0.f};
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                S ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const S fac = alpha * T;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    // 式子（16）偏Ci/偏cn * 偏Li/偏Ci
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                S v_alpha = 0.f;
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    // 式子（18）偏Ci/偏alpha * 偏Li/偏Ci
                    v_alpha += (rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;

                // 若有background，根据loss的公式推算出background相关梯度
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    S accum = 0.f;
                    PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= 0.999f) {
                    // 式子（19）偏alpha/偏sigma * 偏Li/偏alpha
                    const S v_sigma = -opac * vis * v_alpha;
                    // 偏sigma/偏SIGMA * 偏Li/偏sigma
                    v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
                                     0.5f * v_sigma * delta.x * delta.y,
                                     0.5f * v_sigma * delta.y * delta.y};
                    // 偏sigma/偏delta * 偏Li/偏sigma
                    v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                  v_sigma * (conic.y * delta.x + conic.z * delta.y)};

                    // 式子（19）偏alpha/偏opacity * 偏Li/偏alpha
                    v_opacity_local = vis * v_alpha;
                }

                // 式子（18） 累计Sn
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                }

                // 计算投影的反向传播
                const int32_t global_idx = id_batch[t]; // flatten index in [C * N] or [nnz]
                const uint32_t gaussian_id = global_idx % N;

                S v_coeffs_local[3] = {0.f};
                coeffs += gaussian_id * K * 3; dirs += gaussian_id;
                for (int c =0; c < 4 ; c++) {
                    sh_coeffs_to_color_fast_vjp(degrees_to_use, c, dirs,
                        coeffs, v_rgb_local, v_coeffs_local, nullptr
                        );
                }

                // v_depths += global_idx;
                means += gaussian_id * 3;
                viewmats += camera_id * 16;
                Ks += camera_id * 9;

                // vjp: compute the inverse of the 2d covariance
                mat2<S> covar2d_inv = mat2<S>(conic.x, conic.y, conic.y, conic.z);
                mat2<S> v_covar2d_inv =
                    mat2<S>(v_conic_local[0], v_conic_local[1], v_conic_local[1], v_conic_local[2]);
                // 式子（21） 拿到 偏Li/偏SIGMA
                mat2<S> v_covar2d(0.f);
                inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

                // if (v_compensations != nullptr) {
                //     // vjp: compensation term
                //     const S compensation = compensations[global_idx];
                //     const S v_compensation = v_compensations[global_idx];
                //     add_blur_vjp(eps2d, covar2d_inv, compensation, v_compensation, v_covar2d);
                // }

                // transform Gaussian to camera space
                mat3<S> R = mat3<S>(viewmats[0], viewmats[4], viewmats[8], // 1st column
                                    viewmats[1], viewmats[5], viewmats[9], // 2nd column
                                    viewmats[2], viewmats[6], viewmats[10] // 3rd column
                );
                vec3<S> translate = vec3<S>(viewmats[3], viewmats[7], viewmats[11]);

                mat3<S> covar; vec4<S> quat; vec3<S> scale;
                if (covars != nullptr) {
                    covars += gaussian_id * 6;
                    covar = mat3<S>(covars[0], covars[1], covars[2], // 1st column
                                    covars[1], covars[3], covars[4], // 2nd column
                                    covars[2], covars[4], covars[5]  // 3rd column
                    );
                } else {
                    // compute from quaternions and scales
                    quat = glm::make_vec4(quats + gaussian_id * 4);
                    scale = glm::make_vec3(scales + gaussian_id * 3);
                    quat_scale_to_covar_preci<S>(quat, scale, &covar, nullptr);
                }
                vec3<S> mean_c;
                pos_world_to_cam(R, translate, glm::make_vec3(means), mean_c);
                mat3<S> covar_c;
                // 对协方差矩阵做线性变换
                covar_world_to_cam(R, covar, covar_c);

                // vjp: perspective projection
                S fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
                mat3<S> v_covar_c(0.f); vec3<S> v_mean_c(0.f);
                // 拿到 偏Li/偏SIGMA' 和 偏Li/偏t
                persp_proj_vjp<S>(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height,
                               v_covar2d, v_xy_local, v_mean_c, v_covar_c);

                // add contribution from v_depths
                // v_mean_c.z += v_depths[0];

                // vjp: transform Gaussian covariance to camera space
                vec3<S> v_mean(0.f); mat3<S> v_covar(0.f);
                mat3<S> v_R(0.f); vec3<S> v_t(0.f);
                pos_world_to_cam_vjp(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
                covar_world_to_cam_vjp(R, covar, v_covar_c, v_R, v_covar);

                // 写入v_means
                if (means != nullptr) {
                    PRAGMA_UNROLL
                    for (uint32_t k = 0; k < 3; k++) {

                    }
                }

                // 写入(v_scales, v_quats)或者(v_covars)
                if (covars != nullptr) {
                    // Output gradients w.r.t. the covariance matrix

                } else {
                    // Directly output gradients w.r.t. the quaternion and scale
                    mat3<S> rotmat = quat_to_rotmat<S>(quat);
                    vec4<S> v_quat(0.f);
                    vec3<S> v_scale(0.f);
                    quat_scale_to_covar_vjp<S>(quat, scale, rotmat, v_covar, v_quat, v_scale);

                }

                // 写入v_opacities, v_coeffs

                // 写入v_viewmats
                // if (viewmats != nullptr) {
                //     PRAGMA_UNROLL
                //     for (uint32_t p = 0; p < 3; p++) { // rows
                //         PRAGMA_UNROLL
                //         for (uint32_t q = 0; q < 3; q++) { // cols
                //
                //         }
                //
                //     }
                // }


            }
        }
    }
}

template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad) {

    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(v_render_colors);
    CHECK_INPUT(v_render_alphas);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }

    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t COLOR_DIM = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor v_means2d = torch::zeros_like(means2d);
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_colors = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = torch::zeros_like(means2d);
    }

    if (n_isects) {
        const uint32_t shared_mem = tile_size * tile_size *
                                    (sizeof(int32_t) + sizeof(vec3<float>) +
                                     sizeof(vec3<float>) + sizeof(float) * COLOR_DIM);
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

        if (cudaFuncSetAttribute(jacobian_bwd_kernel<CDIM, float>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        jacobian_bwd_kernel<CDIM, float>
            <<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                colors.data_ptr<float>(), opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                masks.has_value() ? masks.value().data_ptr<bool>(): nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(), v_render_alphas.data_ptr<float>(),
                absgrad
                    ? reinterpret_cast<vec2<float> *>(v_means2d_abs.data_ptr<float>())
                    : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),
                v_colors.data_ptr<float>(), v_opacities.data_ptr<float>());
    }

    return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
jacobian_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients inputs from loss which is automatically computed by torch
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad) {

    CHECK_INPUT(colors);
    uint32_t COLOR_DIM = colors.size(-1);

#define __GS__CALL_(N)                                                                 \
    case N:                                                                            \
        return call_kernel_with_dim<N>(                                                \
            means2d, conics, colors, opacities, backgrounds, masks, image_width,       \
            image_height, tile_size, tile_offsets, flatten_ids, render_alphas,         \
            last_ids, v_render_colors, v_render_alphas, absgrad);

    switch (COLOR_DIM) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
    }
}
