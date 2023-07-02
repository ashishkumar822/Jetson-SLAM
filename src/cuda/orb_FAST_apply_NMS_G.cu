#include "cuda/orb_gpu.hpp"

#include<chrono>
#include<tictoc.hpp>

using namespace std;

namespace orb_cuda {

#define CUDA_NUM_THREADS_PER_BLOCK 32


#define CUDA_NUM_THREADS_PER_BLOCK_x 32
#define CUDA_NUM_THREADS_PER_BLOCK_y 5


#define GRID_LAUNCH

#define NMS_WINDOW 3
#define NMS_WINDOW_HALF (NMS_WINDOW - 1 ) / 2

#define DIVERGENCE_LESS_CODE

__global__ void Tile_unrolling_kernel(int n_threads,
                                      int imheight, int imwidth,
                                      int roiheight, int roiwidth,
                                      int tile_h, int tile_w,
                                      int n_tiles_h, int n_tiles_w,
                                      int warp_tile_h, int warp_tile_w,
                                      int* score_data_gpu,
                                      int* score_unroll_gpu,
                                      int fuse_nms_L_with_nms_G
                                      )
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx <  n_threads)
    {
        const int h = thread_idx / roiwidth;
        const int w = thread_idx % roiwidth;

        const int h_im = h + BORDER_SKIP;
        const int w_im = w + BORDER_SKIP;

        int score = score_data_gpu[h_im * imwidth + w_im];

        if(fuse_nms_L_with_nms_G)
        {
#if(NMS_WINDOW==3)
            //unrolled for 3x3
            int row_offset = (h_im + -1) * imwidth;
            int col_offset = w_im + -1;
            int total_offset = row_offset + col_offset;

            score *= score >= score_data_gpu[total_offset];
            score *= score >= score_data_gpu[total_offset+1];
            score *= score >= score_data_gpu[total_offset+2];

            row_offset = (h_im + 0) * imwidth;
            total_offset = row_offset + col_offset;

            score *= score >= score_data_gpu[total_offset];
            score *= score >= score_data_gpu[total_offset+1];
            score *= score >= score_data_gpu[total_offset+2];


            row_offset = (h_im + 1) * imwidth;
            total_offset = row_offset + col_offset;

            score *= score >= score_data_gpu[total_offset];
            score *= score >= score_data_gpu[total_offset+1];
            score *= score >= score_data_gpu[total_offset+2];

#else

#ifdef DIVERGENCE_LESS_CODE

            //no divergence
            for(int i=-NMS_WINDOW_HALF;i<=NMS_WINDOW_HALF;i++)
            {
                int row_offset = (h_im + i) * imwidth;

                for(int j=-NMS_WINDOW_HALF;j<=NMS_WINDOW_HALF;j++)
                {
                    int col_offset = w_im + j;

                    int res = (score >= score_data_gpu[row_offset + col_offset]);
                    score *= (res);
                }
            }

#else
            // divergence
            for(int i=-NMS_WINDOW_HALF;i<=NMS_WINDOW_HALF;i++)
            {
                int row_offset = (h_im + i) * imwidth;

                for(int j=-NMS_WINDOW_HALF;j<=NMS_WINDOW_HALF;j++)
                {
                    int col_offset = w_im + j;

                    if(score < score_data_gpu[row_offset + col_offset])
                    {
                        score = 0;
                        break;
                    }
                }

                if(!score)break;
            }
#endif

#endif
        }


        //        const int eff_h = imheight - 2 * BORDER_SKIP;
        const int eff_w = imwidth  - 2 * BORDER_SKIP;

        // height width of the tile in the image
        const int tile_idx_h = (thread_idx / eff_w) / tile_h;
        const int tile_idx_w = (thread_idx % eff_w) / tile_w;

        // height width inside the tile
        const int tile_loc_h = (thread_idx / eff_w) % tile_h;
        const int tile_loc_w = (thread_idx % eff_w) % tile_w;


        const int warp_tile_count = warp_tile_h * warp_tile_w;

        const int tile_idx = tile_idx_h * n_tiles_w + tile_idx_w;

        const int inter_tile_offset = tile_idx * warp_tile_count;
        const int intra_tile_offset = tile_loc_h * tile_w + tile_loc_w;

        score_unroll_gpu[inter_tile_offset +  intra_tile_offset] = score;
    }


}


__global__ void Tile_reduction_kernel(int n_threads,
                                      int tile_h, int tile_w,
                                      int n_tiles_h, int n_tiles_w,
                                      int warp_tile_h, int warp_tile_w,
                                      int* image_unroll_gpu,
                                      int* kp_x, int* kp_y,
                                      int* kp_score
                                      )
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_blk_id = threadIdx.x;

    if(thread_idx <  n_threads)
    {
        __shared__ int shared_idx_h[32];
        __shared__ int shared_idx_w[32];
        __shared__ int shared_score[32];

        const int tile_idx_h = thread_idx / warp_tile_w;
        const int tile_idx_w = thread_idx % warp_tile_w;

        const int warp_tile_count = warp_tile_h * warp_tile_w;
        int* ptr_tile = image_unroll_gpu +  tile_idx_h *  warp_tile_count;

        int score = ptr_tile[tile_idx_w];
        int idx_h = 0;

        for(int i=1;i<warp_tile_h;i++)
        {
            int temp_score = ptr_tile[i * warp_tile_w + tile_idx_w];

            if(temp_score > score)
            {
                idx_h = i;
                score = temp_score;
            }
        }

        int idx_w = thread_blk_id;

        shared_idx_h[thread_blk_id] = idx_h;
        shared_idx_w[thread_blk_id] = idx_w;
        shared_score[thread_blk_id] = score;

        const int warp_size = 32;
        int group_size = warp_size / 2;
        const int log_warp_size = 5; // log(32);

        __syncthreads();

        for(int i=0;i< log_warp_size;i++)
        {
            if(thread_blk_id < group_size)
            {
                int nbr_idx = group_size + thread_blk_id;

                int temp_score = shared_score[nbr_idx];

                if(temp_score > score)
                {
                    idx_h = shared_idx_h[nbr_idx];
                    idx_w = shared_idx_w[nbr_idx];
                    score = temp_score;
                }

                shared_idx_h[thread_blk_id] = idx_h;
                shared_idx_w[thread_blk_id] = idx_w;
                shared_score[thread_blk_id] = score;
            }

            group_size = group_size >> 1; // group_size /= 2;

            __syncthreads();
        }


        if(thread_blk_id == 0)
        {
            int score = shared_score[0];
            kp_score[tile_idx_h] = shared_score[0];

            if(score > 0)
            {
                const int idx = shared_idx_h[0] * warp_tile_w + shared_idx_w[0];
                const int tile_loc_h = idx / tile_w;
                const int tile_loc_w = idx % tile_w;

                const int roi_h = (tile_idx_h / n_tiles_w) * tile_h + tile_loc_h;
                const int roi_w = (tile_idx_h % n_tiles_w) * tile_w + tile_loc_w;


                kp_y[tile_idx_h] = roi_h + BORDER_SKIP;
                kp_x[tile_idx_h] = roi_w + BORDER_SKIP;
            }
        }
    }
}


void ORB_GPU::FAST_apply_NMS_G_unroll_reduce(int height, int width,
                                             int tile_h, int tile_w,
                                             int n_tiles_h, int n_tiles_w,
                                             int warp_tile_h, int warp_tile_w,
                                             int fuse_nms_L_with_nms_G,
                                             int* image_unroll_gpu,
                                             int* score_data_nms_gpu,
                                             int* keypoints_x,
                                             int* keypoints_y,
                                             int* keypoints_score,
                                             cudaStream_t& cuda_stream)
{



    {
        // image unrolling
        {
            int roiheight = (height - 2*BORDER_SKIP);
            int roiwidth  = (width - 2*BORDER_SKIP);

            int n_threads =  roiheight * roiwidth;

            int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK + 1;

            Tile_unrolling_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                     n_threads,
                                                                                                     height, width,
                                                                                                     roiheight, roiwidth,
                                                                                                     tile_h, tile_w,
                                                                                                     n_tiles_h, n_tiles_w,
                                                                                                     warp_tile_h, warp_tile_w,
                                                                                                     score_data_nms_gpu,
                                                                                                     image_unroll_gpu,
                                                                                                     fuse_nms_L_with_nms_G
                                                                                                     );

        }



        // NMS reduction
        {
            int n_threads =  n_tiles_h * n_tiles_w * warp_tile_w;

            int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK + 1;

            Tile_reduction_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                     n_threads,
                                                                                                     tile_h, tile_w,
                                                                                                     n_tiles_h, n_tiles_w,
                                                                                                     warp_tile_h, warp_tile_w,
                                                                                                     image_unroll_gpu,
                                                                                                     keypoints_x,
                                                                                                     keypoints_y,
                                                                                                     keypoints_score);


        }
    }
}





__global__ void Tile_unrolling_kernel_v2(int n_threads,
                                         int imheight, int imwidth,
                                         int roiheight, int roiwidth,
                                         int tile_h, int tile_w,
                                         int n_tiles_h, int n_tiles_w,
                                         int warp_tile_h, int warp_tile_w,
                                         int* score_data_gpu,
                                         int* score_unroll_gpu,
                                         int* score_unroll_x_gpu,
                                         int* score_unroll_y_gpu,
                                         int fuse_nms_L_with_nms_G)
{
#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if(w < roiwidth && h < n_tiles_h)
    {
        int thread_idx = h * roiwidth + w;

#else
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx <  n_threads)
    {
        int h = thread_idx / roiwidth;
        int w = thread_idx % roiwidth;
#endif


        int h_im = h * tile_h + BORDER_SKIP;
        const int w_im = w + BORDER_SKIP;

        int score = score_data_gpu[h_im * imwidth + w_im];

        if(fuse_nms_L_with_nms_G)
        {
#if(NMS_WINDOW==3)
            //unrolled for 3x3
            int row_offset = (h_im + -1) * imwidth;
            int col_offset = w_im + -1;
            int total_offset = row_offset + col_offset;

            score *= score >= score_data_gpu[total_offset];
            score *= score >= score_data_gpu[total_offset+1];
            score *= score >= score_data_gpu[total_offset+2];

            row_offset = (h_im + 0) * imwidth;
            total_offset = row_offset + col_offset;

            score *= score >= score_data_gpu[total_offset];
            score *= score >= score_data_gpu[total_offset+1];
            score *= score >= score_data_gpu[total_offset+2];


            row_offset = (h_im + 1) * imwidth;
            total_offset = row_offset + col_offset;

            score *= score >= score_data_gpu[total_offset];
            score *= score >= score_data_gpu[total_offset+1];
            score *= score >= score_data_gpu[total_offset+2];

#else

#ifdef DIVERGENCE_LESS_CODE

            //no divergence
            for(int i=-NMS_WINDOW_HALF;i<=NMS_WINDOW_HALF;i++)
            {
                int row_offset = (h_im + i) * imwidth;

                for(int j=-NMS_WINDOW_HALF;j<=NMS_WINDOW_HALF;j++)
                {
                    int col_offset = w_im + j;

                    int res = (score >= score_data_gpu[row_offset + col_offset]);
                    score *= (res);
                }
            }

#else
            // divergence
            for(int i=-NMS_WINDOW_HALF;i<=NMS_WINDOW_HALF;i++)
            {
                int row_offset = (h_im + i) * imwidth;

                for(int j=-NMS_WINDOW_HALF;j<=NMS_WINDOW_HALF;j++)
                {
                    int col_offset = w_im + j;

                    if(score < score_data_gpu[row_offset + col_offset])
                    {
                        score = 0;
                        break;
                    }
                }

                if(!score)break;
            }
#endif

#endif
        }

        int max_score = score;
        int max_x = w_im;
        int max_y = h_im;

        int max_h = h_im + tile_h;

        if(max_h > imheight - BORDER_SKIP)
            max_h = imheight - BORDER_SKIP;

        for(int i=h_im+1;i<max_h;i++)
        {
            const int score = score_data_gpu[i * imwidth + w_im];

            if(score > max_score)
            {
                max_score = score;
                max_y = i;
            }
        }

        const int eff_w = imwidth  - 2 * BORDER_SKIP;

        // height width of the tile in the image
        const int tile_idx_h = h;
        const int tile_idx_w = (thread_idx % eff_w) / tile_w;
        //inter tile offset
        const int tile_idx = tile_idx_h * n_tiles_w + tile_idx_w;
        const int warp_tile_count = warp_tile_h * warp_tile_w;
        const int inter_tile_offset = tile_idx * warp_tile_count;


        // height width inside the tile
        const int tile_loc_w = (thread_idx % eff_w) % tile_w;

        //intra tile offset
        const int intra_tile_offset = tile_loc_w;

        score_unroll_gpu[inter_tile_offset +  intra_tile_offset] = max_score;
        score_unroll_x_gpu[inter_tile_offset +  intra_tile_offset] = max_x;
        score_unroll_y_gpu[inter_tile_offset +  intra_tile_offset] = max_y;

    }


}



__global__ void Tile_unrolling_kernel_v3(int n_threads,
                                         int imheight, int imwidth,
                                         int roiheight, int roiwidth,
                                         int tile_h, int tile_w,
                                         int n_tiles_h, int n_tiles_w,
                                         int warp_tile_h, int warp_tile_w,
                                         int* score_data_gpu,
                                         int* score_unroll_gpu,
                                         int* score_unroll_x_gpu,
                                         int* score_unroll_y_gpu,
                                         int fuse_nms_L_with_nms_G)
{
#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if(w < roiwidth && h < n_tiles_h)
    {
        int thread_idx = h * roiwidth + w;

#else
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx <  n_threads)
    {
        int h = thread_idx / roiwidth;
        int w = thread_idx % roiwidth;
#endif


        int h_im = h * tile_h + BORDER_SKIP;
        const int w_im = w + BORDER_SKIP;

        int max_score = 0;
        int max_x = w_im;
        int max_y = h_im;

        int max_h = h_im + tile_h;

        if(max_h > imheight - BORDER_SKIP)
            max_h = imheight - BORDER_SKIP;

        for(int i=h_im;i<max_h;i++)
        {
            int score = score_data_gpu[i * imwidth + w_im];

            //            if(score > max_score)
            {
                if(fuse_nms_L_with_nms_G)
                {
#if(NMS_WINDOW==3)

                    int valid = 1;
                    //unrolled for 3x3
                    int row_offset = (i + -1) * imwidth;
                    int col_offset = w_im + -1;
                    int total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];

                    row_offset = (i + 0) * imwidth;
                    total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];


                    row_offset = (i + 1) * imwidth;
                    total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];

#else

#ifdef DIVERGENCE_LESS_CODE

                    //no divergence
                    for(int i=-NMS_WINDOW_HALF;i<=NMS_WINDOW_HALF;i++)
                    {
                        int row_offset = (h_im + i) * imwidth;

                        for(int j=-NMS_WINDOW_HALF;j<=NMS_WINDOW_HALF;j++)
                        {
                            int col_offset = w_im + j;

                            int res = (score >= score_data_gpu[row_offset + col_offset]);
                            score *= (res);
                        }
                    }

#else
                    // divergence
                    for(int i=-NMS_WINDOW_HALF;i<=NMS_WINDOW_HALF;i++)
                    {
                        int row_offset = (h_im + i) * imwidth;

                        for(int j=-NMS_WINDOW_HALF;j<=NMS_WINDOW_HALF;j++)
                        {
                            int col_offset = w_im + j;

                            if(score < score_data_gpu[row_offset + col_offset])
                            {
                                score = 0;
                                break;
                            }
                        }

                        if(!score)break;
                    }
#endif

#endif

                    score *= valid;
                }

                if(score > max_score)
                {
                    max_score = score;
                    max_y = i;
                }
            }
        }

        const int eff_w = imwidth  - 2 * BORDER_SKIP;

        // height width of the tile in the image
        const int tile_idx_h = h;
        const int tile_idx_w = (thread_idx % eff_w) / tile_w;
        //inter tile offset
        const int tile_idx = tile_idx_h * n_tiles_w + tile_idx_w;
        const int warp_tile_count = warp_tile_h * warp_tile_w;
        const int inter_tile_offset = tile_idx * warp_tile_count;


        // height width inside the tile
        const int tile_loc_w = (thread_idx % eff_w) % tile_w;

        //intra tile offset
        const int intra_tile_offset = tile_loc_w;

        score_unroll_gpu[inter_tile_offset +  intra_tile_offset] = max_score;
        score_unroll_x_gpu[inter_tile_offset +  intra_tile_offset] = max_x;
        score_unroll_y_gpu[inter_tile_offset +  intra_tile_offset] = max_y;

    }


}



__global__ void Tile_unrolling_kernel_v4(int n_threads,
                                         int imheight, int imwidth,
                                         int roiheight, int roiwidth,
                                         int tile_h, int tile_w,
                                         int n_tiles_h, int n_tiles_w,
                                         int warp_tile_h, int warp_tile_w,
                                         int* score_data_gpu,
                                         int score_pitch,
                                         int* score_unroll_gpu,
                                         int* score_unroll_x_gpu,
                                         int* score_unroll_y_gpu,
                                         int fuse_nms_L_with_nms_G)
{

    const int n_threads_per_tile = 5;


#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    int h_im = (h/n_threads_per_tile) * tile_h;
    const int w_im = w;

    int tile_boundry_h = h_im + tile_h;

    if(h_im < BORDER_SKIP)
        h_im = BORDER_SKIP;

    if(h_im > imheight - BORDER_SKIP)
        h_im = imheight - BORDER_SKIP;

    if(w < imwidth)
    {

#else
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx <  n_threads)
    {
        int h = thread_idx / roiwidth;
        int w = thread_idx % roiwidth;
#endif


        __shared__ int shared_score[32*n_threads_per_tile];
        __shared__ int shared_x[32*n_threads_per_tile];
        __shared__ int shared_y[32*n_threads_per_tile];


        int max_score = 0;
        int max_x = w_im;
        int max_y = h_im;

        int my_idx = threadIdx.y * blockDim.x + threadIdx.x;
        int my_idx_reduce  = threadIdx.y;

        int mini_tile = (tile_h - 1) / n_threads_per_tile + 1;


        if(tile_boundry_h > imheight - BORDER_SKIP)
            tile_boundry_h = imheight - BORDER_SKIP;


        for(int i=0;i<mini_tile;i++)
        {
            int h = h_im + my_idx_reduce + i * n_threads_per_tile;

            if(h < tile_boundry_h)
            {
                int score = score_data_gpu[h * score_pitch + w_im];

                if(fuse_nms_L_with_nms_G)
                {
                    int valid = 1;
                    //unrolled for 3x3
                    int row_offset = (h + -1) * score_pitch;
                    int col_offset = w_im + -1;
                    int total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];

                    row_offset = (h + 0) * score_pitch;
                    total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];


                    row_offset = (h + 1) * score_pitch;
                    total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];

                    score *= valid;
                }

                if(score > max_score)
                {
                    max_score = score;
                    max_y = h;
                }
            }
        }

        if((w_im < BORDER_SKIP) || (w_im > imwidth - BORDER_SKIP))
            max_score = 0;

        shared_score[my_idx] = max_score;
        shared_x[my_idx] = w_im;
        shared_y[my_idx] = max_y;

        __syncthreads();

        if(my_idx_reduce == 0)
        {
            for(int i=1;i<n_threads_per_tile;i++)
            {
                int temp_score = shared_score[i*32+threadIdx.x];

                if(max_score < temp_score)
                {
                    max_score = temp_score;
                    max_y = shared_y[i*32+threadIdx.x];
                    max_x = shared_x[i*32+threadIdx.x];
                }
            }

            int thread_idx = (h/n_threads_per_tile) * imwidth + w;

            const int eff_w = imwidth;

            // height width of the tile in the image
            const int tile_idx_h = (h/n_threads_per_tile);
            const int tile_idx_w = (thread_idx % eff_w) / tile_w;
            //inter tile offset
            const int tile_idx = tile_idx_h * n_tiles_w + tile_idx_w;
            const int warp_tile_count = warp_tile_h * warp_tile_w;
            const int inter_tile_offset = tile_idx * warp_tile_count;


            // height width inside the tile
            const int tile_loc_w = (thread_idx % eff_w) % tile_w;

            //intra tile offset
            const int intra_tile_offset = tile_loc_w;

            score_unroll_gpu[inter_tile_offset +  intra_tile_offset] = max_score;
            score_unroll_x_gpu[inter_tile_offset +  intra_tile_offset] = max_x;
            score_unroll_y_gpu[inter_tile_offset +  intra_tile_offset] = max_y;
        }

    }


}




__global__ void Tile_reduction_kernel_v2(int n_threads,
                                         int tile_h, int tile_w,
                                         int n_tiles_h, int n_tiles_w,
                                         int warp_tile_h, int warp_tile_w,
                                         int* image_unroll_gpu,
                                         int* image_unroll_x_gpu,
                                         int* image_unroll_y_gpu,
                                         int* kp_x, int* kp_y,
                                         int* kp_score)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_blk_id = threadIdx.x;

    if(thread_idx <  n_threads)
    {
        __shared__ int shared_idx_h[32];
        __shared__ int shared_idx_w[32];
        __shared__ int shared_score[32];

        const int tile_idx_h = thread_idx / warp_tile_w;
        const int tile_idx_w = thread_idx % warp_tile_w;

        const int warp_tile_count = warp_tile_h * warp_tile_w;
        int* ptr_tile = image_unroll_gpu +  tile_idx_h *  warp_tile_count;

        int score = ptr_tile[tile_idx_w];
        int idx_h = 0;

        for(int i=1;i<warp_tile_h;i++)
        {
            int temp_score = ptr_tile[i * warp_tile_w + tile_idx_w];

            if(temp_score > score)
            {
                idx_h = i;
                score = temp_score;
            }
        }

        int idx_w = thread_blk_id;

        shared_idx_h[thread_blk_id] = idx_h;
        shared_idx_w[thread_blk_id] = idx_w;
        shared_score[thread_blk_id] = score;

        //        const int warp_size = 32;
        //        int group_size = warp_size / 8;
        //        const int log_warp_size = 2; // log(32);

        //        // no need because warp size is only 32
        //        //        __syncthreads();

        //        for(int i=0;i< log_warp_size;i++)
        //        {
        //            if(thread_blk_id < group_size)
        //            {
        //                int nbr_idx = group_size + thread_blk_id;

        //                int temp_score = shared_score[nbr_idx];

        //                if(temp_score > score)
        //                {
        //                    idx_h = shared_idx_h[nbr_idx];
        //                    idx_w = shared_idx_w[nbr_idx];
        //                    score = temp_score;
        //                }

        //                shared_idx_h[thread_blk_id] = idx_h;
        //                shared_idx_w[thread_blk_id] = idx_w;
        //                shared_score[thread_blk_id] = score;
        //            }

        //            group_size = group_size >> 1; // group_size /= 2;

        //            // no need because warp size is only 32
        //            //            __syncthreads();
        //        }


        //        if(thread_blk_id == 0)
        //        {
        //            const int score = shared_score[0];
        //            kp_score[tile_idx_h] = shared_score[0];

        //            if(score > 0)
        //            {
        //                const int idx = shared_idx_h[0] * warp_tile_w + shared_idx_w[0];

        //                int* ptr_tile_x = image_unroll_x_gpu +  tile_idx_h *  warp_tile_count;
        //                int* ptr_tile_y = image_unroll_y_gpu +  tile_idx_h *  warp_tile_count;

        //                const int x = ptr_tile_x[idx];
        //                const int y = ptr_tile_y[idx];

        //                kp_y[tile_idx_h] = y;
        //                kp_x[tile_idx_h] = x;
        //            }
        //        }



#define WARP_MASK                            0xFFFFFFFF

        const int warp_size = 32;
        int group_size = warp_size / 2;
        const int log_warp_size = 5; // log(32);

#pragma unroll
        for(int i=0;i< log_warp_size;i++)
        {
            //            if(thread_blk_id < group_size)
            {
                int temp_score = __shfl_down_sync(WARP_MASK, score, group_size);
                int idx_w_new  = __shfl_down_sync(WARP_MASK, idx_w, group_size);
                int idx_h_new  = __shfl_down_sync(WARP_MASK, idx_h, group_size);

                if(temp_score > score)
                {
                    idx_h = idx_h_new;
                    idx_w = idx_w_new;
                    score = temp_score;
                }
            }

            group_size = group_size >> 1; // group_size /= 2;
        }

        if(thread_blk_id == 0)
        {
            kp_score[tile_idx_h] = score;

            if(score > 0)
            {
                const int idx = idx_h * warp_tile_w + idx_w;

                int* ptr_tile_x = image_unroll_x_gpu +  tile_idx_h *  warp_tile_count;
                int* ptr_tile_y = image_unroll_y_gpu +  tile_idx_h *  warp_tile_count;

                const int x = ptr_tile_x[idx];
                const int y = ptr_tile_y[idx];

                kp_y[tile_idx_h] = y;
                kp_x[tile_idx_h] = x;
            }
        }

    }
}






__global__ void Tile_unrolling_reduction_kernel(int imheight, int imwidth,
                                                int roiheight, int roiwidth,
                                                int tile_h, int tile_w,
                                                int n_tiles_h, int n_tiles_w,
                                                int n_loc_per_thread,
                                                int n_threads_y_per_tile,
                                                int n_tiles_per_block,
                                                int* score_data_gpu,
                                                int score_pitch,
                                                int* kp_x, int* kp_y,
                                                int* kp_score,
                                                int fuse_nms_L_with_nms_G)
{

#ifdef GRID_LAUNCH

    int w = n_tiles_per_block * tile_w * blockIdx.x + threadIdx.x;
    int h = blockIdx.y * tile_h;

    int block_max = n_tiles_per_block * tile_w;

    if(w < roiwidth && h < roiheight)
    {

#else
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx <  n_threads)
    {
        int h = thread_idx / roiwidth;
        int w = thread_idx % roiwidth;
#endif

        int h_im = h + BORDER_SKIP;
        const int w_im = w + BORDER_SKIP;

        int max_score = 0;
        int max_x = w_im;
        int max_y = h_im;

        int my_idx_reduce  = threadIdx.y;

        int mini_tile = (tile_h - 1) / n_threads_y_per_tile + 1;

        int tile_boundry_h = h_im + tile_h;

        if(tile_boundry_h > imheight - BORDER_SKIP)
            tile_boundry_h = imheight - BORDER_SKIP;


        for(int i=0;i<mini_tile;i++)
        {
            int h = h_im + my_idx_reduce + i * n_threads_y_per_tile;

            if(h < tile_boundry_h)
            {
                int score = score_data_gpu[h * score_pitch + w_im];

                if(fuse_nms_L_with_nms_G)
                {
                    int valid = 1;
                    //unrolled for 3x3
                    int row_offset = (h + -1) * score_pitch;
                    int col_offset = w_im + -1;
                    int total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];

                    row_offset = (h + 0) * score_pitch;
                    total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    //                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];


                    row_offset = (h + 1) * score_pitch;
                    total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];

                    score *= valid;
                }

                if(score > max_score)
                {
                    max_score = score;
                    max_y = h;
                }
            }
        }

        __shared__ int shared_score[128*15];
        __shared__ int shared_x[128*15];
        __shared__ int shared_y[128*15];


        int my_idx = threadIdx.y * blockDim.x + threadIdx.x;

        shared_score[my_idx] = max_score;
        shared_x[my_idx] = w_im;
        shared_y[my_idx] = max_y;

        __syncthreads();

        if(my_idx_reduce == 0)
        {
            for(int i=1;i<n_threads_y_per_tile;i++)
            {
                int offset = i*blockDim.x+threadIdx.x;

                int temp_score = shared_score[offset];

                if(max_score < temp_score)
                {
                    max_score = temp_score;
                    max_y = shared_y[offset];
                    max_x = shared_x[offset];
                }
            }


            shared_score[my_idx] = max_score;
            shared_x[my_idx] = max_x;
            shared_y[my_idx] = max_y;
        }

        __syncthreads();

        int tile_loc_w = threadIdx.x % tile_w;

        if(tile_loc_w == 0)
        {
            int tile_idx_w = threadIdx.x / tile_w;

            for(int i=1;i<tile_w;i++)
            {
                int offset = i + tile_idx_w * tile_w;

                int temp_score = shared_score[offset];

                if(max_score < temp_score)
                {
                    max_score = temp_score;
                    max_y = shared_y[offset];
                    max_x = shared_x[offset];
                }
            }


            int tile_idx_h_img = blockIdx.y;
            int tile_idx_w_img = blockIdx.x * n_tiles_per_block + tile_idx_w;

            int tile_idx = tile_idx_h_img * n_tiles_w + tile_idx_w_img;

            //            if(tile_idx >= n_tiles_h * n_tiles_w)
            //            printf("%d %d %d %d \n", tile_idx, max_x, max_y, n_tiles_h * n_tiles_w);

            kp_score[tile_idx] = max_score;
            kp_x[tile_idx] = max_x;
            kp_y[tile_idx] = max_y;

        }




        //        int tile_idx_w = threadIdx.x / tile_w;
        //        int tile_loc_w = threadIdx.x % tile_w;

        //        int log2_tile_w = log2((float)tile_w);

        //        int group_size = (tile_w - 1) / 2 + 1;

        //        my_idx = threadIdx.x;

        //        for(int i=0;i<log2_tile_w;i++)
        //        {
        //            if(tile_loc_w < group_size)
        //            {
        //                int offset = tile_loc_w +  group_size;

        //                if(offset < tile_w)
        //                {
        //                    offset = threadIdx.x + group_size;

        //                    int temp_score = shared_score[offset];

        //                    if(max_score < temp_score)
        //                    {
        //                        max_score = temp_score;
        //                        max_y = shared_y[offset];
        //                        max_x = shared_x[offset];
        //                    }
        //                }

        //                shared_score[my_idx] = max_score;
        //                shared_x[my_idx] = max_x;
        //                shared_y[my_idx] = max_y;
        //            }

        //            group_size = (group_size - 1) / 2 + 1;

        //            __syncthreads();
        //        }

        //        if(tile_loc_w == 0)
        //        {
        //            int tile_idx_h_img = blockIdx.y;
        //            int tile_idx_w_img = blockIdx.x * n_tiles_per_block + tile_idx_w;

        //            int tile_idx = tile_idx_h_img * n_tiles_w + tile_idx_w_img;

        //            //            if(tile_idx >= n_tiles_h * n_tiles_w)
        //            //            printf("%d %d %d %d \n", tile_idx, max_x, max_y, n_tiles_h * n_tiles_w);

        //            kp_score[tile_idx] = max_score;
        //            kp_x[tile_idx] = max_x;
        //            kp_y[tile_idx] = max_y;

        //        }


    }


}








__global__ void Tile_unrolling_reduction_kernel_v2(int imheight, int imwidth,
                                                   int tile_h, int tile_w,
                                                   int n_tiles_h, int n_tiles_w,
                                                   int n_loc_per_thread,
                                                   int n_threads_y_per_tile,
                                                   int n_tiles_per_block,
                                                   int* score_data_gpu,
                                                   int score_pitch,
                                                   int* kp_x,
                                                   int* kp_y,
                                                   int* kp_score,
                                                   int fuse_nms_L_with_nms_G)
{

    const int n_threads_per_tile = n_threads_y_per_tile;

#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    int block_max = n_tiles_per_block * tile_w;


    int h_im = (h/n_threads_per_tile) * tile_h;
    const int w_im =  blockIdx.x * block_max + threadIdx.x;

    int tile_boundry_h_min = h_im;
    int tile_boundry_h_max = h_im + tile_h;

    if(tile_boundry_h_min < BORDER_SKIP)
        tile_boundry_h_min = BORDER_SKIP;

    if(tile_boundry_h_max > imheight - BORDER_SKIP)
        tile_boundry_h_max = imheight - BORDER_SKIP;


    __shared__ int shared_score[128*10];
    __shared__ int shared_x[128*10];
    __shared__ int shared_y[128*10];



    int max_score = 0;
    int max_x = w_im;
    int max_y = h_im;

    int my_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int my_idx_reduce  = threadIdx.y;


    if(w_im < imwidth && threadIdx.x < block_max)
    {

#else
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_idx <  n_threads)
    {
        int h = thread_idx / roiwidth;
        int w = thread_idx % roiwidth;
#endif

        //vertical cell aggregation


        const int mini_tile = (tile_h - 1) / n_threads_per_tile + 1;

        const int h_start = h_im + my_idx_reduce;

        for(int i=0;i<mini_tile;i++)
        {
            int h = h_start + i * n_threads_per_tile;

            if(h >=tile_boundry_h_min && h < tile_boundry_h_max)
            {
                int score = score_data_gpu[h * score_pitch + w_im];

                if(fuse_nms_L_with_nms_G)
                {
                    int valid = 1;
                    //unrolled for 3x3
                    int row_offset = (h + -1) * score_pitch;
                    int col_offset = w_im + -1;
                    int total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];

                    row_offset = (h + 0) * score_pitch;
                    total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    //                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];


                    row_offset = (h + 1) * score_pitch;
                    total_offset = row_offset + col_offset;

                    valid &= score >= score_data_gpu[total_offset];
                    valid &= score >= score_data_gpu[total_offset+1];
                    valid &= score >= score_data_gpu[total_offset+2];

                    score *= valid;
                }

                if(score > max_score)
                {
                    max_score = score;
                    max_y = h;
                }
            }
        }
    }

    shared_score[my_idx] = max_score;
    shared_y[my_idx] = max_y;

    // shouldn't be called inside only if blocks
    __syncthreads();

    int tile_idx_w = threadIdx.x / tile_w;

    if(w_im < imwidth && threadIdx.x < block_max)
    {
        if(my_idx_reduce == 0)
        {
            for(int i=1;i<n_threads_per_tile;i++)
            {
                int offset = i*blockDim.x+threadIdx.x;
                int temp_score = shared_score[offset];

                if(max_score < temp_score)
                {
                    max_score = temp_score;
                    max_y = shared_y[offset];
                }

            }

            shared_score[my_idx] = max_score;
            shared_x[my_idx] = max_x;
            shared_y[my_idx] = max_y;
        }
    }

    __syncthreads();


    //horizontal cell aggregation
    //log2
    int tile_loc_w = threadIdx.x % tile_w;
    int log2_tile_w = ceilf(log2f((float)tile_w));

    int group_size = (tile_w - 1) / 2 + 1;

    for(int i=0;i<log2_tile_w;i++)
    {
        if(w_im < imwidth && my_idx_reduce == 0 && tile_loc_w < group_size && threadIdx.x < block_max)
        {
            int offset = tile_loc_w +  group_size;

            if(offset < tile_w)
            {
                offset = my_idx + group_size;

                int temp_score = shared_score[offset];

                if(max_score < temp_score)
                {
                    max_score = temp_score;
                    max_y = shared_y[offset];
                    max_x = shared_x[offset];
                }
            }

            shared_score[my_idx] = max_score;
            shared_x[my_idx] = max_x;
            shared_y[my_idx] = max_y;
        }

        group_size = (group_size - 1) / 2 + 1;

        __syncthreads();
    }

    //MLPT
    {

    }


    if(w_im < imwidth && my_idx_reduce == 0 && tile_loc_w == 0 && threadIdx.x < block_max)
    {

        int tile_idx_h_img = blockIdx.y;
        int tile_idx_w_img = blockIdx.x * n_tiles_per_block + tile_idx_w;

        int tile_idx = tile_idx_h_img * n_tiles_w + tile_idx_w_img;

        kp_score[tile_idx] = max_score;
        kp_x[tile_idx] = max_x;
        kp_y[tile_idx] = max_y;

    }
}


void ORB_GPU::FAST_apply_NMS_G_reduce_unroll_reduce(int height, int width,
                                                    int tile_h, int tile_w,
                                                    int n_tiles_h, int n_tiles_w,
                                                    int warp_tile_h, int warp_tile_w,
                                                    int fuse_nms_L_with_nms_G,
                                                    int* image_unroll_gpu,
                                                    int* image_unroll_x_gpu,
                                                    int* image_unroll_y_gpu,
                                                    int* score_data_gpu,
                                                    int score_pitch,
                                                    int* keypoints_x,
                                                    int* keypoints_y,
                                                    int* keypoints_score,
                                                    cudaStream_t& cuda_stream)
{

    {
        int n_loc_per_thread = std::max(1, std::min(10, tile_w/3));

        std::vector<int> block_sizes_x = {128};

        int residual = 1E5;
        int block_size_x = -1;

        for(int i=0;i<block_sizes_x.size();i++)
        {
            int tmp_block_size_x = block_sizes_x[i];

            int tmp_residual = tmp_block_size_x % tile_w;

            if(tmp_residual < residual)
            {
                residual = tmp_residual;
                block_size_x = tmp_block_size_x;
            }
        }

        if(n_loc_per_thread > tile_h)
            n_loc_per_thread = tile_h;

        int n_threads_y_per_tile = (tile_h - 1) / n_loc_per_thread + 1;

        if(n_threads_y_per_tile * block_size_x > 1024)
            n_threads_y_per_tile = 1024 / block_size_x;


        int n_tiles_per_block = block_size_x / tile_w;

        //shared_mem_size


        int CUDA_NUM_BLOCKS_x = (n_tiles_w  - 1) / n_tiles_per_block + 1;
        int CUDA_NUM_BLOCKS_y = n_tiles_h;

//        std::cout << CUDA_NUM_BLOCKS_x << " "  << CUDA_NUM_BLOCKS_y << "\n";
//        std::cout << block_size_x << " "  << n_threads_y_per_tile << "\n";

        dim3 grid_dim(CUDA_NUM_BLOCKS_x, CUDA_NUM_BLOCKS_y, 1);
        dim3 block_dim(block_size_x, n_threads_y_per_tile, 1);


        int roiheight = (height - 2*0);
        int roiwidth  = (width - 2*0);

        //        Tile_unrolling_reduction_kernel<<<grid_dim, block_dim, 0, cuda_stream>>>(
        //                                                                                      height, width,
        //                                                                                      roiheight, roiwidth,
        //                                                                                      tile_h, tile_w,
        //                                                                                      n_tiles_h, n_tiles_w,
        //                                                                                      n_loc_per_thread,
        //                                                                                      n_threads_y_per_tile,
        //                                                                                      n_tiles_per_block,
        //                                                                                      score_data_gpu,
        //                                                                                      score_pitch,
        //                                                                                      keypoints_x,
        //                                                                                      keypoints_y,
        //                                                                                      keypoints_score,
        //                                                                                      fuse_nms_L_with_nms_G);


        Tile_unrolling_reduction_kernel_v2<<<grid_dim, block_dim, 0, cuda_stream>>>(
                                                                                      height, width,
                                                                                      tile_h, tile_w,
                                                                                      n_tiles_h, n_tiles_w,
                                                                                      n_loc_per_thread,
                                                                                      n_threads_y_per_tile,
                                                                                      n_tiles_per_block,
                                                                                      score_data_gpu,
                                                                                      score_pitch,
                                                                                      keypoints_x,
                                                                                      keypoints_y,
                                                                                      keypoints_score,
                                                                                      fuse_nms_L_with_nms_G);

        return;
    }




    // image unrolling
    {
        int roiheight = (height - 2*BORDER_SKIP);
        int roiwidth  = (width - 2*BORDER_SKIP);

        int n_threads =  n_tiles_h * roiwidth;

#ifdef GRID_LAUNCH

        {
            const int n_threads_per_tile = 5;


            int CUDA_NUM_BLOCKS_x = (width  - 1) / CUDA_NUM_THREADS_PER_BLOCK_x + 1;
            int CUDA_NUM_BLOCKS_y = n_tiles_h;

            dim3 grid_dim(CUDA_NUM_BLOCKS_x, CUDA_NUM_BLOCKS_y, 1);
            dim3 block_dim(CUDA_NUM_THREADS_PER_BLOCK_x, n_threads_per_tile, 1);

            Tile_unrolling_kernel_v4<<<grid_dim, block_dim, 0, cuda_stream>>>(
                                                                                n_threads,
                                                                                height, width,
                                                                                roiheight, roiwidth,
                                                                                tile_h, tile_w,
                                                                                n_tiles_h, n_tiles_w,
                                                                                warp_tile_h, warp_tile_w,
                                                                                score_data_gpu,
                                                                                score_pitch,
                                                                                image_unroll_gpu,
                                                                                image_unroll_x_gpu,
                                                                                image_unroll_y_gpu,
                                                                                fuse_nms_L_with_nms_G);

            cudaStreamSynchronize(cuda_stream);
            std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
        }

#else
        {



            int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK + 1;

            Tile_unrolling_kernel_v2<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                        n_threads,
                                                                                                        height, width,
                                                                                                        roiheight, roiwidth,
                                                                                                        tile_h, tile_w,
                                                                                                        n_tiles_h, n_tiles_w,
                                                                                                        warp_tile_h, warp_tile_w,
                                                                                                        score_data_nms_gpu,
                                                                                                        image_unroll_gpu,
                                                                                                        image_unroll_x_gpu,
                                                                                                        image_unroll_y_gpu,
                                                                                                        fuse_nms_L_with_nms_G);

        }

#endif
    }


    {
        // NMS reduction
        {
            int n_threads =  n_tiles_h * n_tiles_w * warp_tile_w;

            int CUDA_NUM_BLOCKS = (n_threads - 1) / CUDA_NUM_THREADS_PER_BLOCK + 1;

            Tile_reduction_kernel_v2<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                                        n_threads,
                                                                                                        tile_h, tile_w,
                                                                                                        n_tiles_h, n_tiles_w,
                                                                                                        warp_tile_h, warp_tile_w,
                                                                                                        image_unroll_gpu,
                                                                                                        image_unroll_x_gpu,
                                                                                                        image_unroll_y_gpu,
                                                                                                        keypoints_x,
                                                                                                        keypoints_y,
                                                                                                        keypoints_score);
        }
    }

}


}

