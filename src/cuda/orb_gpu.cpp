#include "cuda/orb_gpu.hpp"

#include<vector>

#include<opencv2/opencv.hpp>


#include<cuda.h>
#include<cuda_device_runtime_api.h>
#include<cuda_runtime_api.h>

#include<chrono>

#include<bitset>
#include <opencv2/imgproc/types_c.h>

using namespace cv;

namespace orb_cuda {


ORB_GPU::ORB_GPU(int im_height, int im_width,
                 int n_levels, float scale_factor,
                 int FAST_N_MIN, int FAST_N_MAX,
                 int th_FAST_MIN, int th_FAST_MAX,
                 int tile_h, int tile_w,
                 bool fixed_multi_scale_tile_size,
                 bool apply_nms_ms, bool nms_ms_mode_gpu,
                 std::string str_mask,
                 int device_id)
{
    cudaSetDevice(device_id);

    float scale = scale_factor;
    n_levels_ = n_levels;

    apply_nms_ms_ = apply_nms_ms && (n_levels_ > 1);
    nms_ms_mode_gpu_ = nms_ms_mode_gpu;

    patternSize_ = 16;

    th_FAST_MIN_ = th_FAST_MIN;
    th_FAST_MIN_ = th_FAST_MAX;
    FAST_N_MIN_ = FAST_N_MIN;
    FAST_N_MAX_ = FAST_N_MAX;

    threshold_ = th_FAST_MAX;

    scale_.push_back(1.0);
    inv_scale_.push_back(1.0);

    height_.push_back(im_height);
    width_.push_back(im_width);

    for(int i=1;i<n_levels;i++)
    {
        scale_.push_back(scale*scale_[i-1]);
        inv_scale_.push_back(1.0f/scale_[i]);

        height_.push_back(im_height * inv_scale_[i]);
        width_.push_back(im_width * inv_scale_[i]);
    }

    {
        masks_.resize(n_levels);

        cv::Mat mask = cv::imread(str_mask);

        if(mask.empty())
        {
            mask = cv::Mat(height_[0], width_[0], CV_8UC1);
            memset(mask.data,255,height_[0]*width_[0]);
        }
        else
            cv::cvtColor(mask,mask, CV_BGR2GRAY);


        for(int i=0;i<n_levels_;i++)
        {
            cv::Mat resize_mask;
            cv::resize(mask, resize_mask, cv::Size(width_[i], height_[i]),0,0,CV_INTER_NN);
            cv::threshold(resize_mask, resize_mask, 10, 255, CV_THRESH_BINARY);

            masks_[i].resize(width_[i] * height_[i]);
            masks_[i].pitch_ = width_[i];

            memcpy(masks_[i].cpu_data(), resize_mask.data, width_[i] * height_[i]);

            masks_[i].to_gpu();
        }
    }


    {
        image_.resize(n_levels_);
        image_gaussian_.resize(n_levels_);

        score_.resize(n_levels_);
        score_nms_.resize(n_levels_);

        umax_.resize(n_levels_);
        threshold_tab_.resize(n_levels_);

        pattern_x_.resize(n_levels_);
        pattern_y_.resize(n_levels_);

        gaussian_weights_.resize(n_levels_);
    }

    n_keypoints_.resize(n_levels);
    cuda_streams_.resize(n_levels);

    for(int i=0;i<n_levels;i++)
    {
        // dont used pitched memory for SLAM
        // because in the stereo-match GPU,
        // currently pitched memory support is not there, only linear memory is there.

        //        image_[i].resize_pitched(width_[i], height_[i]);
        //        image_gaussian_[i].resize_pitched(width_[i], height_[i]);
        //        score_[i].resize_pitched(width_[i], height_[i]);
        //        score_nms_[i].resize_pitched(width_[i], height_[i]);

        image_[i].resize(width_[i] * height_[i]);
        image_[i].pitch_ = width_[i];

        image_gaussian_[i].resize(height_[i] * width_[i]);
        image_gaussian_[i].pitch_ = width_[i];

        score_[i].resize(height_[i] * width_[i]);
        score_[i].pitch_ = width_[i] * 4;

        score_nms_[i].resize(height_[i] * width_[i]);
        score_nms_[i].pitch_ = width_[i] * 4;

        umax_[i].resize(CIRCULAR_HALF_PATCH_SIZE + 1);
        threshold_tab_[i].resize(512);

        pattern_x_[i].resize(512);
        pattern_y_[i].resize(512);

        gaussian_weights_[i].resize(7*7);

        n_keypoints_[i] = 0;

        cudaStreamCreate(&cuda_streams_[i]);
    }


    for(int i=0;i<n_levels;i++)
    {
        int* threshold_tab_cpu_ = threshold_tab_[i].cpu_data();

        for(int j = -255; j <= 255; j++ )
            threshold_tab_cpu_[j+255] = (int)(j < -threshold_ ? 1 : j > threshold_ ? 2 : 0);

        threshold_tab_[i].to_gpu();
    }


    for(int i=0;i<n_levels;i++)
    {
        int* umax_cpu_ = umax_[i].cpu_data();

        int v, v0, vmax = cvFloor(CIRCULAR_HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(CIRCULAR_HALF_PATCH_SIZE * sqrt(2.f) / 2);

        const double hp2 = CIRCULAR_HALF_PATCH_SIZE * CIRCULAR_HALF_PATCH_SIZE;

        for (v = 0; v <= vmax; ++v)
            umax_cpu_[v] = cvRound(sqrt(hp2 - v * v));

        // Make sure we are symmetric
        for (v = CIRCULAR_HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
        {
            while (umax_cpu_[v0] == umax_cpu_[v0 + 1])
                ++v0;
            umax_cpu_[v] = v0;
            ++v0;
        }
        umax_[i].to_gpu();
    }


    for(int i=0;i<n_levels;i++)
    {
        signed char* pattern_x_cpu_ = pattern_x_[i].cpu_data();
        signed char* pattern_y_cpu_ = pattern_y_[i].cpu_data();

        fill_bitpattern(pattern_x_cpu_, pattern_y_cpu_);

        pattern_x_[i].to_gpu();
        pattern_y_[i].to_gpu();
    }

    {
        for(int i=0;i<n_levels;i++)
        {
            float* gaussian_weights_cpu_ = gaussian_weights_[i].cpu_data();

            const float sigma = 10;
            const float sigma2 = sigma * sigma;

            float sum = 0;
            int count = 0;
            for(int j=-3;j<=3;j++)
                for(int k=-3;k<=3;k++)
                {
                    gaussian_weights_cpu_[count] =  exp(-(j*j+k*k)/(2*sigma2));
                    sum += gaussian_weights_cpu_[count];
                    count++;
                }

            for(int j=0;j<7*7;j++)
                gaussian_weights_cpu_[j] /= sum;

            gaussian_weights_[i].to_gpu();
        }

    }


    // NMS-G setup
    {
        tile_h_.resize(n_levels_);
        tile_w_.resize(n_levels_);

        n_tile_h_.resize(n_levels_);
        n_tile_w_.resize(n_levels_);

        warp_tile_h_.resize(n_levels_);
        warp_tile_w_.resize(n_levels_);

        tile_h_[0] = tile_h;
        tile_w_[0] = tile_w;

        image_unroll_.resize(n_levels_);
        image_unroll_x_.resize(n_levels_);
        image_unroll_y_.resize(n_levels_);

        for(int i=0;i<n_levels_;i++)
        {

            if(fixed_multi_scale_tile_size)

            {
                tile_h_[i] = tile_h_[0];
                tile_w_[i] = tile_w_[0];
            }
            else
            {
                tile_h_[i] = tile_h_[0] * inv_scale_[i];
                tile_w_[i] = tile_w_[0] * inv_scale_[i];
            }


            n_tile_h_[i] = ((height_[i] - 2*0) - 1) / tile_h_[i] + 1;
            n_tile_w_[i] = ((width_[i] - 2*0) - 1) / tile_w_[i] + 1;


            const int warp_size = 32;

            //// reduce_unroll_reduce requires less memory .
            //// disabled for experimenation insead allocating for unroll_reduce which covers all memory req.
            //            int tile_count_warped = ((tile_w_[i] - 1) / warp_size + 1) * warp_size; // simplified ceil(tile_count / warp_size);

            // unroll_reduce
            int tile_count_warped = ((tile_w_[i] * tile_h_[i] - 1) / warp_size + 1) * warp_size; // simplified ceil(tile_count / warp_size);

            int unroll_count = tile_count_warped * n_tile_h_[i]  * n_tile_w_[i];

            warp_tile_h_[i] = tile_count_warped / warp_size;
            warp_tile_w_[i] = warp_size;

            image_unroll_[i].resize(unroll_count);
            memset(image_unroll_[i].cpu_data(),0, sizeof(int)*unroll_count);
            image_unroll_[i].to_gpu();

            image_unroll_x_[i].resize(unroll_count);
            memset(image_unroll_x_[i].cpu_data(),0, sizeof(int)*unroll_count);
            image_unroll_x_[i].to_gpu();

            image_unroll_y_[i].resize(unroll_count);
            memset(image_unroll_y_[i].cpu_data(),0, sizeof(int)*unroll_count);
            image_unroll_y_[i].to_gpu();

            //            std::cout << tile_h_[i] << " " << tile_w_[i] << "\n";
            //            std::cout << n_tile_h_[i] << " " << n_tile_w_[i] << "\n";
            //            std::cout << warp_tile_h_[i] << " " << warp_tile_w_[i] << "\n";
            //            std::cout << tile_count_warped << " " << unroll_count << "\n";

        }
    }



    if(apply_nms_ms_)
    {
        nms_s_score_.resize(height_[0] * width_[0]);
        nms_s_level_.resize(height_[0] * width_[0]);

        s0_score_.resize(n_levels_ * height_[0] * width_[0]);
    }

    {
        int count =0;
        for(int i=0;i<n_levels_;i++)
        {
            level_offset_.push_back(count);
            count += n_tile_h_[i] * n_tile_w_[i];
        }

        x_offset_ = 0*count;
        y_offset_ = 1*count;
        s_offset_ = 2*count;
        l_offset_ = 3*count;
        a_offset_ = 4*count;

        max_kp_count_ = count;
        x_count_ = count;
        y_count_ = count;
        s_count_ = count;
        l_count_ = count;
        a_count_ = count;

        keypoints_.resize(x_count_ + y_count_ + s_count_ + l_count_ + a_count_);
        keypoints_desc_.resize(max_kp_count_ * 32);

        nms_ms_cpu_x_.resize(n_tile_h_[0]  * n_tile_w_[0]);
        nms_ms_cpu_y_.resize(n_tile_h_[0]  * n_tile_w_[0]);
        nms_ms_cpu_score_.resize(n_tile_h_[0]  * n_tile_w_[0]);
        nms_ms_cpu_level_.resize(n_tile_h_[0]  * n_tile_w_[0]);
        nms_ms_cpu_idx_.resize(n_tile_h_[0]  * n_tile_w_[0]);
    }


    // setup for NMS-MS GPU
    {
        grid_levels_.resize(max_kp_count_);
        grid_scale_factor_.resize(max_kp_count_);

        int* level_data = grid_levels_.cpu_data();
        float* scale_factor_data = grid_scale_factor_.cpu_data();

        int count = 0;
        for(int i=0;i<n_levels_;i++)
        {
            int n_grids = n_tile_h_[i] * n_tile_w_[i];
            for(int j=0;j<n_grids;j++)
            {
                level_data[count] = i;
                scale_factor_data[count] = scale_[i];
                count++;
            }
        }

        grid_levels_.to_gpu();
        grid_scale_factor_.to_gpu();
    }


    s0_score_.set_zero_gpu();

    fuse_nms_L_with_nms_G_ = true;

    //lookpu tables init
    {
        lookup_table_.resize(n_levels_);

        for(int i=0;i<n_levels_;i++)
        {
            SyncedMem<int>& lut = lookup_table_[i];
            lut.resize(0x0000FFFF);

            int* data = lut.cpu_data();

            for(int j=0;j<0x0000FFFF;j++)
            {
                int num =  j;
                int n_valid = 0;
                int valid_bit = 0x00008000;

                int need_further_check = true;

                for(int k=0;k<16;k++)
                {
                    if(num & valid_bit)
                        n_valid++;
                    else
                    {
                        if(n_valid >= FAST_N_MIN_ && n_valid <= FAST_N_MAX_)
                        {
                            need_further_check = false;
                            break;
                        }
                        else
                            n_valid = 0;
                    }

                    valid_bit >>= 1;
                }

                //                std::cout <<  need_further_check << " " << n_valid << " ";

                if(need_further_check)
                {
                    int num = j;
                    int valid_bit = 0x00008000;

                    for(int k=0;k<16;k++)
                    {
                        if(num & valid_bit)
                            n_valid++;
                        else
                            break;

                        valid_bit >>= 1;
                    }
                }

                //                std::cout <<  n_valid << " " << std::bitset<16>(j);
                if(n_valid >= FAST_N_MIN_ && n_valid <= FAST_N_MAX_)
                {
                    data[j] = 1;
                    //                    std::cout <<  " " << "found 1\n";
                }
                else
                {
                    data[j] = 0;
                    //                    std::cout <<  " " << "found 0\n";
                }
            }

            lut.to_gpu();
        }
    }


    dummy_kernel_launch_tosetup_context();

}

ORB_GPU::~ORB_GPU()
{

    for(int i=0;i<n_levels_;i++)
    {
        cudaStreamDestroy(cuda_streams_[i]);
    }

}




class tictoc
{
public:
    void tic(void)
    {
        t1 = std::chrono::steady_clock::now();
    }

    float toc(void)
    {
        t2 = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() * 1000.0;
        return time_ms;
    }

    void toc_print(void)
    {
        t2 = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() * 1000.0;
        std::cout << "Time elapsed = " << time_ms << " ms\n";
    }

private:
    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
    float time_ms;
};





//v2
void ORB_GPU::extract(const cv::Mat& image,
                      SyncedMem<int>& out_keypoints,
                      SyncedMem<unsigned char>& out_keypoints_desc)
{

    // compute pyramid for all levels async
    {
        //        cudaMemcpy(image_[0].gpu_data(), image.data, height_[0] * width_[0], cudaMemcpyHostToDevice);
        cudaMemcpy2D(image_[0].gpu_data(),image_[0].pitch_, image.data, width_[0], width_[0],height_[0], cudaMemcpyHostToDevice);

        // resize each image async
        for(int i=1;i<n_levels_;i++)
        {
            cudaStream_t& cuda_stream = cuda_streams_[i];

            Compute_pyramid(height_[0], width_[0],
                    height_[i], width_[i],
                    inv_scale_[i],
                    image_[0].gpu_data(),
                    image_[0].pitch_,
                    image_[i].gpu_data(),
                    image_[i].pitch_,
                    cuda_stream);
        }

        //        for(int i=0;i<n_levels_;i++)
        //        {
        //            cudaStream_t& cuda_stream = cuda_streams_[i];
        //            cudaStreamSynchronize(cuda_stream);

        //            cv::Mat img(height_[i], width_[i], CV_8UC1);
        //            cudaMemcpy2D(img.data, width_[i], image_[i].gpu_data(),image_[i].pitch_, width_[i], height_[i], cudaMemcpyDeviceToHost);

        //            {
        //                cudaError_t cuerror = cudaGetLastError();
        //                if(cuerror) std::cout <<  cuerror << " error incurred 1\n";
        //            }

        //            std::stringstream winname;
        //            winname << "img" << i;

        //            cv::imshow(winname.str(),img);
        //            cv::waitKey(0);

        //        }
    }

    // compute FAST score for all levels async
    for(int i=0;i<n_levels_;i++)
    {
        cudaStream_t& cuda_stream = cuda_streams_[i];

        //// no need to synchronize because different scales have different stream
        //// and same scale stream always queue the jobs launched in them
        //// also its overhead intriduces a delay of us
        //   cudaStreamSynchronize(cuda_stream);

        //                        FAST_compute_score(height_[i], width_[i],
        //                                           image_[i].gpu_data(),
        //                                           threshold_tab_[i].gpu_data(),
        //                                           score_[i].gpu_data(),
        //                                           cuda_stream);

        //        FAST_compute_score_lookpup(height_[i], width_[i],
        //                                   image_[i].gpu_data(),
        //                                   image_[i].pitch_,
        //                                   threshold_,
        //                                   lookup_table_[i].gpu_data(),
        //                                   score_[i].gpu_data(),
        //                                   score_[i].pitch_ / 4,
        //                                   cuda_stream);

        FAST_compute_score_lookpup_mask(height_[i], width_[i],
                                        image_[i].gpu_data(),
                                        image_[i].pitch_,
                                        masks_[i].gpu_data(),
                                        masks_[i].pitch_,
                                        threshold_,
                                        lookup_table_[i].gpu_data(),
                                        score_[i].gpu_data(),
                                        score_[i].pitch_ / 4,
                                        cuda_stream);

    }


    if(!fuse_nms_L_with_nms_G_)
    {
        // apply NMS-L for all levels  async
        for(int i=0;i<n_levels_;i++)
        {
            cudaStream_t& cuda_stream = cuda_streams_[i];
            FAST_apply_NMS_L(height_[i], width_[i],
                             score_[i].gpu_data(),
                             score_[i].pitch_ / 4,
                             score_nms_[i].gpu_data(),
                             score_nms_[i].pitch_ / 4,
                             cuda_stream);


        }
    }


    // compute keypoints on CPU for all levels sync and send to GPU async
    for(int i=0;i<n_levels_;i++)
    {
        cudaStream_t& cuda_stream = cuda_streams_[i];

        if(fuse_nms_L_with_nms_G_)
        {
            FAST_apply_NMS_G_reduce_unroll_reduce(height_[i], width_[i],
                                                  tile_h_[i], tile_w_[i],
                                                  n_tile_h_[i], n_tile_w_[i],
                                                  warp_tile_h_[i], warp_tile_w_[i],
                                                  fuse_nms_L_with_nms_G_,
                                                  image_unroll_[i].gpu_data(),
                                                  image_unroll_x_[i].gpu_data(),
                                                  image_unroll_y_[i].gpu_data(),
                                                  score_[i].gpu_data(),
                                                  score_[i].pitch_ / 4,
                                                  keypoints_.gpu_data() + x_offset_ + level_offset_[i],
                                                  keypoints_.gpu_data() + y_offset_ + level_offset_[i],
                                                  keypoints_.gpu_data() + s_offset_ + level_offset_[i],
                                                  cuda_stream);

            //            FAST_apply_NMS_G_unroll_reduce(height_[i], width_[i],
            //                                           tile_h_[i], tile_w_[i],
            //                                           n_tile_h_[i], n_tile_w_[i],
            //                                           warp_tile_h_[i], warp_tile_w_[i],
            //                                           fuse_nms_L_with_nms_G_,
            //                                           image_unroll_[i].gpu_data(),
            //                                           score_[i].gpu_data(),
            //                score_[i].pitch_,
            //                                           keypoints_.gpu_data() + x_offset_ + level_offset_[i],
            //                                           keypoints_.gpu_data() + y_offset_ + level_offset_[i],
            //                                           keypoints_.gpu_data() + s_offset_ + level_offset_[i],
            //                                           cuda_stream);
        }
        else
        {
            FAST_apply_NMS_G_reduce_unroll_reduce(height_[i], width_[i],
                                                  tile_h_[i], tile_w_[i],
                                                  n_tile_h_[i], n_tile_w_[i],
                                                  warp_tile_h_[i], warp_tile_w_[i],
                                                  fuse_nms_L_with_nms_G_,
                                                  image_unroll_[i].gpu_data(),
                                                  image_unroll_x_[i].gpu_data(),
                                                  image_unroll_y_[i].gpu_data(),
                                                  score_nms_[i].gpu_data(),
                                                  score_nms_[i].pitch_ / 4,
                                                  keypoints_.gpu_data() + x_offset_ + level_offset_[i],
                                                  keypoints_.gpu_data() + y_offset_ + level_offset_[i],
                                                  keypoints_.gpu_data() + s_offset_ + level_offset_[i],
                                                  cuda_stream);

            //            FAST_apply_NMS_G_unroll_reduce(height_[i], width_[i],
            //                                           tile_h_[i], tile_w_[i],
            //                                           n_tile_h_[i], n_tile_w_[i],
            //                                           warp_tile_h_[i], warp_tile_w_[i],
            //                                           fuse_nms_L_with_nms_G_,
            //                                           image_unroll_[i].gpu_data(),
            //                                           score_nms_[i].gpu_data(),
            //                score_nms_[i].pitch_,
            //                                           keypoints_.gpu_data() + x_offset_ + level_offset_[i],
            //                                           keypoints_.gpu_data() + y_offset_ + level_offset_[i],
            //                                           keypoints_.gpu_data() + s_offset_ + level_offset_[i],
            //                                           cuda_stream);
        }



    }


    //// NMS-MS
    if(apply_nms_ms_)
    {
        if(nms_ms_mode_gpu_)
        {
            // NMS-MS GPU
            {
                for(int i=0;i<n_levels_;i++)
                {
                    cudaStream_t& cuda_stream = cuda_streams_[i];
                    cudaStreamSynchronize(cuda_stream);
                }

                cudaStream_t& cuda_stream = cuda_streams_[0];

                nms_s_score_.set_zero_gpu();

                apply_NMS_MS_S_L(height_[0], width_[0],
                        n_levels_,
                        max_kp_count_,
                        keypoints_.gpu_data() + x_offset_,
                        keypoints_.gpu_data() + y_offset_,
                        keypoints_.gpu_data() + s_offset_,
                        grid_levels_.gpu_data(),
                        grid_scale_factor_.gpu_data(),
                        s0_score_.gpu_data(),
                        nms_s_score_.gpu_data(),
                        nms_s_level_.gpu_data(),
                        cuda_stream);

                cudaStreamSynchronize(cuda_stream);
            }

        }
        else
        {
            // NMS-MS CPU
            for(int i=0;i<n_levels_;i++)
            {
                cudaStream_t& cuda_stream = cuda_streams_[i];
                cudaStreamSynchronize(cuda_stream);
            }

            keypoints_.to_cpu(x_count_ + y_count_ + s_count_);

            FAST_apply_NMS_MS_cpu();

            keypoints_.to_gpu(x_count_ + y_count_ + s_count_);
        }
    }


    // obtain keypoints
    {


        FAST_obtain_keypoints();

        keypoints_.to_gpu(x_count_ + y_count_ + s_count_);
    }


    // compute orientations async
    for(int i=0;i<n_levels_;i++)
    {
        cudaStream_t& cuda_stream = cuda_streams_[i];

        FAST_compute_orientation(height_[i], width_[i],
                                 image_[i].gpu_data(),
                                 image_[i].pitch_,
                                 n_keypoints_[i],
                                 keypoints_.gpu_data() + x_offset_ + level_offset_[i],
                                 keypoints_.gpu_data() + y_offset_ + level_offset_[i],
                                 keypoints_.gpu_data() + s_offset_ + level_offset_[i],
                                 (float*)keypoints_.gpu_data() + a_offset_ + level_offset_[i],
                                 umax_[i].gpu_data(),
                                 cuda_stream);
    }



    //compute gaussian async
    for(int i=0;i<n_levels_;i++)
    {
        cudaStream_t& cuda_stream = cuda_streams_[i];

        compute_gaussian(height_[i], width_[i],
                         image_[i].gpu_data(),
                         image_[i].pitch_,
                         image_gaussian_[i].gpu_data(),
                         image_gaussian_[i].pitch_,
                         gaussian_weights_[i].gpu_data(),
                         cuda_stream);
    }


    //compute descriptor
    for(int i=0;i<n_levels_;i++)
    {
        cudaStream_t& cuda_stream = cuda_streams_[i];

        ORB_compute_descriptor(height_[i], width_[i],
                               image_gaussian_[i].gpu_data(),
                               image_gaussian_[i].pitch_,
                               pattern_x_[i].gpu_data(),
                               pattern_y_[i].gpu_data(),
                               n_keypoints_[i],
                               keypoints_.gpu_data() + x_offset_ + level_offset_[i],
                               keypoints_.gpu_data() + y_offset_ + level_offset_[i],
                               (float*)keypoints_.gpu_data() + a_offset_ + level_offset_[i],
                               keypoints_desc_.gpu_data() + 32 * level_offset_[i],
                               cuda_stream);
    }

    //copy to output
    {
        int n_total_keypoints = 0;
        for(int i=0;i<n_levels_;i++)
            n_total_keypoints += n_keypoints_[i];

        out_keypoints.resize(n_total_keypoints * (1 + 1 + 1 + 1 + 1 + 1)); // x y s a l size
        out_keypoints_desc.resize(32 * n_total_keypoints);

        int kp_offset = 0;
        for(int i=0;i<n_levels_;i++)
        {
            cudaStream_t& cuda_stream = cuda_streams_[i];

            const int out_x_offset  = 0 * n_total_keypoints;
            const int out_y_offset  = 1 * n_total_keypoints;
            const int out_s_offset  = 2 * n_total_keypoints; // score or response
            const int out_a_offset  = 3 * n_total_keypoints; // angle
            const int out_l_offset  = 4 * n_total_keypoints; // level or octave
            const int out_sz_offset = 5 * n_total_keypoints; // size

            const int out_level_offset = kp_offset;

            copy_output(n_keypoints_[i],
                        i, width_[i], scale_[i],
                        keypoints_.gpu_data() + x_offset_ + level_offset_[i],
                        keypoints_.gpu_data() + y_offset_ + level_offset_[i],
                        keypoints_.gpu_data() + s_offset_ + level_offset_[i],
                        (float*)keypoints_.gpu_data() + a_offset_ + level_offset_[i],
                        out_keypoints.gpu_data() + out_x_offset + out_level_offset,
                        out_keypoints.gpu_data() + out_y_offset + out_level_offset,
                        out_keypoints.gpu_data() + out_s_offset + out_level_offset,
                        (float*)out_keypoints.gpu_data() + out_a_offset + out_level_offset,
                        out_keypoints.gpu_data() + out_l_offset + out_level_offset,
                        out_keypoints.gpu_data() + out_sz_offset + out_level_offset,
                        cuda_stream
                        );

            kp_offset += n_keypoints_[i];
        }

        int offset =  0;
        for(int i=0;i<n_levels_;i++)
        {
            cudaStream_t& cuda_stream = cuda_streams_[i];

            cudaMemcpyAsync(out_keypoints_desc.gpu_data() + offset,
                            keypoints_desc_.gpu_data() + 32 * level_offset_[i],
                            n_keypoints_[i]*32,
                            cudaMemcpyDeviceToDevice,
                            cuda_stream);

            offset += 32 * n_keypoints_[i];
        }

        for(int i=0;i<n_levels_;i++)
        {
            cudaStream_t& cuda_stream = cuda_streams_[i];
            cudaStreamSynchronize(cuda_stream);
        }

    }

}


}

