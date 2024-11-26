/**
* This file is part of  Jetson-SLAM.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

#include "cuda/orb_gpu.hpp"

using namespace cv;
using namespace std;
using namespace orb_cuda;

namespace Jetson_SLAM
{

class ORBExtractor
{
public:
    
    ORBExtractor(int im_height, int im_width,
                 float scale_factor, int n_levels,
                 int FAST_N_MIN,
                 int FAST_N_MAX,
                 int th_FAST_MIN,
                 int th_FAST_MAX,
                 std::string str_mask,
                 int tile_h, int tile_w,
                 bool fixed_multi_scale_tile_size,
                 bool apply_nms_ms, bool nms_ms_mode_gpu,
                 bool use_gpu = false);

    ~ORBExtractor();


    void extract(const cv::Mat& image,
                 SyncedMem<int> &keypoints_,
                 SyncedMem<unsigned char> &keypoints_desc);

    int inline get_levels()
    {
        return n_levels_;
    }

    float inline get_scale_factor()
    {
        return scale_factor_;
    }

    inline const std::vector<float> get_scale_factors()
    {
        return scale_;
    }

    inline std::vector<float> get_inverse_scale_factors()
    {
        return inv_scale_;
    }

    inline std::vector<float> get_scale_sigma_squares()
    {
        return level_sigma2_;
    }

    inline std::vector<float> get_inverse_scale_sigma_squares()
    {
        return inv_level_sigma2_;
    }


    orb_cuda::ORB_GPU* orb_gpu_;


protected:

    std::vector<int> height_;
    std::vector<int> width_;

    std::vector<float> scale_;
    std::vector<float> inv_scale_;
    std::vector<float> level_sigma2_;
    std::vector<float> inv_level_sigma2_;

    std::vector<int> max_features_per_scale_;

    int n_levels_;
    float scale_factor_;
    bool use_gpu_;
};

} //namespace ORB_SLAM

#endif

