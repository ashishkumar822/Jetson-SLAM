#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "ORBextractor.h"

#include<pcl/console/time.h>

#include<ctime>

#include<chrono>

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

ORBExtractor::ORBExtractor(int im_height, int im_width,
                           float scale_factor, int n_levels,
                           int FAST_N_MIN,
                           int FAST_N_MAX,
                           int th_FAST_MIN,
                           int th_FAST_MAX,
                           string str_mask,
                           int tile_h, int tile_w,
                           bool fixed_multi_scale_tile_size,
                           bool apply_nms_ms, bool nms_ms_mode_gpu,
                           bool use_gpu)
{
    n_levels_ = n_levels;
    scale_factor_ = scale_factor;
    use_gpu_ = use_gpu;


    scale_.resize(n_levels_);
    inv_scale_.resize(n_levels_);

    level_sigma2_.resize(n_levels_);
    inv_level_sigma2_.resize(n_levels_);

    scale_[0] = 1.0f;
    level_sigma2_[0] = 1.0f;

    for(int i=1; i<n_levels_; i++)
    {
        scale_[i] = scale_[i-1] * scale_factor_;
        level_sigma2_[i] = scale_[i] * scale_[i];
    }

    for(int i=0; i<n_levels_; i++)
    {
        inv_scale_[i] = 1.0f / scale_[i];
        inv_level_sigma2_[i] = 1.0f / level_sigma2_[i];
    }

    height_.resize(n_levels_);
    width_.resize(n_levels_);

    for(int i=0;i<n_levels_;i++)
    {
        height_[i] = im_height * inv_scale_[i];
        width_[i] = im_width * inv_scale_[i];
    }


    orb_gpu_ = NULL;
    //    if(use_gpu)
    orb_gpu_ = new ORB_GPU(im_height, im_width,
                           n_levels_, scale_factor_,
                           FAST_N_MIN,
                           FAST_N_MAX,
                           th_FAST_MIN,
                           th_FAST_MAX,
                           tile_h, tile_w,
                           fixed_multi_scale_tile_size,
                           apply_nms_ms,
                           nms_ms_mode_gpu,
                           str_mask,
                           0);

}

ORBExtractor::~ORBExtractor()
{
    if(orb_gpu_)delete orb_gpu_;
}


void ORBExtractor::extract(const cv::Mat& image,
                           SyncedMem<int>& keypoints,
                           SyncedMem<unsigned char>& keypoints_desc)
{

    //    if(use_gpu_)
    orb_gpu_->extract(image,
                      keypoints,
                      keypoints_desc);
}

} //namespace ORB_SLAM
