/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;


int main(int argc, char **argv)
{

//    {cv::VideoCapture vcl(0);


//    while(1)
//    {

//        cv::Mat image;
//        vcl >> image;

//        cv::imshow("dfj",image);
//        cv::waitKey(1);

//    }

//    return 0;


//}
    //    if(argc != 3)
    //    {
    //        cerr << endl << "Usage: ./stereo_live path_to_vocabulary path_to_settings" << endl;
    //        return 1;
    //    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;

    int nImages = vstrImageLeft.size();

    std::string str_vocab = "/home/isl-server/ashish/libraries/orb_slam2/ORB_SLAM2-master/Vocabulary/ORBvoc.txt";
    //    std::string str_settings = "/home/isl-server/ashish/libraries/orb_slam2/ORB_SLAM2-master/Examples/Stereo/KITTI00-02.yaml";
    std::string str_settings = "/home/isl-server/ashish/libraries/orb_slam2/ORB_SLAM2-IROS_2022_new/Examples/Stereo/stereo_rig_lennovo.yaml";
//    std::string str_settings = "/home/isl-server/ashish/libraries/orb_slam2/ORB_SLAM2-master/Examples/Stereo/stereo_rig_lennovo_fisheye.yaml";
//    std::string str_settings = "/home/isl-server/ashish/libraries/orb_slam2/ORB_SLAM2-master/Examples/Stereo/stereo_rig_elp_global_shutter.yaml";



    cv::Rect valid_roi;

    cv::Mat left_rectify_map_x;
    cv::Mat left_rectify_map_y;
    cv::Mat right_rectify_map_x;
    cv::Mat right_rectify_map_y;

    {
//        cv::FileStorage fs1("/home/isl-server/ashish/stereo_rig_lennovo/rectify_params_1280x720.txt", cv::FileStorage::READ);
//        cv::FileStorage fs1("/home/isl-server/ashish/stereo_rig_lennovo/rectify_params_640x480.txt", cv::FileStorage::READ);
                cv::FileStorage fs1("/home/isl-server/ashish/stereo_rig_lennovo/rectify_params_320x240.txt", cv::FileStorage::READ);
//            cv::FileStorage fs1("/home/isl-server/ashish/stereo_rig_lennovo_320x240/rectify_params_320x240.txt", cv::FileStorage::READ);
//        cv::FileStorage fs1("/home/isl-server/ashish/stereo_rig_lennovo_fisheye/rectify_params_320x240.txt", cv::FileStorage::READ);

//        cv::FileStorage fs1("/home/isl-server/ashish/stereo_rig_elp_global_shutter/rectify_params_640x480.txt", cv::FileStorage::READ);


        fs1["valid_roi"] >> valid_roi;
        fs1["left_rectify_map_x"] >> left_rectify_map_x;
        fs1["left_rectify_map_y"] >> left_rectify_map_y;
        fs1["right_rectify_map_x"] >> right_rectify_map_x;
        fs1["right_rectify_map_y"] >> right_rectify_map_y;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(str_vocab, str_settings,ORB_SLAM2::System::STEREO,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imLeft, imRight;

    nImages = 200;

    cv::VideoCapture vcl(0);
    cv::VideoCapture vcr(1);

//        int im_width = 1280;
//        int im_height = 720;

//        int im_width = 640;
//        int im_height = 480;

    int im_width = 320;
    int im_height = 240;



//            int resize_width = 1280;
//            int resize_height = 720;

//    int resize_width = 640;
//    int resize_height = 480;

        int resize_width = 320;
        int resize_height = 240;


    vcl.set(CV_CAP_PROP_FRAME_HEIGHT, im_height);
    vcl.set(CV_CAP_PROP_FRAME_WIDTH, im_width);
    vcr.set(CV_CAP_PROP_FRAME_HEIGHT, im_height);
    vcr.set(CV_CAP_PROP_FRAME_WIDTH, im_width);

    cv::Mat lleft;
    cv::Mat lright;
    vcl >> lleft;
    vcr >> lright;

    float total_time = 0;

    int count = 0;
    while(1)
    {

        {
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

//            vcl.grab();
//            vcr.grab();

            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            std::cout << __func__ << " -grab time = " << ttrack * 1000.0 << " ms \n";

            // Read left and right images from file
            cv::Mat lleft;
            cv::Mat lright;

            t1 = std::chrono::steady_clock::now();

            vcl >> lleft;
            vcr >> lright;

            t2 = std::chrono::steady_clock::now();
            ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            std::cout << __func__ << " -fill time = " << ttrack * 1000.0 << " ms \n";


            t1 = std::chrono::steady_clock::now();

            cv::cvtColor(lleft,lleft,CV_BGR2GRAY);
            cv::cvtColor(lright,lright,CV_BGR2GRAY);

            t2 = std::chrono::steady_clock::now();
            ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            std::cout << __func__ << " -cvtcolor time = " << ttrack * 1000.0 << " ms \n";

//            cv::imshow("vcl",lleft);
//            cv::imshow("vcr",lright);
//            cv::waitKey(1);

            t1 = std::chrono::steady_clock::now();

            cv::remap(lleft, lleft, left_rectify_map_x, left_rectify_map_y, cv::INTER_LINEAR);
            cv::remap(lright, lright, right_rectify_map_x, right_rectify_map_y, cv::INTER_LINEAR);

            t2 = std::chrono::steady_clock::now();
            ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            std::cout << __func__ << " -remap time = " << ttrack * 1000.0 << " ms \n";

            //        cv::rectangle(imLeft, left_valid_roi, cv::Scalar(0,0,0),2);
            //        cv::rectangle(imRight, right_valid_roi, cv::Scalar(0,0,0),2);

            // if you crop, then shift the center in the camera intrinsics.
            // focal length remains same
            t1 = std::chrono::steady_clock::now();

//            lleft(valid_roi).copyTo(imLeft);
//            lright(valid_roi).copyTo(imRight);

            lleft.copyTo(imLeft);
            lright.copyTo(imRight);

//            cv::imshow("vclm",lleft);
//            cv::imshow("vcrm",lright);
//            cv::waitKey(1);

            t2 = std::chrono::steady_clock::now();
            ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            std::cout << __func__ << " -roi copy time = " << ttrack * 1000.0 << " ms \n";

            t1 = std::chrono::steady_clock::now();

            cv::resize(imLeft, imLeft, cv::Size(resize_width, resize_height));
            cv::resize(imRight, imRight, cv::Size(resize_width, resize_height));

//            cv::imshow("vclml",imLeft);
//            cv::imshow("vcrmr",imRight);
//            cv::waitKey(1);

            t2 = std::chrono::steady_clock::now();
            ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            std::cout << __func__ << " -resize time = " << ttrack * 1000.0 << " ms \n";
        }
        //        lleft.copyTo(imLeft);
        //        lright.copyTo(imRight);

        //        cv::imshow("rvcl",imLeft);
        //        cv::imshow("rvcr",imRight);
        //        cv::waitKey(0);
        //       unsigned char key = cv::waitKey(1);

        //       if(key == 32)
        //           break;

        double tframe =  std::time(0);// vTimestamps[ni];

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the images to the SLAM system
        cv::Mat pose = SLAM.TrackStereo(imLeft,imRight,tframe);
        if(!pose.empty())
            std::cout << pose << "\n";

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        total_time += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        //        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        count++;
        std::cout << count << "\n";



        if(count >=nImages)
            break;
    }

    std::cout << "Total time = " << total_time/ nImages << "\n";
    // Stop all threads
    SLAM.Shutdown();

    //    // Tracking time statistics
    //    sort(vTimesTrack.begin(),vTimesTrack.end());
    //    float totaltime = 0;
    //    for(int ni=0; ni<nImages; ni++)
    //    {
    //        totaltime+=vTimesTrack[ni];
    //    }
    //    cout << "-------" << endl << endl;
    //    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    //    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    return 0;
}
