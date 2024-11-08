
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
//    if(argc != 5)
//    {
//        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
//        return 1;
//    }

    //    if(argc != 4)
    //    {
    //        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
    //        return 1;
    //    }

        std::string str_vocab = "/home/isl-server/ashish/libraries/orb_slam2/ORB_SLAM2-IROS_2022_new/Vocabulary/ORBvoc.txt";
        std::string str_settings = "/home/isl-server/ashish/libraries/orb_slam2/ORB_SLAM2-IROS_2022_new/Examples/RGB-D/rgbd_live.yaml";


        std::string str_seq_path  = "/home/isl-server/ashish/datasets/generalautonomy/fridge_open/";
        std::string str_seq_dir  = "data_20240814_183747/";
        std::string str_seq_cam  = "right_gripper/";


    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(str_vocab,str_settings,ORB_SLAM2::System::RGBD,true);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;

    // Main loop
    cv::Mat imRGB, imD;

    int id = 0;
    while(1)
    {
        std::stringstream idx;
        idx << id++;
        // Read image and depthmap from file
        imRGB = cv::imread("/home/isl-server/ashish/datasets/generalautonomy/small_any_pick/data_20240817_185627/right_gripper/RGBD_aligned/rgb_" +idx.str() +".png");
        imD = cv::imread("/home/isl-server/ashish/datasets/generalautonomy/small_any_pick/data_20240817_185627/right_gripper/RGBD_aligned/depth_" +idx.str() +".png", cv::IMREAD_UNCHANGED);

        std::cout << imRGB.rows << " " << imRGB.cols << "\n";
        std::cout << imD.rows << " " << imD.cols << "\n";

        cv::resize(imRGB, imRGB, cv::Size(imD.cols, imD.rows));

        cv::imshow("cap_rgb", imRGB);
        cv::imshow("cap_d", imD);
        cv::waitKey(5);



        if(imRGB.empty())
        {
            cerr << endl << "video is empty" << endl;
            return 1;
        }

        double tframe =  std::time(0);// vTimestamps[ni];

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the images to the SLAM system
        cv::Mat pose = SLAM.TrackRGBD(imRGB,imD,tframe);
        if(!pose.empty())
            std::cout << pose << "\n";


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        usleep(1000);
    }

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

//    // Save camera trajectory
//    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
//    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

