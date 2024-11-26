
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>
#include<unistd.h>


using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./rgbd_live path_to_settings path_to_sequence" << endl;
        return 1;
    }

    std::string str_vocab = argv[1];
    std::string str_settings = argv[2];

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    Jetson_SLAM::System SLAM(str_vocab,str_settings,Jetson_SLAM::System::RGBD,true);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;

    // Main loop
    cv::Mat imRGB, imD;

    cv::VideoCapture v_rgb   = cv::VideoCapture("output.avi");
    cv::VideoCapture v_depth = cv::VideoCapture("depth.mkv");

    v_depth.set(cv::CAP_PROP_CONVERT_RGB, 0);

    int id = 0;
    while(1)
    {
        v_rgb >> imRGB;
        v_depth >> imD;

        if(imRGB.empty())
        {
            cerr << endl << "video is empty" << endl;
            return 1;
        }

        double tframe =  std::time(0);// vTimestamps[ni];

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#endif

        // Pass the images to the SLAM system
        cv::Mat pose = SLAM.TrackRGBD(imRGB,imD,tframe);
        if(!pose.empty())
            std::cout << pose << "\n";


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
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

