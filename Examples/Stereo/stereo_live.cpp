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
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/videoio/legacy/constants_c.h>

#include<System.h>

using namespace std;


int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage: ./stereo_live path_to_vocabulary path_to_settings" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;

    int nImages = vstrImageLeft.size();

    std::string str_vocab = argv[1];
    std::string str_settings = argv[2];

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    Jetson_SLAM::System SLAM(str_vocab, str_settings, Jetson_SLAM::System::STEREO,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imLeft, imRight;

    cv::VideoCapture vcl(0);
    cv::VideoCapture vcr(1);

    int im_width = 320;
    int im_height = 240;

    int resize_width = 320;
    int resize_height = 240;


    vcl.set(CV_CAP_PROP_FRAME_HEIGHT, im_height);
    vcl.set(CV_CAP_PROP_FRAME_WIDTH, im_width);
    vcr.set(CV_CAP_PROP_FRAME_HEIGHT, im_height);
    vcr.set(CV_CAP_PROP_FRAME_WIDTH, im_width);


    float total_time = 0;

    int count = 0;
    while(1)
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack;

        t1 = std::chrono::steady_clock::now();

        vcl >> imLeft;
        vcr >> imRight;

        double tframe =  std::time(0);// vTimestamps[ni];

        // Pass the images to the SLAM system
        cv::Mat pose = SLAM.TrackStereo(imLeft,imRight,tframe);
        if(!pose.empty())
            std::cout << pose << "\n";

//        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//        std::cout << "Track time = " << ttrack << "\n";

        if(count >=nImages)
            break;
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

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    return 0;
}
