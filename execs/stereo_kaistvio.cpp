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

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

void LoadImages_VO_seq(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
//    if(argc != 4)
//    {
//        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
//        return 1;
//    }

    std::string str_vocab = "/home/isl-server/ashish/libraries/orb_slam2/ORB_SLAM2-IROS_2022_new/Vocabulary/ORBvoc.txt";
    std::string str_settings = "/home/isl-server/ashish/libraries/orb_slam2/ORB_SLAM2-IROS_2022_new/Examples/Stereo/KITTI00-02.yaml";
//std::string str_seq_path  = "/home/isl-server/ashish/datasets/kitti_sequences/raw/2011_09_26/2011_09_26_drive_0051_sync";
//    std::string str_seq_path  = "/home/isl-server/ashish/datasets/kitti_sequences/raw/2011_09_26/2011_09_26_drive_0061_sync";

//    std::string str_settings = "/home/isl-server/ashish/libraries/orb_slam2/ORB_SLAM2-IROS_2022_new/Examples/Stereo/KITTI04-12.yaml";
    std::string str_seq_path  = "/media/isl-server/My Research/datasets/kitti/vo_sequences/dataset/sequences/01/";


    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
//    LoadImages(string(str_seq_path), vstrImageLeft, vstrImageRight, vTimestamps);
    LoadImages_VO_seq(string(str_seq_path), vstrImageLeft, vstrImageRight, vTimestamps);

    int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(str_vocab, str_settings,ORB_SLAM2::System::STEREO,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;


    {cv::Mat im(200,200, CV_8UC1);
    cv::imshow("he",im);
    cv::waitKey(0);}

    // Main loop
    cv::Mat imLeft, imRight;
//    nImages = 4;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
//        imLeft = cv::imread(vstrImageLeft[76],CV_LOAD_IMAGE_UNCHANGED);
//        imRight = cv::imread(vstrImageRight[76],CV_LOAD_IMAGE_UNCHANGED);

        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);

//        cv::boxFilter(imLeft, imLeft, -1,cv::Size(3,3));
//        cv::boxFilter(imRight, imRight, -1,cv::Size(3,3));



//        cv::Mat l1;
//        cv::Mat l2;


////        cv::bilateralFilter(imLeft, l1, -1,3,3);
////        cv::bilateralFilter(imRight, l2, -1,3,3);

////        cv::GaussianBlur(imLeft, imLeft, cv::Size(7,7),3,3);
////        cv::GaussianBlur(imRight, imRight, cv::Size(7,7),3,3);

//        cv::Mat kernel(3,3, CV_32FC1);
//        ((float*)kernel.data)[0] = 0;
//        ((float*)kernel.data)[1] = -1;
//        ((float*)kernel.data)[2] = 0;
//        ((float*)kernel.data)[3] = -1;
//        ((float*)kernel.data)[4] = 5;
//        ((float*)kernel.data)[5] = -1;
//        ((float*)kernel.data)[6] = 0;
//        ((float*)kernel.data)[7] = -1;
//        ((float*)kernel.data)[8] = 0;

//        cv::filter2D(imLeft, imLeft, -1, kernel);
//        cv::filter2D(imRight, imRight, -1, kernel);

//        imLeft = l1;
//        imRight = l2;

        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the images to the SLAM system
        SLAM.TrackStereo(imLeft,imRight,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();


        vTimesTrack[ni]=ttrack;

//        // Wait to load the next frame
//        double T=0;
//        if(ni<nImages-1)
//            T = vTimestamps[ni+1]-tframe;
//        else if(ni>0)
//            T = tframe-vTimestamps[ni-1];

//        if(ttrack<T)
//            usleep((T-ttrack)*1e6);

//        usleep(10000);

//        cv::imwrite("/home/isl-server/ashish/temp_l.png", imLeft);


//        cv::imshow("imLeft", imLeft);
//        cv::imshow("imRight", imRight);
//        cv::waitKey(0);



        std::cout << ni << "\n";

    }



    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");


    cv::Mat im(200,200, CV_8UC1);
    cv::imshow("he",im);
    cv::waitKey(0);

    // Stop all threads
    SLAM.Shutdown();


    return 0;
}

void LoadImages_VO_seq(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}


void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/timestamps.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_00/data/";
    string strPrefixRight = strPathToSequence + "/image_01/data/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(10) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}
