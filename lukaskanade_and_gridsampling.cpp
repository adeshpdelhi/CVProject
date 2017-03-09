#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/highgui/highgui.hpp"


#include <iostream>
#include <ctype.h>
#include <time.h>

using namespace cv;
using namespace std;


Point2f point;
bool addRemovePt = false;

// static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
// {
//     if( event == CV_EVENT_LBUTTONDOWN )
//     {
//         point = Point2f((float)x, (float)y);
//         addRemovePt = true;
//     }
// }

int main( int argc, char** argv )
{

    VideoCapture cap;
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 500, MAX_COUNT_INCREMENT = 100;
    bool needToInit = true;
    bool nightMode = false;

    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        cap.open(argc == 2 ? argv[1][0] - '0' : 0);
    else if( argc == 2 )
        cap.open(argv[1]);

    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }

    namedWindow( "LK Demo", 1 );

    Mat gray, prevGray, image;

    int step = 20; // 10 pixels spacing between kp's

    vector<Point2f> points[2];
    vector <bool> foreground;

    bool addPoints = false;
    time_t t1 = time(0) ;
    int color = 0;
    srand (time(NULL));
    vector<Point2f> new_points;
    bool increase_points = false;
    int iterator_count = 0;
    time_t r1 = time(0),r2;
    for(;;)
    {
        Mat frame;
        cap >> frame;
        Size reduced_size(640,480);
        cv::resize(frame, frame, reduced_size);
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);

        if( nightMode )
            image = Scalar::all(0);

        
        if( needToInit )
        {
            // automatic initialization
            // cout<<"initialization ========================================"<<endl;
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 1, Mat(), 3, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            addRemovePt = false;
      //       int offsetx;
      //       int offsety;
      //       points[1].clear();
      //       for (int y=step; y<frame.rows-step; y+=step){
		    //     for (int x=step; x<frame.cols-step; x+=step){
		    //     	offsetx =rand()%10;
		    //     	offsety = rand()%10;
		    //         Point2f point = Point2f((float)x+offsetx, (float)y+offsety);
		    //         points[1].push_back(point);

		    //     }
		    // }
            
            // cout<<"New size of feature points "<<points[1].size()<<endl;
		    color++;
            color= color%3;
            needToInit = false;
        }
        else if(increase_points){
            cout<<"Incrementing +++++++++++++"<<endl;
            goodFeaturesToTrack(gray, new_points, MAX_COUNT_INCREMENT, 0.01, 10, Mat(), 3, 0, 0.04);
            addRemovePt = false;
            int final_size = new_points.size() + points[1].size();
            points[1].insert(points[1].end(), new_points.begin(), new_points.end());
            cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            // points[1].resize(final_size);
            cout<<"Added points size "<<new_points.size()<<". Final size of feature points "<<points[1].size()<<endl;
            increase_points = false;
        }
        else if( !points[0].empty() )
        {
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            size_t i, k;
            foreground.clear();
		  	Mat_ <float> H = findHomography( points[0], points[1], CV_RANSAC );

            for (int i = 0; i < points[0].size(); ++i)
            {
                // if(norm(points[0][i] - points[1][i])>3)
                //     foreground.push_back(true);
                // else
                //     foreground.push_back(false);
	            Vec3f old_point = Vec3f((float)points[0][i].x, (float)points[0][i].y, 1.0);
                Mat new_point_matrix  = H*Mat(old_point);
                Point2f new_point = Point2f(new_point_matrix.at<float>(0,0)/new_point_matrix.at<float>(2,0),new_point_matrix.at<float>(1,0)/new_point_matrix.at<float>(2,0));
                // cout<<new_point_matrix.at<float>(2,0)<<endl;
                // cout<<old_point<<endl<<new_point<<endl<<endl;
                if(norm(points[1][i] - new_point)>3)
                    foreground.push_back(true);
                else
                    foreground.push_back(false);
            }

            for( i = k = 0; i < points[1].size(); i++ )
            {
                // if( addRemovePt )
                // {
                //     if( norm(point - points[1][i]) <= 5 )
                //     {
                //         addRemovePt = false;
                //         continue;
                //     }
                // }

                // if( !status[i] )
                //     continue;

                // points[1][k++] = points[1][i];
                
             //    if(color == 0)
	            //     circle( image, points[1][i], 3, Scalar(255,0,0), -1, 8);
	            // if(color == 1)
	            //     circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
	            // if(color == 2)
	            //     circle( image, points[1][i], 3, Scalar(0,0,255), -1, 8);
                if(foreground[i])
                    circle( image, points[1][i], 3, Scalar(0,0,255), -1, 8);
                else
                    circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
            }
            // points[1].resize(k);
        }

        // if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
        // {
        //     vector<Point2f> tmp;
        //     tmp.push_back(point);
        //     cornerSubPix( gray, tmp, winSize, cvSize(-1,-1), termcrit);
        //     points[1].push_back(tmp[0]);
        //     addRemovePt = false;
        // }

        needToInit = false;
        transpose(image, image);
        flip(image, image , 1);
        imshow("LK Demo", image);

        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        switch( c )
        {
        case 'r':
            needToInit = true;
            points[0].clear();
            points[1].clear();
            break;
        case 'c':
            points[0].clear();
            points[1].clear();
            break;
        case 'n':
            nightMode = !nightMode;
            break;
        }

        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
        time_t t2 = time(0) ;
        // int p=1;
        // while(p++<1000);
        if(difftime(t2,t1)>1){
        	t1 = time(0);
            // iterator_count++; 
            // if(iterator_count%3 == 0)
            {
                iterator_count=0;
                needToInit = true;
            }
            // else
                // increase_points = true;
              
        }

    }

    return 0;
}
