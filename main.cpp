#include <iostream>
#include <ctype.h>
#include <time.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;

bool REMOVE_UNTRACKED_POINTS = false;
vector<Point2f> temp_points[2];
vector<Point2f> points[2];
vector <bool> foreground;
vector<Point2f> new_points;

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
    namedWindow( "Output", 1 );
    Mat gray, prevGray, image;
    int step = 20; // 10 pixels spacing between kp's
    time_t t1 = time(0) ;
    srand (time(NULL));
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
            int offsetx;
            int offsety;
            points[1].clear();
            for (int y=step; y<frame.rows-step; y+=step){
		        for (int x=step; x<frame.cols-step; x+=step){
		        	offsetx =rand()%10;
		        	offsety = rand()%10;
		            Point2f point = Point2f((float)x+offsetx, (float)y+offsety);
		            points[1].push_back(point);
		        }
		    }
            needToInit = false;
        }
        else if( !points[0].empty() )
        {
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            foreground.clear();
            if(REMOVE_UNTRACKED_POINTS){
            	temp_points[0].clear(); temp_points[1].clear();
		        for (int i = 0; i < points[0].size(); ++i)
		        {
		        	if(status[i] != 0){
		        		temp_points[0].push_back(points[0][i]);
		        		temp_points[1].push_back(points[1][i]);
		        	}
		        }
		        points[0].clear(); 
				points[1].clear();
				points[0].insert(points[0].end(), temp_points[0].begin(), temp_points[0].end());
				points[1].insert(points[1].end(), temp_points[1].begin(), temp_points[1].end());
			}
		  	Mat_ <float> H = findHomography( points[0], points[1], CV_RANSAC );
            for (int i = 0; i < points[0].size(); ++i)
            {
            	if(status[i] == 0){
            		foreground.push_back(false);
            		continue;
            	}
	            Vec3f old_point = Vec3f((float)points[0][i].x, (float)points[0][i].y, 1.0);
                Mat new_point_matrix  = H*Mat(old_point);
                Point2f new_point = Point2f(new_point_matrix.at<float>(0,0)/new_point_matrix.at<float>(2,0),new_point_matrix.at<float>(1,0)/new_point_matrix.at<float>(2,0));

                if(norm(points[1][i] - new_point)>2)
                    foreground.push_back(true);
                else
                    foreground.push_back(false);
            }

            for(int  i = 0; i < points[1].size(); i++ )
            {
                if(foreground[i])
                    circle( image, points[1][i], 3, Scalar(0,0,255), -1, 8);
                else
                    circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
            }
        }
        needToInit = false;
        if(argc>=2){
	        transpose(image, image);
	        flip(image, image , 1);
    	}
        imshow("Output", image);
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
        if(difftime(t2,t1)>1){
        	t1 = time(0);
	        needToInit = true;          
        }
    }
    return 0;
}
