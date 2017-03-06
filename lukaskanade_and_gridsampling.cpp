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

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
    if( event == CV_EVENT_LBUTTONDOWN )
    {
        point = Point2f((float)x, (float)y);
        addRemovePt = true;
    }
}

int main( int argc, char** argv )
{

    VideoCapture cap;
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 500;
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
    
    /////////////////////

    Mat img;
    cap >> img;
    SiftDescriptorExtractor sift;

    vector<KeyPoint> keypoints; // keypoint storage
    Mat descriptors; // descriptor storage

    // manual keypoint grid

    int step = 40; // 10 pixels spacing between kp's

    vector<Point2f> points[2];

    for (int y=step; y<img.rows-step; y+=step){
        for (int x=step; x<img.cols-step; x+=step){

            // x,y,radius
            // keypoints.push_back(KeyPoint(float(x), float(y), float(step)));
            Point2f point = Point2f((float)x, (float)y);
            points[1].push_back(point);
        }
    }

    // compute descriptors

    // sift.compute(img, keypoints, descriptors);

    // for(int i = 0;i < descriptors.rows;i++){
    //     for(int j= 0 ;j<descriptors.cols; j++){
    //     	float p=i;
    //     	float q=j;
    //         if(descriptors.at<double>(i,j)==0){
    //         	Point2f point = Point2f((float)i, (float)j);
    //             points[1].push_back(point);
    //         }
    //     }
    // }


/////////////////


    // vector<Point2f> points[2];
    bool addPoints = false;
    time_t t1 = time(0) ;
    int color = 0;
    srand (time(NULL));
    for(;;)
    {
        Mat frame;
        cap >> frame;
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);

        if( nightMode )
            image = Scalar::all(0);

        
        if( needToInit )
        {
            // automatic initialization
            // goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
            // cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            // addRemovePt = false;
            int offsetx =rand()%10;
            int offsety = rand()%10;
            for (int y=step+offsety; y<img.rows-step; y+=step){
		        for (int x=step+offsetx; x<img.cols-step; x+=step){
		        	offsetx =rand()%10;
		        	offsety = rand()%10;
		            // x,y,radius
		            // keypoints.push_back(KeyPoint(float(x), float(y), float(step)));
		            Point2f point = Point2f((float)x+offsetx, (float)y+offsety);
		            points[1].push_back(point);

		        }
		        // break;
		    }
		    color++;
            color= color%3;
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
            
            for( i = k = 0; i < points[1].size(); i++ )
            {
                if( addRemovePt )
                {
                    if( norm(point - points[1][i]) <= 5 )
                    {
                        addRemovePt = false;
                        continue;
                    }
                }

                if( !status[i] )
                    continue;

                points[1][k++] = points[1][i];
                
                if(color == 0)
	                circle( image, points[1][i], 3, Scalar(255,0,0), -1, 8);
	            if(color == 1)
	                circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
	            if(color == 2)
	                circle( image, points[1][i], 3, Scalar(0,0,255), -1, 8);
            }
            points[1].resize(k);
        }

        if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
        {
            vector<Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix( gray, tmp, winSize, cvSize(-1,-1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }

        needToInit = false;
        imshow("LK Demo", image);

        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        switch( c )
        {
        case 'r':
            needToInit = true;
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
        // cout<<"t1 "<<t1<<endl;
        // cout<<"t2 "<<t2<<endl;
        if(difftime(t2,t1)>1){
        	t1 = time(0);
        	// cout<<"here";
        	needToInit = true;
        }
    }

    return 0;
}
