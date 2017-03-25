//Obtained from lkdemo.cpp in opencv samples
#include <iostream>
#include <ctype.h>
#include <time.h>
#include <set>
#include <algorithm>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/highgui/highgui.hpp"
#define PI 3.14159265

using namespace cv;
using namespace std;

class motion_vector{
public:
	Point2f point;
	float angle;	//angle in degrees
	motion_vector(Point2f point, float angle){
		this->point = point; this->angle = acos(angle)*180/PI;
	};
};

struct points_compare {
    bool operator() (const Point2f& lhs, const Point2f& rhs) const{
        if(lhs.x < rhs.x) return true;
        if(rhs.x > lhs.x) return false;
        if(lhs.y < rhs.y) return true;
        if(rhs.y > lhs.y) return false;
        return false;
    }
};

int colors[10][3];
bool REMOVE_UNTRACKED_POINTS = true;
int PIXEL_WINDOW_FOR_MOTION_CLASS_ESTIMATION = 30;
float ANGULAR_THRESHOLD_FOR_MOTION_CLASS_ESTIMATION = 20.0f;
float THRESHOLD_FOR_BACKGROUND_FOREGROUND_DISTINCTION = 3.0f;
int MINIMUM_NUMBER_OF_POINTS_IN_CLUSTER = 100;
vector<Point2f> temp_points[2];
vector<Point2f> points[2];
vector <bool> foreground;
vector<Point2f> new_points;
vector<motion_vector> foreground_motion_vectors;
vector<Point2f> foreground_motion_vectors_points;
vector< set<Point2f, points_compare> > cluster_foreground_vectors;

bool compare_motion_vectors(const motion_vector m1, const motion_vector m2){
	return m1.angle < m2.angle;
}


void merge_points(Point2f p1, Point2f p2){
	int s1 = -1;
	for(int i = 0; i<cluster_foreground_vectors.size();i++)
		if(cluster_foreground_vectors[i].find(p1) != cluster_foreground_vectors[i].end()){
			s1 = i; break;
		}
	int s2 = -1;
	for(int i = 0; i<cluster_foreground_vectors.size();i++)
		if(cluster_foreground_vectors[i].find(p2) != cluster_foreground_vectors[i].end()){
			s2 = i; break;
		}
	if(s1 == s2){
		if(s1== -1) {
			// create new cluster and add both
			set<Point2f, points_compare> s;
			s.insert(p1);s.insert(p2);
			cluster_foreground_vectors.push_back(s);
			return;
		}
		else //already in same cluster
			return;
	}
	if(s1 != -1 && s2 == -1){
		cluster_foreground_vectors[s1].insert(p2);
	}
	else if(s1 == -1 && s2 != -1){
		cluster_foreground_vectors[s2].insert(p1);
	}
	else{
		set <Point2f, points_compare> set1 (cluster_foreground_vectors[s1]);
		set <Point2f, points_compare> set2 (cluster_foreground_vectors[s2]);
		cluster_foreground_vectors.erase(cluster_foreground_vectors.begin() + s1);
		cluster_foreground_vectors.erase(cluster_foreground_vectors.begin() + s2);
		set <Point2f, points_compare> set_final;
		for(set<Point2f, points_compare>::iterator i = set1.begin(); i!= set1.end();i++)
			set_final.insert(*i);
		for(set<Point2f, points_compare>::iterator i = set2.begin(); i!= set2.end();i++)
			set_final.insert(*i);
		cluster_foreground_vectors.push_back(set_final);
	}

}

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
    time_t t1 = time(0), t2 ;
    srand (time(NULL));
    Mat frame; 
	Mat_ <float> H;
    Size reduced_size(640,480);
    int offsetx;
    int offsety;
    int SAMPLE_EVERY_N_FRAMES = 3;
    int counter = 0;
    for(;;)
    {
    	// counter++;
    	// if(counter%SAMPLE_EVERY_N_FRAMES != 0)
    	// 	continue;
        cap >> frame;
        cv::resize(frame, frame, reduced_size);
        if( frame.empty() )
            break;
        frame.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);
        if( nightMode )
            image = Scalar::all(0);
        if( needToInit )
        {
            points[1].clear();
            for (int y=step; y<frame.rows-step; y+=step){
		        for (int x=step; x<frame.cols-step; x+=step){
		        	offsetx =rand()%10;
		        	offsety = rand()%10;
		            points[1].push_back(Point2f((float)x+offsetx, (float)y+offsety));
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
			try{
		  		H = findHomography( points[0], points[1], CV_RANSAC );
			}
		  	catch(Exception e){
		  		cout<<"Error! Insufficient number of points for findHomography. Skipping frame.\n";
		  		continue;
		  	}
		  	// foreground_points.clear();
		  	// foreground_motion_vectors.clear();
            for (int i = 0; i < points[0].size(); ++i)
            {
            	if(status[i] == 0){
            		// foreground.push_back(false);
            		continue;
            	}
	            Vec3f old_point = Vec3f((float)points[0][i].x, (float)points[0][i].y, 1.0);
                Mat new_point_matrix  = H*Mat(old_point);
                Point2f new_point = Point2f(new_point_matrix.at<float>(0,0)/new_point_matrix.at<float>(2,0),new_point_matrix.at<float>(1,0)/new_point_matrix.at<float>(2,0));
                // if(norm(points[1][i] - new_point)>2)
                //     foreground.push_back(true);
                // else
                //     foreground.push_back(false);
                if(norm(points[1][i] - new_point)>THRESHOLD_FOR_BACKGROUND_FOREGROUND_DISTINCTION){
                	//foreground point
                	float cos_angle = (points[1][i].x - points[0][i].x)/(float)norm(points[1][i] - points[0][i]);
                	motion_vector mVector(points[1][i], cos_angle);
                	foreground_motion_vectors_points.push_back(points[1][i]);//////////////////////////////////
                	foreground_motion_vectors.push_back(mVector);
                	cout<<"added";
                }
            }
            // sort(foreground_motion_vectors.begin(), foreground_motion_vectors.end(), compare_motion_vectors);
            cluster_foreground_vectors.clear();
           /* for (int i = 0; i < foreground_motion_vectors.size() ; ++i)
            {
            	for(int j = i+1; j<foreground_motion_vectors.size();j++){
            		if(norm(foreground_motion_vectors[i].point - foreground_motion_vectors[j].point) < PIXEL_WINDOW_FOR_MOTION_CLASS_ESTIMATION && abs(foreground_motion_vectors[i].angle - foreground_motion_vectors[j].angle)<ANGULAR_THRESHOLD_FOR_MOTION_CLASS_ESTIMATION){
            			merge_points(foreground_motion_vectors[i].point, foreground_motion_vectors[j].point);
            		}
            	}
            }
            cout<<"size";
			cout<<foreground_motion_vectors_points.size();*/
			//kmeans/////////////////////////////
			cout<<"size...///...";
			cout<<foreground_motion_vectors.size();
			if(foreground_motion_vectors_points.size()>50)
				{
					Mat sample = Mat(foreground_motion_vectors_points); 
					int clusterCount = 2;
					  Mat labels;
					  int attempts = 5;
					  Mat centers;
					  kmeans(sample, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 
					  	10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );

					for(int i = 0; i<foreground_motion_vectors_points.size();i++)
					{
							int l=labels.at<int>(i,0);
	            			circle( image, foreground_motion_vectors_points[i], 3, Scalar(colors[l][0],colors[l][1],colors[l][2]), -1, 8);
	            	
        			}


				}
				foreground_motion_vectors_points.clear();
			////////////////////////////////////

    		int r1,r2,r3;
    		bool flag_atleast_one_cluster = false;  
    		int cluster_counter = 0;  
            for(int i = 0; i<cluster_foreground_vectors.size();i++){
        		if(cluster_foreground_vectors[i].size()>MINIMUM_NUMBER_OF_POINTS_IN_CLUSTER){
        			flag_atleast_one_cluster = true;
        			r1 = colors[cluster_counter][0];r2 = colors[cluster_counter][1]; r3 = colors[cluster_counter][2];
        			cout<<"Cluster size"<<cluster_foreground_vectors[i].size()<<endl;
	            	for(set<Point2f, points_compare> :: iterator j = cluster_foreground_vectors[i].begin(); j!= cluster_foreground_vectors[i].end();j++){
	            		//circle( image, *j, 3, Scalar(r1,r2,r3), -1, 8);
        			}
        			cluster_counter++;
            	}
            }
            if(flag_atleast_one_cluster)
            	cout<<"-----------\n";
             //for(int  i = 0; i < points[1].size(); i++ )
             //{
             //  if(foreground[i])
             //        circle( image, points[1][i], 3, Scalar(0,0,255), -1, 8);
             //    else
             //        circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
            // }
        }

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
        t2 = time(0) ;
        if(difftime(t2,t1)>1){
    		srand(time(NULL));	  int r1,r2,r3;  
        	t1 = time(0);
	        needToInit = true;
	        for(int i=0;i<10;i++){
	    		r1 = rand()%255; r2 = rand()%255; r3 = rand()%255;
	    		colors[i][0] = r1; colors[i][1] = r2; colors[i][2] = r3;
	    	}
	        foreground_motion_vectors.clear();     
        }
    }
    return 0;
}
