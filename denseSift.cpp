#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
using namespace cv;

int main(int argc, char** argv)
{

    Mat img = imread("elephant.jpg", 0);
    SiftDescriptorExtractor sift;

	vector<KeyPoint> keypoints; // keypoint storage
	Mat descriptors; // descriptor storage

	// manual keypoint grid

	int step = 50; // 10 pixels spacing between kp's

	for (int y=step; y<img.rows-step; y+=step){
	    for (int x=step; x<img.cols-step; x+=step){

	        // x,y,radius
	        keypoints.push_back(KeyPoint(float(x), float(y), float(step)));
	    }
	}

	// compute descriptors

	sift.compute(img, keypoints, descriptors);
    Mat output_img;
    // drawKeypoints(img, descriptors, output_img);

    namedWindow("Image");
    imshow("Image", descriptors);
    waitKey(0);
    destroyWindow("Image");

    return 0;
}