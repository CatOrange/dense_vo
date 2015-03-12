#ifndef __TESTDATAGENERATIOIN_H
#define __TESTDATAGENERATIOIN_H

#include "dataStructure.h"
#include "variableDefinition.h"
#include "utility.h"
#include <cmath>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <list>
#include <omp.h>
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
using namespace std;
using namespace cv;
using namespace Eigen;
const int testDataNum = 1000;

struct PLANE
{
	Vector3d normal;
	vector<Vector3d> pList;
	vector<uchar>Intensity;
	PLANE()
	{
		pList.clear();
		Intensity.clear();
	}
	~PLANE()
	{

	}
};

class testDataGenerator
{
public:
	int numOfPlane, numOfState;
	int imgHeight, imgWidth;
	double cx, cy, fx, fy;

	char filePath[256];
	char rgbListPath[256];
	char rgbFileNameList[testDataNum][128];

	vector<PLANE> planeList;
	vector<Matrix3d> stateRList;
	vector<Vector3d> stateTList;

	testDataGenerator()
	{
#ifdef DOWNSAMPLING
		imgHeight = 240;
		imgWidth = 320;
#else
		imgHeight = 480;
		imgWidth = 640;
#endif
		fx = fy = 525.0;
		cx = 319.5;
		cy = 239.5;
		strcpy(filePath, "D:\\Dataset\\rgbd_dataset_freiburg1_xyz\\");

		numOfPlane = 2;
		planeList.clear();
		planeList.resize(numOfPlane);

		numOfState = 5;
		stateRList.clear();
		stateTList.clear();
		stateRList.resize(numOfState);
		stateTList.resize(numOfState);
	}

	~testDataGenerator()
	{
	}

	void generate()
	{
		char tmp[256];
		RNG rng(161);

		strcpy(rgbListPath, filePath);
		strcat(rgbListPath, "rgb.txt");

		//read rgb image name list
		FILE* fp = NULL ;
		fp = fopen(rgbListPath, "r");
		if (fp == NULL){
			puts("rgbList Path error");
		}
		while (fgets(tmp, 256, fp) != NULL){
			if (tmp[0] != '#') break;
		}
		for (int i = 0, j; i < numOfPlane; i++)
		{
			if (fgets(tmp, 256, fp) == NULL) break;

			char tt[128];
			int n, ns;
			sscanf(tmp, "%d.%d %s", &n, &ns, tt);

			for (j = 0; tmp[j] != '\0'; j++){
				if (tmp[j] == '/') break;
			}
			strcpy(rgbListPath, filePath);
			strcat(rgbListPath, "rgb\\");
			strcat(rgbListPath, &tmp[j + 1]);
			strcpy(rgbFileNameList[i], rgbListPath);
			rgbFileNameList[i][strlen(rgbFileNameList[i]) - 1] = '\0';

			//sprintf(rgbFileNameList[i], "%d.png", i + 1);
		}
		fclose(fp);

		double centerZ = 3.0 ;
		double centerX = -(imgWidth) / fx *centerZ / 2.0;
		double centerY = -(imgHeight) / fx *centerZ / 2.0;
		for (int i = 0; i < numOfPlane; i++)
		{
			//cout << rgbFileNameList[i] << endl;
			Mat rgbImage = imread(rgbFileNameList[i], CV_LOAD_IMAGE_COLOR);
			Mat Image;
			cvtColor(rgbImage, Image, CV_BGR2GRAY);

			GaussianBlur(Image, Image, Size(131, 131), 0);

			double roll = rng.uniform(-PI / 10, PI / 10);
			double pitch = rng.uniform(-PI / 10, PI / 10);
			double yaw = rng.uniform(-PI / 10, PI / 10);

			centerX += (imgWidth ) / fx *centerZ / 4.0;
			centerY +=  (imgHeight) / fy *centerZ / 4.0;
			centerZ += 0.1;

			Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());
			Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
			Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());
			Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
			Eigen::Matrix3d R = q.matrix();
			
			//generate the world
			Vector3d n1, n2;
			n1 << 0, 0, -1;
			n2 = R*n1;
			planeList[i].normal = n2;
			for (int  u = 0; u < imgHeight; u++)
			{
				for (int v = 0; v < imgWidth; v++)
				{
					Vector3d p;
					p << centerX + (v - cx) / fx *centerZ / 4.0 , centerY + (u-cy)/fy *centerZ / 4.0, centerZ;
					Vector3d p2 = R*p;
					planeList[i].pList.push_back(p2);
					uchar tmp = Image.at<uchar>(u, v);
					if (tmp == 0){
						tmp = 1;
					}
					planeList[i].Intensity.push_back(tmp);
				}
			}
		}
		
		//generate the state
		double roll = 0;
		double pitch = 0;
		double yaw = 0;
		Vector3d p = Vector3d::Zero();
		
		for (int i = 0; i < numOfState; i++)
		{
			Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());
			Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
			Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());
			Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;

			stateRList[i] = q.matrix();
			stateTList[i] = p;

			roll += rng.uniform(0.0, PI / 30);
			pitch += rng.uniform(0.0, PI / 30);
			yaw += rng.uniform(0.0, PI / 30);
			p(0) += rng.uniform(0.0, 0.08);
			p(1) += rng.uniform(0.0, 0.08);
			p(2) += rng.uniform(0.0, 0.08);
		}
	}

	void testProgram(STATEESTIMATION*slidingWindows )
	{
		//generate the images
		bool vst = false;
		for (int i = 0; i < numOfState; i++)
		{
			Mat currentImage(imgHeight, imgWidth, CV_8UC1);
			Mat depthImage(imgHeight, imgWidth, CV_32F);
			Mat grayImage[maxPyramidLevel];

			currentImage.setTo(0);
			depthImage.setTo(0.0);
			double maxDepth = 0;

			for (int k = 0; k < numOfPlane; k++)
			{
				int sz = planeList[k].pList.size();
				Vector3d normal = stateRList[i] * planeList[k].normal;
				double depth = 0;
				for (int j = 0; j < sz; j++)
				{
					Vector3d p2 = stateRList[i] * (planeList[k].pList[j] - stateTList[i]);
					depth -= p2.transpose()*normal;
				}
				depth /= sz;

				printf("%d:groundtruth Normal\n", i ) ;
				cout << normal << endl;
				cout << depth << endl;

				for (int j = 0; j < sz; j++)
				{
					Vector3d p2 = stateRList[i] * (planeList[k].pList[j] - stateTList[i]);
					int u = p2(1) / p2(2) * fy + cy + 0.5 ;
					int v = p2(0) / p2(2) * fx + cx +0.5 ;
					if (u < 0 || u >= imgHeight || v < 0 || v >= imgWidth){
						continue;
					}
					double tmp = p2.transpose()*normal;
					if (fabs(tmp + depth) > zeroThreshold){
						cout << tmp + depth << endl;
					}
					currentImage.at<uchar>(u, v) = planeList[k].Intensity[j];
					depthImage.at<float>(u, v) = p2(2);

					if (p2(2) > maxDepth){
						maxDepth = p2(2);
					}
				}
			}
			imshow("currentImage", currentImage);
			imshow("depthImage", depthImage/maxDepth );
			waitKey(0);

			grayImage[0] = currentImage.clone();
			for (int kk = 1; kk < maxPyramidLevel; kk++){
				pyrDown(grayImage[kk - 1], grayImage[kk]);//down-sampling
			}

			if (vst == false)//the first frame
			{
				vst = true;
				slidingWindows->insertKeyFrame(grayImage, depthImage, Matrix3d::Identity(), Vector3d::Zero());
				continue;
			}

			slidingWindows->insertKeyFrame(grayImage, depthImage, stateRList[i].transpose(), stateTList[i] );
			//slidingWindows->PhotometricStructureOnlyBA();
			slidingWindows->PhotometricBA();
			
			waitKey(0);
		}
	}
};


#endif