#ifndef __KMEANSCLUSTERING_H
#define __KMEANSCLUSTERING_H

#include <iostream>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <vector>
#include "variableDefinition.h"
#include "utility.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
//using namespace std;
using namespace cv;

class kMeansClustering
{
public:
	kMeansClustering(){
		depth = NULL;
		mask = NULL;
	}
	~kMeansClustering(){

	}
	double fx, fy, cx, cy;
	int n, m;
	int seedsNum;
	float* depth;
	bool* mask;
	bool valid[IMAGE_HEIGHT][IMAGE_WIDTH];
	double normalMap[IMAGE_HEIGHT][IMAGE_WIDTH][3];
	double kSeedNormal[maxKMeansNum][3];
	int kSeedX[maxKMeansNum];
	int kSeedY[maxKMeansNum];
	int labels[IMAGE_HEIGHT][IMAGE_WIDTH];
	int nlabels[IMAGE_HEIGHT][IMAGE_WIDTH];
	int numOfEachLabels[maxKMeansNum];

	void setCameraParameters(int input_n, int input_m,
		double input_fx, double input_fy, double input_cx, double input_cy){
		n = input_n;
		m = input_m;
		fx = input_fx;
		fy = input_fy;
		cx = input_cx;
		cy = input_cy;
	}

	void setInputData(float* depthMap){
		depth = depthMap;
	}

	void setMask(bool* maskMap){
		mask = maskMap;
	}

	inline void crossProduct(double* u, double* v, double* output)
	{
		output[0] = u[1] * v[2] - u[2] * v[1];
		output[1] = u[2] * v[0] - u[0] * v[2];
		output[2] = u[0] * v[1] - u[1] * v[0];

		//normalized
		double norm = output[0] * output[0] + output[1] * output[1] + output[2] * output[2];
		norm = sqrt(norm);

		output[0] /= norm;
		output[1] /= norm;
		output[2] /= norm;
	}

	void initNormalMap()
	{
		if (depth == NULL){
			puts("error! Depth map does not exist!");
			return;
		}

		for (int i = 0; i < n; i++){
			valid[i][0] = valid[i][m - 1] = false;
		}
		for (int j = 0; j < m; j++){
			valid[0][j] = valid[n - 1][j] = false;
		}
		int nn = n - 1;
		int mm = m - 1;
		for (int i = 1; i < nn; i++)
		{
			for (int j = 1; j < mm; j++)
			{
				if (depth[i*m + j] < zeroThreshold || mask[INDEX(i, j, n, m)] == false ){
					valid[i][j] = false;
					continue;
				}
				int ty, tx;
				bool flag = true;
				for (int k = 0; k < 4; k++)
				{
					ty = dy[k] + i;
					tx = dx[k] + j;
					if (depth[ty*m + tx] < zeroThreshold ){
						flag = false;
					}
				}
				if (flag == false){
					valid[i][j] = false;
					continue;
				}
				valid[i][j] = true;
				int right = j + 1;
				int left = j - 1;
				int up = i - 1;
				int down = i + 1;
				double v_x[3], v_y[3];

				v_x[0] = (right - cx)*depth[i*m + right] / fx - (left - cx)*depth[i*m + left] / fx;
				v_x[1] = (i - cy)*depth[i*m + right] / fy - (i - cy)*depth[i*m + left] / fy;
				v_x[2] = depth[i*m + right] - depth[i*m + left];

				v_y[0] = (j - cx)*depth[down*m + j] / fx - (j - cx)*depth[up*m + j] / fx;
				v_y[1] = (down - cy)*depth[down*m + j] / fy - (up - cy)*depth[up*m + j] / fy;
				v_y[2] = depth[down*m + j] - depth[up*m + j];

				crossProduct(v_x, v_y, normalMap[i][j]);
				if (normalMap[i][j][0] * j + normalMap[i][j][1] * i + normalMap[i][j][2] * depth[i*m + j] > 0)
				{
					normalMap[i][j][0] = -normalMap[i][j][0];
					normalMap[i][j][1] = -normalMap[i][j][1];
					normalMap[i][j][2] = -normalMap[i][j][2];
				}
			}
		}
	}

	void initSeeds(int K)
	{
		double step = sqrt(double(n*m) / double(K));
		int xoff = int(step / 2);
		int yoff = int(step / 2);

		int r = 0;
		seedsNum = 0;
		for (int y = 0; y < n; y++)
		{
			int Y = int(y*step + yoff);
			if (Y >= n) break;

			for (int x = 0; x < m; x++)
			{
				//int X = x*step + xoff;//square grid
				int X = int(x*step) + (xoff << (r & 0x1));//hex grid
				if (X >= m) break;
				bool flag = false;
				double minGradient = DBL_MAX;
				int minX = -1;
				int minY = -1;
				if (valid[Y][X] == true)
				{
					bool can = true;
					for (int p = 0; p < 4; p++)
					{
						int ty = Y + dy[p];
						int tx = X + dx[p];
						if (ty < 0 || ty >= n || tx < 0 || tx >= m || valid[ty][tx] == false){
							can = false;
							break;
						}
					}
					if (can)
					{
						double dy = 0, dx = 0;
						dy += SQ(normalMap[Y - 1][X][0] - normalMap[Y + 1][X][0]);
						dy += SQ(normalMap[Y - 1][X][1] - normalMap[Y + 1][X][1]);
						dy += SQ(normalMap[Y - 1][X][2] - normalMap[Y + 1][X][2]);
						dx += SQ(normalMap[Y][X + 1][0] - normalMap[Y][X - 1][0]);
						dx += SQ(normalMap[Y][X + 1][1] - normalMap[Y][X - 1][1]);
						dx += SQ(normalMap[Y][X + 1][2] - normalMap[Y][X - 1][2]);
						if (dx + dy < minGradient){
							minGradient = dx + dy;
							minY = Y;
							minX = X;
							flag = true;
						}
					}
				}
				for (int k = 0; k < 8; k++)
				{
					int ty = Y + dy8[k];
					int tx = X + dx8[k];
					if (valid[ty][tx] == false){
						continue;
					}
					bool can = true;
					for (int p = 0; p < 4; p++)
					{
						int ty2 = ty + dy[p];
						int tx2 = tx + dx[p];
						if (ty2 < 0 || ty2 >= n || tx2 < 0 || tx2 >= m || valid[ty2][tx2] == false){
							can = false;
							break;
						}
					}
					if (can)
					{
						double dy = 0, dx = 0;
						dy += SQ(normalMap[ty - 1][tx][0] - normalMap[ty + 1][tx][0]);
						dy += SQ(normalMap[ty - 1][tx][1] - normalMap[ty + 1][tx][1]);
						dy += SQ(normalMap[ty - 1][tx][2] - normalMap[ty + 1][tx][2]);
						dx += SQ(normalMap[ty][tx + 1][0] - normalMap[ty][tx - 1][0]);
						dx += SQ(normalMap[ty][tx + 1][1] - normalMap[ty][tx - 1][1]);
						dx += SQ(normalMap[ty][tx + 1][2] - normalMap[ty][tx - 1][2]);
						if (dx + dy < minGradient)
						{
							minGradient = dx + dy;
							minY = ty;
							minX = tx;
							flag = true;
						}
					}
				}
				if (flag == true)
				{
					kSeedNormal[seedsNum][0] = normalMap[minY][minX][0];
					kSeedNormal[seedsNum][1] = normalMap[minY][minX][1];
					kSeedNormal[seedsNum][2] = normalMap[minY][minX][2];
					kSeedY[seedsNum] = minY;
					kSeedX[seedsNum] = minX;
					seedsNum++;
				}
			}
			r++;
		}
	}

	void runClustering(int K, int iterNum)
	{
		double t = (double)cvGetTickCount();

		initNormalMap();
		initSeeds(K);

		int STEP = (int)sqrt(double(n*m) / double(K));
		int offset = STEP;
		double invxywt = 1.0 / (STEP*STEP);
		vector<double>sigmaNormalX(seedsNum);
		vector<double>sigmaNormalY(seedsNum);
		vector<double>sigmaNormalZ(seedsNum);
		vector<double>sigmaX(seedsNum);
		vector<double>sigmaY(seedsNum);
		vector<int> clusterSize(seedsNum, 1);
		vector<vector <double> > distNormal(n, vector<double>(m));
		//vector<vector <double> > distXY(n, vector<double>(m));
		while (iterNum--)
		{
			for (int i = 0; i < n; i++){
				distNormal[i].assign(m, DBL_MAX);
			}
			for (int i = 0; i < seedsNum; i++)
			{
				if (clusterSize[i] == 0) continue;

				int leftTopY = max(0, kSeedY[i] - offset);
				int leftTopX = max(0, kSeedX[i] - offset);
				int rightDownY = min(n - 1, kSeedY[i] + offset);
				int rightDownX = min(m - 1, kSeedX[i] + offset);
				for (int y = leftTopY; y <= rightDownY; y++)
				{
					for (int x = leftTopX; x <= rightDownX; x++)
					{
						if (valid[y][x] == false){
							//labels[y][x] = -1;
							continue;
						}
						double dist = 0;
						dist += SQ(normalMap[y][x][0] - kSeedNormal[i][0]);
						dist += SQ(normalMap[y][x][1] - kSeedNormal[i][1]);
						dist += SQ(normalMap[y][x][2] - kSeedNormal[i][2]);

#ifdef REGULARITY
						dist += (SQ(double(y - kSeedY[i])) + SQ(double(x - kSeedX[i]))) *  invxywt;
#endif

						if (dist < distNormal[y][x]){
							distNormal[y][x] = dist;
							labels[y][x] = i;
						}
					}
				}
			}

			for (int i = 0; i < seedsNum; i++){
				sigmaNormalX[i] = 0;
				sigmaNormalY[i] = 0;
				sigmaNormalZ[i] = 0;
				sigmaX[i] = 0;
				sigmaY[i] = 0;
				clusterSize[i] = 0;
			}

			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < m; j++)
				{
					if (valid[i][j] == false || distNormal[i][j] == DBL_MAX){
						continue;
					}
					int k = labels[i][j];
					sigmaNormalX[k] += normalMap[i][j][0];
					sigmaNormalY[k] += normalMap[i][j][1];
					sigmaNormalZ[k] += normalMap[i][j][2];
					sigmaY[k] += i;
					sigmaX[k] += j;
					clusterSize[k]++;
				}
			}

			for (int i = 0; i < seedsNum; i++)
			{
				if (clusterSize[i] == 0) {
					continue;
				}
				kSeedNormal[i][0] = sigmaNormalX[i] / clusterSize[i];
				kSeedNormal[i][1] = sigmaNormalY[i] / clusterSize[i];
				kSeedNormal[i][2] = sigmaNormalZ[i] / clusterSize[i];
				kSeedY[i] = int(double(sigmaY[i]) / (double)clusterSize[i]);
				kSeedX[i] = int(double(sigmaX[i]) / (double)clusterSize[i]);
			}
		}

		EnforceLabelConnectivity(K);
	}

	void EnforceLabelConnectivity(int K)
	{
		const int sz = n*m;
		const int SUPSZ = (sz / K) >> 3;
		//nlabels.resize(sz, -1);
		memset(nlabels, -1, sizeof(nlabels));
		memset(numOfEachLabels, 0, sizeof(numOfEachLabels));
		//for (int i = 0; i < n; i++){
		//	for (int j = 0; j < m; j++){
		//		nlabels[i][j] = -1;
		//	}
		//}

		int labelNum = 1;
		vector<int>Qx(sz);
		vector<int>Qy(sz);

		int adjlabel = 0;//adjacent label
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				if (valid[i][j] == false) {
					nlabels[i][j] = 0;
					continue;
				}
				if (nlabels[i][j] > 0) {
					continue;
				}
				for (int k = 0; k < 4; k++)
				{
					int ty = i + dy[k];
					int tx = j + dx[k];
					if (ty < 0 || ty >= n || tx < 0 || tx >= m || valid[ty][tx] == false){
						continue;
					}
					if (nlabels[ty][tx] > 0){
						adjlabel = nlabels[ty][tx];
						break ;
					}
				}
				nlabels[i][j] = labelNum;
				numOfEachLabels[labelNum] = 1;
				kSeedNormal[labelNum][0] = normalMap[i][j][0] ;
				kSeedNormal[labelNum][1] = normalMap[i][j][1];
				kSeedNormal[labelNum][2] = normalMap[i][j][2];

				int head = 0, tail = 1;
				Qy[0] = i;
				Qx[0] = j;
				while (head < tail)
				{
					int y = Qy[head];
					int x = Qx[head];
					head++;
					for (int k = 0; k < 4; k++)
					{
						int ty = y + dy[k];
						int tx = x + dx[k];
						if (ty < 0 || ty >= n || tx < 0 || tx >= m || valid[ty][tx] == false){
							continue;
						}
						if (nlabels[ty][tx] < 0 && labels[ty][tx] == labels[y][x]){
							nlabels[ty][tx] = labelNum;
							numOfEachLabels[labelNum]++;
							kSeedNormal[labelNum][0] += normalMap[ty][tx][0];
							kSeedNormal[labelNum][1] += normalMap[ty][tx][1];
							kSeedNormal[labelNum][2] += normalMap[ty][tx][2];

							Qx[tail] = tx;
							Qy[tail] = ty;
							tail++;
						}
					}
				}

				if (tail > SUPSZ){
					labelNum++;
				}
				else
				{
					head = 0;
					while (head < tail)
					{
						int y = Qy[head];
						int x = Qx[head];
						head++;

						nlabels[y][x] = adjlabel;
						numOfEachLabels[adjlabel]++;
						kSeedNormal[adjlabel][0] += normalMap[y][x][0];
						kSeedNormal[adjlabel][1] += normalMap[y][x][1];
						kSeedNormal[adjlabel][2] += normalMap[y][x][2];
					}
				}
			}
		}

		seedsNum = labelNum-1 ;
		memcpy(labels, nlabels, sizeof(labels));
		for (int i = 1; i <= seedsNum; i++)
		{
			kSeedNormal[i][0] /= numOfEachLabels[i];
			kSeedNormal[i][1] /= numOfEachLabels[i];
			kSeedNormal[i][2] /= numOfEachLabels[i];
		}

	}

};

#endif