#ifndef __AHCLUSTERING_H
#define __AHCLUSTERING_H

#include <iostream>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <vector>
#include "variableDefinition.h"
#include "dataStructure.h"
#include "utility.h"
#include "dsyevh3.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
using namespace std;
using namespace cv;

class AHClustering
{
public:
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
	int numOfEachLabels[maxKMeansNum];
	NODE nodeMap[(IMAGE_HEIGHT / winH + 5)*(IMAGE_WIDTH / winW + 5)];
	int nodeMapID[IMAGE_HEIGHT / winH + 5][IMAGE_WIDTH / winW + 5];
	int totalNumNodes;
	minHeap myHeap;
	unionSet mySet;
	std::list<int>nodeMapNeighbourList[IMAGE_HEIGHT*IMAGE_WIDTH + 5];

	AHClustering()
	{
		depth = NULL;
		mask = NULL;
	}
	~AHClustering()
	{

	}

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

	inline bool rejectNodeByDepthDiscontinuity(const double& a, const double& b){
		return (fabs(a - b)) > (2 * alpha * (a + 0.0005));
	}

	bool calculateNormal(NODE& tmp)
	{
		//3. MSE reject
		double Cp[3][3], V[3][3], w[3];

		//previous
		//Cp[0][0] = tmp.sumX2 / (tmp.num) - tmp.sumX*tmp.sumX / (tmp.num*tmp.num);
		//Cp[0][1] = tmp.sumXY / (tmp.num) - tmp.sumX*tmp.sumY / (tmp.num*tmp.num);
		//Cp[0][2] = tmp.sumXZ / (tmp.num) - tmp.sumX*tmp.sumZ / (tmp.num*tmp.num);
		//Cp[1][0] = tmp.sumXY / (tmp.num) - tmp.sumY*tmp.sumX / (tmp.num*tmp.num);
		//Cp[1][1] = tmp.sumY2 / (tmp.num) - tmp.sumY*tmp.sumY / (tmp.num*tmp.num);
		//Cp[1][2] = tmp.sumYZ / (tmp.num) - tmp.sumY*tmp.sumZ / (tmp.num*tmp.num);
		//Cp[2][0] = tmp.sumXZ / (tmp.num) - tmp.sumZ*tmp.sumX / (tmp.num*tmp.num);
		//Cp[2][1] = tmp.sumYZ / (tmp.num) - tmp.sumZ*tmp.sumY / (tmp.num*tmp.num);
		//Cp[2][2] = tmp.sumZ2 / (tmp.num) - tmp.sumZ*tmp.sumZ / (tmp.num*tmp.num);

		//modified
#pragma omp parallel
		{
			//Cp[0][0] = tmp.sumX2 - 2.0*tmp.sumX*tmp.sumX / tmp.num + tmp.sumX*tmp.sumX / tmp.num;
			//Cp[0][1] = tmp.sumXY - 2.0*tmp.sumX*tmp.sumY / tmp.num + tmp.sumX*tmp.sumY / tmp.num;
			//Cp[0][2] = tmp.sumXZ - 2.0*tmp.sumX*tmp.sumZ / tmp.num + tmp.sumX*tmp.sumZ / tmp.num;
			//Cp[1][0] = tmp.sumXY - 2.0*tmp.sumY*tmp.sumX / tmp.num + tmp.sumY*tmp.sumX / tmp.num;
			//Cp[1][1] = tmp.sumY2 - 2.0*tmp.sumY*tmp.sumY / tmp.num + tmp.sumY*tmp.sumY / tmp.num;
			//Cp[1][2] = tmp.sumYZ - 2.0*tmp.sumY*tmp.sumZ / tmp.num + tmp.sumY*tmp.sumZ / tmp.num;
			//Cp[2][0] = tmp.sumXZ - 2.0*tmp.sumZ*tmp.sumX / tmp.num + tmp.sumZ*tmp.sumX / tmp.num;
			//Cp[2][1] = tmp.sumYZ - 2.0*tmp.sumZ*tmp.sumY / tmp.num + tmp.sumZ*tmp.sumY / tmp.num;
			//Cp[2][2] = tmp.sumZ2 - 2.0*tmp.sumZ*tmp.sumZ / tmp.num + tmp.sumZ*tmp.sumZ / tmp.num;

			Cp[0][0] = tmp.sumX2 - tmp.sumX*tmp.sumX / tmp.num ;
			Cp[0][1] = tmp.sumXY - tmp.sumX*tmp.sumY / tmp.num ;
			Cp[0][2] = tmp.sumXZ - tmp.sumX*tmp.sumZ / tmp.num ;
			Cp[1][0] = tmp.sumXY - tmp.sumY*tmp.sumX / tmp.num ;
			Cp[1][1] = tmp.sumY2 - tmp.sumY*tmp.sumY / tmp.num ;
			Cp[1][2] = tmp.sumYZ - tmp.sumY*tmp.sumZ / tmp.num ;
			Cp[2][0] = tmp.sumXZ - tmp.sumZ*tmp.sumX / tmp.num ;
			Cp[2][1] = tmp.sumYZ - tmp.sumZ*tmp.sumY / tmp.num ;
			Cp[2][2] = tmp.sumZ2 - tmp.sumZ*tmp.sumZ / tmp.num ;
		}
		int result = dsyevh3(Cp, V, w);
		if (result < 0){
			return false;
		}
		else
		{
			int minIndex = 0;

#pragma omp parallel for
			for (int i = 0; i < 3; i++)
			{
				if (w[i] < 0){
					V[0][i] = -V[0][i];
					V[1][i] = -V[1][i];
					V[2][i] = -V[2][i];
					w[i] = -w[i];
				}
				w[i] *= 1000;
			}

			if (w[1] < w[minIndex])  {
				minIndex = 1;
			}
			if (w[2] < w[minIndex]) {
				minIndex = 2;
			}
			tmp.MSE = w[minIndex] * w[minIndex] / tmp.num;
			tmp.normal[0] = V[0][minIndex];
			tmp.normal[1] = V[1][minIndex];
			tmp.normal[2] = V[2][minIndex];
			return true;
		}
	}

	void initGraph(int height, int width, double fx, double fy, double cx, double cy)
	{
		//node graph initialization
		myHeap.clear();

		int numCount = 0;
		//int numInCol = (width + winW - 1) / winW;
		//int neighborNumDifferences[4];
		//neighborNumDifferences[0] = -numInCol;//up
		//neighborNumDifferences[1] = numInCol;//down
		//neighborNumDifferences[0] = -1;//left
		//neighborNumDifferences[0] = 1;//right

		for (int i = 0; i < height; i += winH)
		{
			for (int j = 0; j < width; j += winW)
			{
				int uBegin = i;
				int uEnd = min(i + winH, height);
				int vBegin = j;
				int vEnd = min(j + winW, width);
				bool flag = true;
				int ii = i / winH;
				int jj = j / winW;
				nodeMapID[ii][jj] = numCount;
				NODE& tmp = nodeMap[numCount];

				tmp.sumX = tmp.sumX2 = tmp.sumXY = tmp.sumXZ = 0;
				tmp.sumY = tmp.sumY2 = tmp.sumYZ = 0;
				tmp.sumZ = tmp.sumZ2 = 0;
				tmp.num = (uEnd - uBegin)* (vEnd - vBegin);
				tmp.id = numCount;
				nodeMapNeighbourList[numCount].clear();
				for (int u = uBegin; u < uEnd; u++)
				{
					for (int v = vBegin; v < vEnd; v++)
					{
						//1. missing data
						float  tmpD = depth[u*width + v];
						if (tmpD < zeroThreshold || mask[INDEX(u, v, height, width)] == false ) {
							flag = false;
							break;
						}

						//2. depth discontinuity
						for (int k = 0; k < 4; k++)
						{
							int tu = u + dy[k];
							int tv = v + dx[k];
							if (tu < 0 || tu >= height || tv < 0 || tv >= width){
								continue;
							}
							//bool output = rejectNodeByDepthDiscontinuity(double(tmpD), double(depth[tu*width + tv]));
							//flag &= !output;
						}
						if (flag == false) {
							break;
						}

						double Z = tmpD / depthFactor;
						double X = (v - cx) * Z / fx;
						double Y = (u - cy) * Z / fy;

						tmp.sumX += X;
						tmp.sumY += Y;
						tmp.sumZ += Z;
						tmp.sumX2 += X*X;
						tmp.sumY2 += Y*Y;
						tmp.sumZ2 += Z*Z;
						tmp.sumXY += X*Y;
						tmp.sumXZ += X*Z;
						tmp.sumYZ += Y*Z;
					}
					if (flag == false) {
						break;
					}
				}
				numCount++;

				tmp.valid = flag;
				if (flag == false){
					continue;
				}

				//3. MSE reject
				bool result = calculateNormal(tmp);
				if (result == false){
					tmp.valid = false;
				}
				else
				{
					double averageDepth = tmp.sumZ / tmp.num;
					double T_MSE = SQ(averageDepth*averageDepth*sigma + eps);
					if (tmp.MSE > T_MSE){
						tmp.valid = false;
					}
					else{
						HEAPNODE tt;
						tt.MSE = tmp.MSE;
						tt.id = tmp.id;
						myHeap.push(tt);
					}
				}

				//4. check normal
				double x = tmp.sumX / tmp.num;
				double y = tmp.sumY / tmp.num;
				double z = tmp.sumZ / tmp.num;
				if (tmp.normal[0] * x + tmp.normal[1] * y + tmp.normal[2] * z >0){
					//puts("found!!!");
					tmp.normal[0] = -tmp.normal[0];
					tmp.normal[1] = -tmp.normal[1];
					tmp.normal[2] = -tmp.normal[2];
				}
			}
		}
		totalNumNodes = numCount;
		mySet.unionSetInit(totalNumNodes + 5);

		//initial edge
		int n = (height + winH - 1) / winH;
		int m = (width + winW - 1) / winW;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				int id = nodeMapID[i][j];
				if (nodeMap[id].valid == false){
					continue;
				}
				for (int k = 0; k < 4; k++)
				{
					int ty = i + dy[k];
					int tx = j + dx[k];
					if (ty < 0 || ty >= n || tx < 0 || tx >= m){
						continue;
					}
					int kID = nodeMapID[ty][tx];
					if (nodeMap[kID].valid == false){
						continue;
					}
					/*				
					double T_ANG = 0.2618 + 0.3740*(nodeMap[id].sumZ/nodeMap[id].num - 0.5) ;
					double tmp_ANG = nodeMap[id].normal[0] * nodeMap[kID].normal[0] + nodeMap[id].normal[1] * nodeMap[kID].normal[1] +
					nodeMap[id].normal[2] * nodeMap[kID].normal[2];
					double tt = acos(tmp_ANG);
					if (tmp_ANG > PI / 2){

					}
					if ( tmp_ANG > cos(T_ANG) )
					*/	
					{
						nodeMapNeighbourList[id].push_back(kID);
					}
				}
			}
		}
	}

	void postprocessing( int height, int width, int K )
	{
		int n = (height + winH - 1) / winH;
		int m = (width + winW - 1) / winW;
		int cnt = 0;
		vector<HEAPNODE> nodeList(n*m);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				int id = nodeMapID[i][j];
				if (nodeMap[id].valid == false){
					continue;
				}
				if (mySet.find(id) != id){
					continue;
				}
				if (nodeMap[id].num > minimumPixelsInASuperpixel )
				{
					nodeList[cnt].id = id;
					nodeList[cnt].MSE = nodeMap[id].MSE;
					cnt++;
				}
			}
		}
		sort(nodeList.begin(), nodeList.begin() + cnt);

		seedsNum = min(K, cnt);
		vector<int>idList(seedsNum);
		for (int i = 0; i < seedsNum; i++)
		{
			int currentID = idList[i] = nodeList[i].id;

			//check normal
			double x = nodeMap[currentID].sumX / nodeMap[currentID].num;
			double y = nodeMap[currentID].sumY / nodeMap[currentID].num;
			double z = nodeMap[currentID].sumZ / nodeMap[currentID].num;
			if (nodeMap[currentID].normal[0] * x + nodeMap[currentID].normal[1] * y + nodeMap[currentID].normal[2] * z >0){
				//puts("found!!!");
				nodeMap[currentID].normal[0] = -nodeMap[currentID].normal[0];
				nodeMap[currentID].normal[1] = -nodeMap[currentID].normal[1];
				nodeMap[currentID].normal[2] = -nodeMap[currentID].normal[2];
			}

			numOfEachLabels[i+1] = 0;
			kSeedNormal[i+1][0] = nodeMap[currentID].normal[0];
			kSeedNormal[i+1][1] = nodeMap[currentID].normal[1];
			kSeedNormal[i+1][2] = nodeMap[currentID].normal[2];
		}

		sort(idList.begin(), idList.end());
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				int id = nodeMapID[i][j];
				int uBegin = i*winH;
				int vBegin = j*winW;
				int uEnd = min(uBegin + winH, height);
				int vEnd = min(vBegin + winW, width);
				if (nodeMap[id].valid == false )
				{
					for (int y = uBegin; y < uEnd; y++){
						for (int x = vBegin; x < vEnd; x++){
							labels[y][x] = 0;
						}
					}
					continue;
				}
				id = mySet.find(id);
				int idx = lower_bound(idList.begin(), idList.end(), id) - idList.begin();
				if ( idx == seedsNum || idList[idx] != id )
				{
					for (int y = uBegin; y < uEnd; y++){
						for (int x = vBegin; x < vEnd; x++){
							labels[y][x] = 0;
						}
					}
					continue;
				}
				else 
				{
					numOfEachLabels[idx + 1] += (uEnd - uBegin)*(vEnd - vBegin);
					for (int y = uBegin; y < uEnd; y++){
						for (int x = vBegin; x < vEnd; x++){
							labels[y][x] = idx+1;
						}
					}
				}
			}
		}
	}

	void runClustering( int K )
	{
		//init
		initGraph(n, m, fx, fy, cx, cy);

		standardClustering();
		//simpleClustering();
		postprocessing(n, m, K);
	}

	void standardClustering()
	{
		seedsNum = 0;
		while (myHeap.size > 0)
		{
			HEAPNODE current = myHeap.data[1];
			myHeap.pop();
			int currentParent = mySet.find(current.id);
			if (current.id != currentParent){
				continue;
			}
			if (nodeMap[currentParent].num > T_NUM){
				continue;
			}
			double averageDepth = nodeMap[currentParent].sumZ / nodeMap[currentParent].num;
			std::list<int>::iterator iter;
			double T_MSE = SQ(sigma*averageDepth*averageDepth + eps);
			double minMSE = DBL_MAX;
			int minIndex = -1;
			for (iter = nodeMapNeighbourList[currentParent].begin(); iter != nodeMapNeighbourList[currentParent].end(); iter++)
			{
				int k = *iter;
				int kParent = mySet.find(k);

				if ( nodeMap[kParent].num > T_NUM) {
					continue;
				}

				NODE tmp;

				tmp.sumX = nodeMap[currentParent].sumX + nodeMap[kParent].sumX;
				tmp.sumY = nodeMap[currentParent].sumY + nodeMap[kParent].sumY;
				tmp.sumZ = nodeMap[currentParent].sumZ + nodeMap[kParent].sumZ;
				tmp.sumX2 = nodeMap[currentParent].sumX2 + nodeMap[kParent].sumX2;
				tmp.sumY2 = nodeMap[currentParent].sumY2 + nodeMap[kParent].sumY2;
				tmp.sumZ2 = nodeMap[currentParent].sumZ2 + nodeMap[kParent].sumZ2;
				tmp.sumXY = nodeMap[currentParent].sumXY + nodeMap[kParent].sumXY;
				tmp.sumXZ = nodeMap[currentParent].sumXZ + nodeMap[kParent].sumXZ;
				tmp.sumYZ = nodeMap[currentParent].sumYZ + nodeMap[kParent].sumYZ;
				tmp.num = nodeMap[currentParent].num + nodeMap[kParent].num;

				if (calculateNormal(tmp)) {
					if (tmp.MSE < minMSE){
						minIndex = kParent;
						minMSE = tmp.MSE;
					}
				}
			}
			if (minIndex < 0){
				continue;
			}
			//printf("DEBUG: minMSE: %f     T_MSE: %f	num:%d\n", minMSE, T_MSE );
			if (minMSE > T_MSE)//Merge fail
			{
				;
			}
			else//Merge succeed
			{
				//1. update current node 
				NODE& tmp = nodeMap[currentParent];

				tmp.sumX = nodeMap[currentParent].sumX + nodeMap[minIndex].sumX;
				tmp.sumY = nodeMap[currentParent].sumY + nodeMap[minIndex].sumY;
				tmp.sumZ = nodeMap[currentParent].sumZ + nodeMap[minIndex].sumZ;
				tmp.sumX2 = nodeMap[currentParent].sumX2 + nodeMap[minIndex].sumX2;
				tmp.sumY2 = nodeMap[currentParent].sumY2 + nodeMap[minIndex].sumY2;
				tmp.sumZ2 = nodeMap[currentParent].sumZ2 + nodeMap[minIndex].sumZ2;
				tmp.sumXY = nodeMap[currentParent].sumXY + nodeMap[minIndex].sumXY;
				tmp.sumXZ = nodeMap[currentParent].sumXZ + nodeMap[minIndex].sumXZ;
				tmp.sumYZ = nodeMap[currentParent].sumYZ + nodeMap[minIndex].sumYZ;
				tmp.num = nodeMap[currentParent].num + nodeMap[minIndex].num;
				calculateNormal(tmp);

				//2. update neighbour list
				std::list<int> tmpList;
				tmpList.clear();

				std::list<int>::iterator iter;
				for (iter = nodeMapNeighbourList[currentParent].begin(); iter != nodeMapNeighbourList[currentParent].end(); iter++){
					if (mySet.find(*iter) != currentParent && mySet.find(*iter) != minIndex){
						tmpList.push_back(*iter);
					}
				}
				for (iter = nodeMapNeighbourList[minIndex].begin(); iter != nodeMapNeighbourList[minIndex].end(); iter++){
					if (mySet.find(*iter) != currentParent && mySet.find(*iter) != minIndex){
						tmpList.push_back(*iter);
					}
				}

				nodeMapNeighbourList[currentParent].clear();
				nodeMapNeighbourList[currentParent] = tmpList;

				//3. merge
				mySet.merge(currentParent, minIndex);

				//4. insert Heap
				HEAPNODE tt;
				tt.id = currentParent;
				tt.MSE = tmp.MSE;

				//printf("id:%d	MSE:%f	numPixels:%d\n", tmp.id, tmp.MSE, tmp.num);
				myHeap.push(tt);
			}
		}
	}

	void simpleClustering()
	{
		while (myHeap.size > 0)
		{
			HEAPNODE current = myHeap.data[1];
			myHeap.pop();
			int currentParent = mySet.find(current.id);
			if (current.id != currentParent){
				continue;
			}
			if (nodeMap[currentParent].num > T_NUM){
				continue;
			}
			bool flag = false;
			std::list<int> tmpList;
			tmpList.clear();
			for (std::list<int>::iterator iter = nodeMapNeighbourList[currentParent].begin(); iter != nodeMapNeighbourList[currentParent].end(); iter++)
			{
				int k = *iter;
				int kParent = mySet.find(k);
				if (kParent == currentParent || nodeMap[kParent].num > T_NUM) {
					continue;
				}
				if (nodeMap[currentParent].normal[0] * nodeMap[kParent].normal[0] + nodeMap[currentParent].normal[1] * nodeMap[kParent].normal[1] +
					nodeMap[currentParent].normal[2] * nodeMap[kParent].normal[2] > cosThreshold)
				{
					flag = true;

					//1. update current node 
					nodeMap[currentParent].sumX += nodeMap[kParent].sumX;
					nodeMap[currentParent].sumY += nodeMap[kParent].sumY;
					nodeMap[currentParent].sumZ += nodeMap[kParent].sumZ;
					nodeMap[currentParent].sumX2 += nodeMap[kParent].sumX2;
					nodeMap[currentParent].sumY2 += nodeMap[kParent].sumY2;
					nodeMap[currentParent].sumZ2 += nodeMap[kParent].sumZ2;
					nodeMap[currentParent].sumXY += nodeMap[kParent].sumXY;
					nodeMap[currentParent].sumXZ += nodeMap[kParent].sumXZ;
					nodeMap[currentParent].sumYZ += nodeMap[kParent].sumYZ;
					nodeMap[currentParent].num += nodeMap[kParent].num;

					//2. update neighbour list
					for (std::list<int>::iterator iter = nodeMapNeighbourList[kParent].begin(); iter != nodeMapNeighbourList[kParent].end(); iter++){
						if (mySet.find(*iter) != currentParent && mySet.find(*iter) != kParent){
							tmpList.push_back(*iter);
						}
					}
					//3. merge
					mySet.merge(currentParent, kParent);
				}
			}
			if (flag == true)
			{
				nodeMapNeighbourList[currentParent].clear();
				nodeMapNeighbourList[currentParent] = tmpList;
				calculateNormal(nodeMap[currentParent]);

				//check normal
				double x = nodeMap[currentParent].sumX / nodeMap[currentParent].num;
				double y = nodeMap[currentParent].sumY / nodeMap[currentParent].num;
				double z = nodeMap[currentParent].sumZ / nodeMap[currentParent].num;
				if (nodeMap[currentParent].normal[0] * x + nodeMap[currentParent].normal[1] * y + nodeMap[currentParent].normal[2] * z >0){
					//puts("found!!!");
					nodeMap[currentParent].normal[0] = -nodeMap[currentParent].normal[0];
					nodeMap[currentParent].normal[1] = -nodeMap[currentParent].normal[1];
					nodeMap[currentParent].normal[2] = -nodeMap[currentParent].normal[2];
				}

				HEAPNODE tt;
				tt.id = currentParent;
				tt.MSE = nodeMap[currentParent].MSE;
				//printf("id:%d	MSE:%f	numPixels:%d\n", tmp.id, tmp.MSE, tmp.num);
				myHeap.push(tt);
			}
		}
	}

};

#endif
