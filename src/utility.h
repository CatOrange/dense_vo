#ifndef __UTILITY_H
#define __UTILITY_H

#include "variableDefinition.h"
#include <vector>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

const int dx[4] = { 0, 0, -1, 1 };//up down left right
const int dy[4] = { -1, 1, 0, 0 };//up down left right
const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

#define	INDEX(i, j, n, m)		( (i)*(m) + (j)  )

inline double SQ(double a){
	return a*a;
}

inline bool linearIntepolation(double y, double x, unsigned char* pIntensity , int n, int m, double& result )
{
	int leftTopY = floorf(y);
	int leftTopX = floorf(x);
	if (leftTopX < 0 || leftTopY < 0){
		return false;
	}
	int rightDownY = leftTopY + 1;
	int rightDownX = leftTopX + 1;
	if (rightDownY >= n || rightDownX >= m){
		return false;
	}
	const double subpix_u_cur = y - leftTopY;
	const double subpix_v_cur = x - leftTopX;
	const double w_cur_tl = (1.0 - subpix_u_cur) * (1.0 - subpix_v_cur);
	const double w_cur_tr = subpix_u_cur * (1.0 - subpix_v_cur);
	const double w_cur_bl = (1.0 - subpix_u_cur) * subpix_v_cur;
	const double w_cur_br = subpix_u_cur * subpix_v_cur;

	result = w_cur_tl*pIntensity[INDEX(leftTopY, leftTopX, n, m)]
		+ w_cur_tr*pIntensity[INDEX(leftTopY, rightDownX, n, m)]
		+ w_cur_bl*pIntensity[INDEX(rightDownY, leftTopX, n, m)]
		+ w_cur_br*pIntensity[INDEX(rightDownY, rightDownX, n, m)];

	return true;
}

inline int findMaxContinousLength(std::vector<int>& b, int n, int& maxNum )
{
	maxNum = 1;
	int maxValue = b[0];
	int cnt = 1;
	int currentValue = b[0];
	for (int i = 1; i < n; i++)
	{
		if (b[i] == currentValue )
		{
			cnt++;
			if (cnt > maxNum )
			{
				maxNum = cnt;
				maxValue = b[i];
			}
		}
		else
		{
			cnt = 1;
			currentValue = b[i];
		}
	}
	return maxValue;
}

template<typename T>
static void pyrDownMeanSmooth(const cv::Mat& in, cv::Mat& out)
{
	Mat tmp;
	tmp.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

	for (int y = 0; y < tmp.rows; ++y)
	{
		for (int x = 0; x < tmp.cols; ++x)
		{
			int x0 = x << 1 ;
			int x1 = x0 + 1;
			int y0 = y << 1 ;
			int y1 = y0 + 1;

			tmp.at<T>(y, x) = (T)((in.at<T>(y0, x0) + in.at<T>(y0, x1) + in.at<T>(y1, x0) + in.at<T>(y1, x1)) / 4.0f);
		}
	}
	out = tmp.clone();
}

template<typename T>
static void pyrDownMedianSmooth(const cv::Mat& in, cv::Mat& out)
{
	Mat tmp;
	tmp.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

	cv::Mat in_smoothed;
	cv::medianBlur(in, in_smoothed, 3);

	for (int y = 0; y < tmp.rows; ++y)
	{
		for (int x = 0; x < tmp.cols; ++x)
		{
			tmp.at<T>(y, x) = in_smoothed.at<T>(y << 1, x << 1 );
		}
	}
	out = tmp.clone();
}

#endif
