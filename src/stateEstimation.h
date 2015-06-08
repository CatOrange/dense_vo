#ifndef __STATEESTIMATION_H
#define __STATEESTIMATION_H
#include "state.h"
#include "ros/ros.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"

class STATEESTIMATION
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  //camera parameters
  int height, width;
  CAMER_PARAMETERS* para;

  //state sequences
  int head, tail;
  int numOfState;
  STATE states[slidingWindowSize];

  //for dense tracking
  Vector3f last_delta_v, last_delta_w;
  Matrix3f last_delta_R;
  Vector3f last_delta_T;

  //track the lastest state
  Matrix3f R_k_2_c;//R_k^(k+1)
  Vector3f T_k_2_c;//T_k^(k+1)
  Vector3f v_c;

  Vector3f gravity_b0;
  float currentTime;

  bool twoWayMarginalizatonFlag = false;//false, marginalize oldest; true, marginalize newest
  bool mask[IMAGE_HEIGHT][IMAGE_WIDTH] ;

  STATEESTIMATION(int hh, int ww, CAMER_PARAMETERS* p)
  {
    height = hh;
    width = ww;
    para = p;

    initSlidingWindow();

    last_delta_v.setZero();
    last_delta_w.setZero();
    last_delta_R.setIdentity();
    last_delta_T.setZero();
  }

  ~STATEESTIMATION(){
  }

  void initSlidingWindow()
  {
    head = 0;
    tail = -1;
    for (int i = 0; i < slidingWindowSize; i++)
    {
      states[i].id = i;
      //states[i].pts = NULL;
      if (i + 1 < slidingWindowSize){
        states[i].next = &states[i + 1];
      }
    }
    states[slidingWindowSize - 1].next = &states[0];
  }

  STATE* getKeyFrame_ptr()
  {
    int i = tail;
    while ( states[i].keyFrameFlag == false )
    {
      i--;
      if (i < 0){
        i += slidingWindowSize;
      }
    }
    return &states[i];
  }

  void insertFrame(const Mat grayImage[maxPyramidLevel], const Matrix3f& R, const Vector3f& T, const Vector3f& vel )
  {
    tail++;
    numOfState++;
    if (tail >= slidingWindowSize){
      tail -= slidingWindowSize;
    }
    STATE *current = &states[tail];

    //copy the intensity
    int n = height;
    int m = width;
    for (int i = 0; i < maxPyramidLevel; i++)
    {
      memcpy(current->intensity[i], (unsigned char*)grayImage[i].data, n*m*sizeof(unsigned char));
      n >>= 1;
      m >>= 1;
    }

    //init the state
    current->R_bk_2_b0 = R;
    current->T_bk_2_b0 = T;
    current->v_bk = vel;

    //set the keyFrameFlag
    current->keyFrameFlag = false;
 }

  void prepareKeyFrame(STATE *current, const Mat depthImage[maxPyramidLevel],
                       Mat& gradientMapForDebug )
  {
    current->keyFrameFlag = true;

    //for debug purpose
    {
      int n = height;
      int m = width;
      n >>= beginPyramidLevel ;
      m >>= beginPyramidLevel ;
      gradientMapForDebug = Mat(n, m, CV_8UC3);
      for (int i = 0; i < n; i++)
      {
        for (int j = 0; j < m; j++)
        {
          gradientMapForDebug.at<cv::Vec3b>(i, j)[0] = current->intensity[beginPyramidLevel][INDEX(i, j, n, m)];
          gradientMapForDebug.at<cv::Vec3b>(i, j)[1] = current->intensity[beginPyramidLevel][INDEX(i, j, n, m)];
          gradientMapForDebug.at<cv::Vec3b>(i, j)[2] = current->intensity[beginPyramidLevel][INDEX(i, j, n, m)];
        }
      }
      //cv::cvtColor(grayImage[0], gradientMapForDebug, CV_GRAY2BGR);
    }

    int n = height;
    int m = width;
    //copy the depth image
    for (int i = 0; i < maxPyramidLevel; i++)
    {
      memcpy(current->depthImage[i], (float*)depthImage[i].data, n*m*sizeof(float));
      n >>= 1;
      m >>= 1;
    }

    //init the graident map
    current->computeGradientMap();

    //init the pixel info in a frame
    for (int level = maxPyramidLevel - 1; level >= beginPyramidLevel; level--)
    {
      int n = height >> level;
      int m = width >> level;
      float* pDepth = current->depthImage[level];
      unsigned char*pIntensity = current->intensity[level];
      float* pGradientX = current->gradientX[level];
      float* pGradientY = current->gradientY[level];

      int validNum = 0;
      vector<GRADIENTNODE> gradientList(n*m);
      for (int u = 0; u < n; u++)
      {
        for (int v = 0; v < m; v++)
        {
          int k = INDEX(u, v, n, m);
          float Z = pDepth[k];
          if ( Z < zeroThreshold ) {
            continue;
          }
          if ( level <= 1 && SQ(pGradientX[k]) + SQ(pGradientY[k]) < graidientThreshold){
            pDepth[k] = 0.0 ;
            continue;
          }
          gradientList[validNum].cost = SQ(pGradientX[k]) + SQ(pGradientY[k]);
          gradientList[validNum].u = u;
          gradientList[validNum].v = v;
          validNum++;
        }
      }
      printf("dense tracking validNum: %d\n", validNum);

//      sort(&gradientList[0], &gradientList[validNum]);

//      int bin[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
//      int tmpCnt = 0;
//      int numThrehold = (minDenseTrackingNum >> level) / 8;
//      //int numThrehold = 1000000000 ;
//      for (int i = 0; i < validNum; i++)
//      {
//        int u = gradientList[i].u;
//        int v = gradientList[i].v;
//        int k = INDEX(u, v, n, m);
//        int index = angelSpace(pGradientX[k], pGradientY[k]);
//        if (bin[index] < numThrehold){
//          bin[index]++;
//          gradientList[tmpCnt++] = gradientList[i];
//        }
//      }

//      //validNum = std::min( validNum, minDenseTrackingNum >> level ) ;
//      validNum = tmpCnt;

      if (level == beginPyramidLevel)//set the mask for dense BA
      {
        for (int cnt = 0; cnt < validNum; cnt++){
          int u = gradientList[cnt].u;
          int v = gradientList[cnt].v;

          gradientMapForDebug.at<Vec3b>(u, v)[0] = 0;
          gradientMapForDebug.at<Vec3b>(u, v)[1] = 255;
          gradientMapForDebug.at<Vec3b>(u, v)[2] = 0;
        }
      }

      PIXEL_INFO_IN_A_FRAME& currentPixelInfo = current->pixelInfo[level];
      currentPixelInfo.piList.resize(3, validNum);
      currentPixelInfo.Aij.clear();
      currentPixelInfo.Aij.resize(validNum);
      currentPixelInfo.AijTAij.clear();
      currentPixelInfo.AijTAij.resize(validNum);
      currentPixelInfo.intensity.clear();
      currentPixelInfo.intensity.resize(validNum);
      currentPixelInfo.goodPixel.clear();
      currentPixelInfo.goodPixel.resize(validNum);

      MatrixXf oneBytwo(1, 2);
      MatrixXf twoBySix(2, 6);
      MatrixXf oneBySix(1, 6);
      MatrixXf oneBySixT(6, 1);


      omp_set_num_threads(ompNumThreads);
      #pragma omp parallel for
      for (int cnt = 0; cnt < validNum; cnt++)
      {
        int u = gradientList[cnt].u;
        int v = gradientList[cnt].v;

        int k = INDEX(u, v, n, m);
        float Z = pDepth[k];

        float X = (v - para->cx[level]) * Z / para->fx[level];
        float Y = (u - para->cy[level]) * Z / para->fy[level];

        currentPixelInfo.piList(0, cnt) = X;
        currentPixelInfo.piList(1, cnt) = Y;
        currentPixelInfo.piList(2, cnt) = Z;
        currentPixelInfo.intensity[cnt] = pIntensity[k];
        currentPixelInfo.goodPixel[cnt] = true ;

        oneBytwo(0, 0) = pGradientX[k];
        oneBytwo(0, 1) = pGradientY[k];

        twoBySix(0, 0) = para->fx[level] / Z;
        twoBySix(0, 1) = 0;
        twoBySix(0, 2) = -X * para->fx[level] / SQ(Z);
        twoBySix(1, 0) = 0;
        twoBySix(1, 1) = para->fy[level] / Z;
        twoBySix(1, 2) = -Y * para->fy[level] / SQ(Z);

        twoBySix(0, 3) = twoBySix(0, 2) * Y;
        twoBySix(0, 4) = twoBySix(0, 0)*Z - twoBySix(0, 2)*X;
        twoBySix(0, 5) = -twoBySix(0, 0)*Y;
        twoBySix(1, 3) = -twoBySix(1, 1)*Z + twoBySix(1, 2)*Y;
        twoBySix(1, 4) = -twoBySix(1, 2)* X;
        twoBySix(1, 5) = twoBySix(1, 1)* X;

        oneBySix = oneBytwo*twoBySix;
        oneBySixT = oneBySix.transpose();

        currentPixelInfo.Aij[cnt] = oneBySixT;
        currentPixelInfo.AijTAij[cnt] = oneBySixT * oneBySix;
      }
    }
  }

  float maxAbsValueOfVector(const VectorXf&a)
  {
    float maxValue = fabs(a(0));
    for (int i = 1; i < 6; i++)
    {
      float tmp = fabs(a(i));
      if (tmp > maxValue){
        maxValue = tmp;
      }
    }
    return maxValue;
  }

  Matrix3f vectorToSkewMatrix(const Vector3f& w)
  {
    Matrix3f skewW(3, 3);
    skewW(0, 0) = skewW(1, 1) = skewW(2, 2) = 0;
    skewW(0, 1) = -w(2);
    skewW(1, 0) = w(2);
    skewW(0, 2) = w(1);
    skewW(2, 0) = -w(1);
    skewW(1, 2) = -w(0);
    skewW(2, 1) = w(0);

    return skewW;
  }

  void updateR_T(Matrix3f& R, Vector3f& T, const Vector3f& v, const Vector3f& w, Matrix3f& incR, Vector3f& incT)
  {
    Matrix3f skewW = vectorToSkewMatrix(w);


    float theta = sqrt(w.squaredNorm());
    Matrix3f deltaR = Matrix3f::Identity() + (sin(theta) / theta)*skewW + ((1 - cos(theta)) / (theta*theta))*skewW*skewW;
    Vector3f deltaT = (Matrix3f::Identity() + ((1 - cos(theta)) / (theta*theta)) *skewW + ((theta - sin(theta)) / (theta*theta*theta)*skewW*skewW)) * v;

    Matrix3f newR = R*deltaR;
    Vector3f newT = R*deltaT + T;

    incT = incR*deltaT + incT;
    incR = incR*deltaR;

    R = newR;
    T = newT;
  }

  float denseTrackingHuber(STATE* current, const Mat grayImage[maxPyramidLevel], Matrix3f& R, Vector3f& T, MatrixXf& lastestATA, float& averageResidial )
  {
    //no assumption on angular and linear velocity
    Matrix3f tmpR = R;
    Vector3f tmpT = T;
    float validPercent = 0 ;
    averageResidial = 1000000000.0 ;

    //linear assumption on angular and linear velocity
    //Matrix3f tmpR = last_delta_R * R ;
    //Vector3f tmpT = last_delta_R * T + last_delta_T;

    Matrix3f incR = Matrix3f::Identity();
    Vector3f incT = Vector3f::Zero();
    for (int level = maxPyramidLevel - 1; level >= beginPyramidLevel; level--)
    {
      int n = height >> level;
      int m = width >> level;
      unsigned char *nextIntensity = grayImage[level].data;
      PIXEL_INFO_IN_A_FRAME& currentPixelInfo = current->pixelInfo[level];
      float lastError = 100000000000.0;
      last_delta_v.Zero();
      last_delta_w.Zero();
      int ith = 0 ;
      for (; ith < maxIteration[level]; ith++)
      {
        int actualNum = 0;
        int goodPixelNum = 0 ;
        float currentError = 0;
        MatrixXf ATA = MatrixXf::Zero(6, 6);
        VectorXf ATb = VectorXf::Zero(6);
        MatrixXf pi2List = tmpR * currentPixelInfo.piList;
        int validNum = currentPixelInfo.intensity.size();

        for (int i = 0; i < validNum; i++)
        {
          if ( currentPixelInfo.goodPixel[i] == false ){
            continue ;
          }
          goodPixelNum++ ;
          Vector3f p2 = pi2List.block(0, i, 3, 1) + tmpT;
          //Vector3f p2 = tmpR* currentPixelInfo.piList.block(0, i, 3, 1) + tmpT;

#ifdef DEBUG_DENSETRACKING
          Vector3f p1 = currentPixelInfo.piList.block(0, i, 3, 1);
          int u = int(p1(1)*para->fy[level] / p1(2) + para->cy[level] + 0.5);
          int v = int(p1(0)*para->fx[level] / p1(2) + para->cx[level] + 0.5);

          gradientMap.at<cv::Vec3b>(u, v)[0] = 0;
          gradientMap.at<cv::Vec3b>(u, v)[1] = 255;
          gradientMap.at<cv::Vec3b>(u, v)[2] = 0;
#endif

          int u2 = int(p2(1)*para->fy[level] / p2(2) + para->cy[level] + 0.5);
          int v2 = int(p2(0)*para->fx[level] / p2(2) + para->cx[level] + 0.5);

          //float u2 = p2(1)*para->fy[level] / p2(2) + para->cy[level];
          //float v2 = p2(0)*para->fx[level] / p2(2) + para->cx[level];
          //float reprojectIntensity;
          //if (linearIntepolation(u2, v2, nextIntensity, n, m, reprojectIntensity) == false){
          //	continue;
          //}
          if (u2 < 0 || u2 >= n || v2 < 0 || v2 >= m){
            continue;
          }

          //#ifdef DEBUG_DENSETRACKING
          //						next.at<cv::Vec3b>(u2, v2)[0] = proportion*next.at<cv::Vec3b>(u2, v2)[0] + (1 - proportion) * R[pointsLabel[i]];
          //						next.at<cv::Vec3b>(u2, v2)[1] = proportion*next.at<cv::Vec3b>(u2, v2)[1] + (1 - proportion) * G[pointsLabel[i]];
          //						next.at<cv::Vec3b>(u2, v2)[2] = proportion*next.at<cv::Vec3b>(u2, v2)[2] + (1 - proportion) * B[pointsLabel[i]];
          //#endif

          float w = 1.0 ;
          //float w = 1.0 /SQ(currentPixelInfo.piList(2, i));
          float r = currentPixelInfo.intensity[i] - nextIntensity[INDEX(u2, v2, n, m)];
          //float r = currentPixelInfo.intensity[i] - reprojectIntensity;
          float r_fabs = fabs(r);

#ifdef WEIGHTEDCOST
          if (r_fabs > huberKernelThreshold){
            w *= huberKernelThreshold / (r_fabs);
          }
#endif

          currentError += w*r_fabs;
          actualNum++;

          ATA += w*currentPixelInfo.AijTAij[i];
          ATb -= (w*r)*currentPixelInfo.Aij[i];
        }


        validPercent = float(actualNum) / (n*m) ;
        if (validPercent < 0.05 ){
          return validPercent;
        }
        if ( float(actualNum) / goodPixelNum < 0.5 ){
          return validPercent = 0 ;
        }

        currentError = averageResidial = currentError/actualNum ;
        if (currentError > lastError){
          //revert
          updateR_T(tmpR, tmpT, -last_delta_v, -last_delta_w, incR, incT);
          break;
        }
        else
        {
          if ( currentError / lastError > 0.999f ){
            ith = maxIteration[level] ;
          }
          lastError = currentError;
        }

        LDLT<MatrixXf> ldltOfA = ATA.ldlt();
        ComputationInfo info = ldltOfA.info();
        if (info == Success)
        {
          lastestATA = ATA;

          VectorXf x = ldltOfA.solve(ATb);

#ifdef DEBUG_DENSETRACKING
          MatrixXf L = lltOfA.matrixL();
          cout << "currntError: " << currentError / actualNum << endl;
          // cout << "lltofATA.L() " << ith << ":\n" <<  L << endl ;
          // cout << "ATb " << ith << ":\n" << ATb << endl ;
          // cout << "dx " << ith << ":\n" << x.transpose() << endl;
#endif
          //printf("x.norm()=%f\n", x.norm() );
          Vector3f w, v;
          v(0) = -x(0);
          v(1) = -x(1);
          v(2) = -x(2);
          w(0) = -x(3);
          w(1) = -x(4);
          w(2) = -x(5);
          updateR_T(tmpR, tmpT, v, w, incR, incT);
          last_delta_v = v;
          last_delta_w = w;
        }
        else {
          ROS_WARN("level=%d, iter=%d can not solve Ax = b", level, ith );
          ROS_WARN("actual=%d goodNum=%d\n", actualNum, goodPixelNum ) ;
          break;
        }
      }//end of interation
      printf("dT, lvl=%d iter=%d\n", level, ith ) ;
    }//end of pyramid level
    R = tmpR;
    T = tmpT;

    return validPercent;
  }

  void badPixelFiltering(STATE* current, const Mat grayImage[maxPyramidLevel], Matrix3f& R, Vector3f& T)
  {
    for (int level = maxPyramidLevel - 1; level >= beginPyramidLevel; level--)
    {
      int n = height >> level;
      int m = width >> level;
      unsigned char *nextIntensity = grayImage[level].data;
      PIXEL_INFO_IN_A_FRAME& currentPixelInfo = current->pixelInfo[level];

      MatrixXf pi2List = R * currentPixelInfo.piList;
      int validNum = currentPixelInfo.intensity.size();

      for (int i = 0; i < validNum; i++)
      {
        if ( currentPixelInfo.goodPixel[i] == false ){
          continue ;
        }
        Vector3f p2 = pi2List.block(0, i, 3, 1) + T;

        int u2 = int(p2(1)*para->fy[level] / p2(2) + para->cy[level] + 0.5);
        int v2 = int(p2(0)*para->fx[level] / p2(2) + para->cx[level] + 0.5);

        if (u2 < 0 || u2 >= n || v2 < 0 || v2 >= m){
          continue;
        }

        float r = currentPixelInfo.intensity[i] - nextIntensity[INDEX(u2, v2, n, m)];
        //float r = currentPixelInfo.intensity[i] - reprojectIntensity;
        float r_fabs = fabs(r);

        if (r_fabs > huberKernelThreshold){
          currentPixelInfo.goodPixel[i] = false ;
        }
      }
    }//end of pyramid level
  }

  void visualizeResidualMap(STATE* current, const Mat grayImage[maxPyramidLevel], const Matrix3f& R,const Vector3f& T, Mat& residualMap, Mat& display)
  {
    int level = beginPyramidLevel;
    int n = height >> level;
    int m = width >> level;
    unsigned char *nextIntensity = grayImage[level].data;
    PIXEL_INFO_IN_A_FRAME& currentPixelInfo = current->pixelInfo[level];

    MatrixXf pi2List = R * currentPixelInfo.piList;
    int validNum = currentPixelInfo.intensity.size();
    residualMap.setTo( uchar(0) ) ;
    float max_r = 0.0 ;
    for (int i = 0; i < validNum; i++)
    {
      if ( currentPixelInfo.goodPixel[i] == false ){
        continue ;
      }
      Vector3f p2 = pi2List.block(0, i, 3, 1) + T;

      int u2 = int(p2(1)*para->fy[level] / p2(2) + para->cy[level] + 0.5);
      int v2 = int(p2(0)*para->fx[level] / p2(2) + para->cx[level] + 0.5);

      if (u2 < 0 || u2 >= n || v2 < 0 || v2 >= m){
        continue;
      }

      float r = currentPixelInfo.intensity[i] - nextIntensity[INDEX(u2, v2, n, m)];
      //float r = currentPixelInfo.intensity[i] - reprojectIntensity;
      float r_fabs = fabs(r)*10;
      if ( r_fabs > max_r ){
        max_r = r_fabs ;
      }
      residualMap.at<uchar>(u2, v2) = r_fabs>255?255:(uchar)r_fabs ;
    }
    applyColorMap(residualMap, display, COLORMAP_RAINBOW ) ;
    //imshow("residualMap", residualMap ) ;
  }

  inline void insertMatrixToSparseMatrix(SparseMatrix<float>& to, const MatrixXf& from, int y, int x, int n, int m)
  {
    for (int i = 0; i < n; i++){
      for (int j = 0; j < m; j++){
        to.insert(y + i, x + j) = from(i, j);
      }
    }
  }

};

#endif
