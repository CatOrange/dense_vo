/*
 * PrimeSenseCam.h
 *
 *  Created on: 20 Feb, 2015
 *      Author: eeuser
 *
 *      Abstraction layer for Asus Xtion Pro using OpenNI
 *      	Code written using : http://com.occipital.openni.s3.amazonaws.com/OpenNI_Programmers_Guide.pdf
 *	 						& OpenNI Reference
 */

#ifndef PRIMESENSECAM_H_
#define PRIMESENSECAM_H_

// STandard Headers
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

// OpenNI
#include <openni2/OpenNI.h>


// OpenCV
#include <opencv2/opencv.hpp>


class PrimeSenseCam
{
public:
	PrimeSenseCam()
	{

		// Note, setting tuned for Asus Xtion Pro Live.
		// In case another cam please refer to OpenNI programming guide and examples

		if( openni::OpenNI::initialize() != openni::STATUS_OK )
		{
			cerr << "Fail\n";
			exit(1);
		}


		// Opening Any available device
		const char * deviceuri = openni::ANY_DEVICE;
		if( device.open(deviceuri) != openni::STATUS_OK )
		{
			cout<< "Device Opening failed\n";
			exit(1);
		}
		cout<< "Device Vendor : "<< device.getDeviceInfo().getVendor() << endl;
		cout<< "Device Name : "<< device.getDeviceInfo().getName() << endl;
		if( device.hasSensor(openni::SENSOR_COLOR) )
			cout<< "This device has a Color Sensor\n";

		if( device.hasSensor(openni::SENSOR_DEPTH) )
			cout<< "This device has a Depth Sensor\n";

		if( device.hasSensor(openni::SENSOR_IR) )
			cout<< "This device has an IR Sensor\n";

		setDeviceParams();

		// Opening video stream
		if( stream.create( device, openni::SENSOR_COLOR ) != openni::STATUS_OK )
		{
			cerr << "Failed to create video stream\n";
			exit(1);
		}
		if( depthStream.create( device, openni::SENSOR_DEPTH ) != openni::STATUS_OK )
		{
			cerr << "Failed to depth create video stream\n";
			exit(1);
		}

		setStreamParams();

		pframe =  new openni::VideoFrameRef();
		dframe = new openni::VideoFrameRef();

		//start stream
		start();

	}
	virtual ~PrimeSenseCam()
	{
		delete pframe;
		delete dframe;
		stream.stop();
		depthStream.stop();
		stream.destroy();
		depthStream.destroy();
		device.close();
		//openni::OpenNI::shutdown();
	}


	// start streaming data
	void start()
	{
		// start streams
		depthStream.start();
		stream.start();

		// load one frame to test
		if( depthStream.readFrame(dframe) == openni::STATUS_OK )
		{
			cout<< "Frame depth read success\n";
		}
		else
		{
			cout << "depth error" << endl ;
		}

		if( stream.readFrame(pframe) == openni::STATUS_OK ){
			cout<< "Frame read success\n";
		}
		else{
			cout << "frame error" << endl ;
		}

		pWidth = pframe->getWidth();
		pHeight = pframe->getHeight();
		pDataSize = pframe->getDataSize();

		dWidth = dframe->getWidth();
		dHeight = dframe->getHeight();
		dDataSize = dframe->getDataSize();
	}

	// sets device params like mirroring, sync enable, etc
	void setDeviceParams()
	{
		//enable framesync
		cout << "Setting Depth-Color Sync\n";
		device.setDepthColorSyncEnabled(true);

		//device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_OFF);
		if( device.isImageRegistrationModeSupported(device.getImageRegistrationMode()) )
			cout<< "Depth-Color Registration is supported\n";

	}

	void setStreamParams()
	{
		cout << "Disable Mirroring\n";
		depthStream.setMirroringEnabled(false);
		stream.setMirroringEnabled(false);


		// setting resolution, FPS, pixel-format
		openni::VideoMode vMode;
		vMode.setFps(30);
		vMode.setResolution(640,480);
		vMode.setPixelFormat(openni::PIXEL_FORMAT_RGB888 );
		openni::Status vmodeColSta = stream.setVideoMode(vMode);
		vMode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);
		openni::Status vmodeDepSta = depthStream.setVideoMode(vMode);
		if ( (vmodeColSta != openni::STATUS_OK) || (vmodeDepSta != openni::STATUS_OK))
		{
			cout << "[setStreamParams] error: Video format not supported..." << endl;
			exit(1);
		}


		// auto exposure & white balance
		openni::CameraSettings * camsettings = stream.getCameraSettings();
		camsettings->setAutoExposureEnabled(false);
		camsettings->setAutoWhiteBalanceEnabled(false);
	}

	void retriveFrame(cv::Mat& rgbImage, cv::Mat& depthImage)
	{
		// check if memory is allocated
		if( rgbImage.rows != pHeight || rgbImage.cols != pWidth || rgbImage.type() != CV_8UC3 )
		{
			rgbImage = cv::Mat::zeros(pframe->getHeight(), pframe->getWidth(), CV_8UC3);
		}

		if( depthImage.rows != pHeight || depthImage.cols != pWidth || depthImage.type() != CV_16U )
		{
			depthImage = cv::Mat::zeros(pframe->getHeight(), pframe->getWidth(), CV_16U );
		}

		// Retrive frame from CAM
		if( depthStream.readFrame(dframe) != openni::STATUS_OK )
			cerr<< "Error Reading depth frame\n";

		if( stream.readFrame(pframe) != openni::STATUS_OK )
			cerr<< "Error Reading frame\n";

		unsigned char * RawData= (unsigned char*)pframe->getData();
		int k = 0 ;
		for( int i = 0 ; i < pHeight ; i++ )
		{
			for	( int j = 0 ; j < pWidth ; j++ )
			{
				rgbImage.at<cv::Vec3b>(i, j)[2] = (uchar)RawData[k++] ;
				rgbImage.at<cv::Vec3b>(i, j)[1] = (uchar)RawData[k++] ;
				rgbImage.at<cv::Vec3b>(i, j)[0] = (uchar)RawData[k++] ;
			}
		}
		//memcpy( &rgbImage.data, colorRawData, sizeof(unsigned char)*pHeight*pWidth ) ;

		openni::DepthPixel * d = (openni::DepthPixel *)dframe->getData();
		k = 0 ;
		for( int i = 0 ; i < pHeight ; i++ )
		{
			for	( int j = 0 ; j < pWidth ; j++ )
			{
				depthImage.at<unsigned short>(i, j) = (unsigned short)d[k++] ;
			}
		}
		//memcpy( &depthImage.data, d, sizeof(CV_16U)*pHeight*pWidth ) ;
		//puts("222");
	}

	int getdDataSize() const {
		return dDataSize;
	}

	int getdHeight() const {
		return dHeight;
	}

	int getdWidth() const {
		return dWidth;
	}

	int getpDataSize() const {
		return pDataSize;
	}

	int getpHeight() const {
		return pHeight;
	}

	int getpWidth() const {
		return pWidth;
	}

private:
	openni::Device device;
	openni::VideoStream stream;
	openni::VideoStream depthStream;

	openni::VideoFrameRef * pframe; //color frame
	openni::VideoFrameRef * dframe; // depth frame
	int pWidth;
	int pHeight;
	int pDataSize;

	int dWidth;
	int dHeight;
	int dDataSize;
/*
	void retriveFrame( uchar *f, cv::Mat imx )
	{
		int height = imx.rows;
		int width = imx.cols;
		int widthStep=0;
		for( int h=0 ; h<height ; h++ )
		{
			for( int w=0 ; w<width ; w++ )
			{
				imx.at<cv::Vec3b>(h,w)[0] = (uchar)f[ widthStep + 3*w + 2];
				imx.at<cv::Vec3b>(h,w)[1] = (uchar)f[ widthStep + 3*w + 1];
				imx.at<cv::Vec3b>(h,w)[2] = (uchar)f[ widthStep + 3*w ];
			}
			widthStep += (3*width);
		}
	}
*/
};

#endif /* PRIMESENSECAM_H_ */
