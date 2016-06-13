
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include "ConvNetCL.h"
#include <ctype.h>
#include <fstream>
#include <time.h>
#include <thread>
#include <cassert>

//BOINC
#ifdef _BOINC_APP_
// #ifdef _WIN32
// #include "boinc_win.h"
// #include "str_util.h"
// #include "diagnostics.h"
// #include "util.h"
// #include "filesys.h"
// #include "boinc_api.h"
// #include "mfile.h"
// #else
// #include "boinc/diagnostics.h"
// #include "boinc/util.h"
// #include "boinc/filesys.h"
// #include "boinc/boinc_api.h"
// #include "boinc/mfile.h"
// #endif
#include "../../test/boincAPI/boinc_api.h"
#endif



using namespace cv;
using namespace std;

typedef vector<vector<vector<double> > > imVector;

char *inPath, *outPath;
int imageNum = 0;
int stride = 1;
bool __useGPU = true;

unsigned int __frameNum = 0;

int __width;
int __height;

int __momentRed = 0;

//BOINC FUNCTIONS
std::string getBoincFilename(std::string filename) throw(std::runtime_error) {
    std::string resolved_path = filename;
	#ifdef _BOINC_APP_
	    if(boinc_resolve_filename_s(filename.c_str(), resolved_path)) {
	        LOG(ERROR) << "Could not resolve filename '" << filename.c_str() << "'";
	        throw std::runtime_error("Boinc could not resolve filename");
	    }
	#endif
    return resolved_path;
}

void writeCheckpoint() throw(std::runtime_error)
{
	string resolved_checkpoint_name = getBoincFilename("checkpoint.yml");
	cv::FileStorage outfile(resolved_checkpoint_name, cv::FileStorage::WRITE);
	if(!outfile.isOpened())
		throw std::runtime_error("Checkpoint file could not be opened");

	outfile << "FRAME_NUM" << __frameNum;
	outfile.release();
}

bool readCheckpoint()
{
	string resolved_checkpoint_name = getBoincFilename("checkpoint.yml");
	cv::FileStorage infile(resolved_checkpoint_name, cv::FileStorage::READ);
	if(!infile.isOpened())
		return false;

	infile["FRAME_NUM"] >> __frameNum;

	infile.release();
	return true;
}

//END BOINC FUNCTIONS

void resize3DVector(vector<vector<vector<double> > > &vect, int width, int height, int depth)
{
	vect.resize(width);
	for(int i=0; i < width; i++)
	{
		vect[i].resize(height);
		for(int j=0; j < height; j++)
		{
			vect[i][j].resize(depth);
		}
	}
}

void setAll3DVector(vector<vector<vector<double> > > &vect, double val)
{
	for(int i=0; i< vect.size(); i++)
	{
		for(int j=0; j< vect[i].size(); j++)
		{
			for(int k=0; k< vect[i][j].size(); k++)
			{
				vect[i][j][k] = val;
			}
		}
	}
}

string secondsToString(time_t seconds)
{
	time_t secs = seconds%60;
	time_t mins = (seconds%3600)/60;
	time_t hours = seconds/3600;
	char out[100];
	if(hours > 0)
		sprintf(out,"%ld hours, %ld mins, %ld secs",hours,mins,secs);
	else if(mins > 0)
		sprintf(out,"%ld mins, %ld secs",mins,secs);
	else
		sprintf(out,"%ld secs",secs);
	string outString = out;
	return outString;
}

double vectorSumSq(const vector<double>& vect)
{
	double sum=0;
	for(int i=0; i<vect.size(); i++)
		sum += vect[i] * vect[i];
	return sum;
}
double vectorSum(const vector<double>& vect)
{
	double sum=0;
	for(int i=0; i<vect.size(); i++)
		sum += vect[i];
	return sum;
}

void squareElements(vector<vector<vector<double> > >& vect)
{
	for(int i=0; i < vect.size(); i++)
		for(int j=0; j < vect[i].size(); j++)
			for(int k=0; k < vect[i][j].size(); k++)
				vect[i][j][k] = vect[i][j][k] * vect[i][j][k];
}

bool allElementsEquals(vector<double>& array)
{
	for(int i=1; i < array.size(); i++)
	{
		if(array[0] != array[i])
			return false;
	}
	return true;
}

void _t_convertColorMatToVector(const Mat& m , vector<vector<vector<double> > > &dest, int row)
{
	for(int j=0; j< m.cols; j++)
	{
		const Vec3b& curPixel = m.at<Vec3b>(row,j);
		dest[row][j][0] = curPixel[0];
		dest[row][j][1] = curPixel[1];
		dest[row][j][2] = curPixel[2];
	}
}

void convertColorMatToVector(const Mat& m, vector<vector<vector<double> > > &dest)
{
	if(m.type() != CV_8UC3)
	{
		throw "Incorrect Mat type. Must be CV_8UC3.";
	}

	int width2 = m.rows;
	int height2 = m.cols;
	int depth2 = 3;
	//resize dest vector
	resize3DVector(dest,width2,height2,depth2);
	thread *t = new thread[width2];
	
	for(int i=0; i< width2; i++)
	{
		t[i] = thread(_t_convertColorMatToVector,ref(m),ref(dest),i);
	}

	for(int i=0; i< width2; i++)
	{
		t[i].join();
	}

	//delete t;
}

/*
 * The inner for loop gets the confidences for each pixel in the image. If a pixel is in more than one subimage
 * (i.e. the stride is less than the subimage size), then the confidences from each subimage is added.
 */
void breakUpImage(Mat& image, Net& net, VideoWriter& outVideo, ofstream& outcsv)
{
	//cout << "starting breakUpImage" << endl;
	//Mat image = imread(imageName,1);
	int numrows = image.rows;
	int numcols = image.cols;
	//printf("%s rows: %d, cols: %d\n",imageName, numrows,numcols);
	int length = 0;
	char tempout[255];

	vector<vector< vector<double> > > fullImage; //2 dims for width and height, last dim for each possible category
	resize3DVector(fullImage,numrows,numcols,net.getNumClasses());
	setAll3DVector(fullImage,0);
	vector<imVector> imageRow(0); // this will hold all subimages from one row
	vector<int> calcedClasses(0);
	vector<vector<double> > confidences(0);//for the confidence for each category for each image
		//the outer vector is the image, the inner vector is the category, the double is output(confidence) of the softmax

	//cout << "here" << endl;
	int numrowsm32 = numrows-32;
	int numcolsm32 = numcols-32;

	for(int i=0; i <= numrowsm32; i+=stride)
	{
		imageRow.resize(0);
		if(i != 0)
		{
			//cout << string(length,'\b');
		}
		//sprintf(tempout,"row %d of %d (%d)\n",i,numrowsm32,numrows);
		string tempstring(tempout); length = tempstring.length();
		//cout << "row " << i << " of " << numrows << endl;
		//cout << tempout;
		//get all subimages from a row
		for(int j=0; j<= numcolsm32; j+=stride) //NOTE: each j is a different subimage
		{
			const Mat out = image(Range(i,i+32),Range(j,j+32));
			//if((i == 0 || i == numrows-32) && j== 0)
				//cout << out << endl << endl;
			//printf("i: %d, j: %d\n",i,j);
			imageRow.resize(imageRow.size()+1);
			convertColorMatToVector(out,imageRow.back());
		}
		//set them as the data in the net
		//preprocess(imageRow);
		net.setData(imageRow);
		net.run();
		net.getConfidences(confidences); //gets the confidence for each category for each image
		//if((i == 0 || i == numrows-32))
		//cout << "row: " << i << endl;
		//printVector(confidences[0]);
		//cout << "conf got" << endl;
		int curImage = 0;
		for(int j=0; j<= numcolsm32; j+=stride) //NOTE: each iteration of this loop is a different subimage
		{
			for(int ii=i; ii < i+32 && ii < numrows; ii++)
			{
				for(int jj=j; jj < j+32 && jj < numcols; jj++)
				{
					for(int cat = 0; cat < confidences[curImage].size(); cat++)
					{
						//printf("%d %d %d %d\n",i,j,jj,cat);
						fullImage[ii][jj][cat] += confidences[curImage][cat];
					}
				}
			}
			curImage++;
		}
	}
	//cout << endl;
	//printVector(fullImage);

	squareElements(fullImage);
	
	//now we have the confidences for every pixel in the image
	//so get the category for each pixel and make a new image from it
	Mat outputMat(numrows,numcols,CV_8UC3);
	assert(__width == outputMat.cols);
	assert(__height == outputMat.rows);
	int redElement = 0;
	for(int i=0; i < numrows; i++)
	{
		for(int j=0; j < numcols; j++)
		{
			double sumsq = vectorSum(fullImage[i][j]);
			for(int n=0; n < fullImage[i][j].size(); n++)
			{
				//fullImage[i][j][n] = fullImage[i][j][n] * fullImage[i][j][n] / sumsq;
				fullImage[i][j][n] /= sumsq;
			}

			//write the pixel
			Vec3b& outPix = outputMat.at<Vec3b>(i,j);
			//int maxEle = getMaxElementIndex(fullImage[i][j]);
			if(allElementsEquals(fullImage[i][j]))
			{
				outPix[0] = 0; outPix[1] = 255; outPix[2] = 0; // green
			}
			else
			{
				double blue = 255*fullImage[i][j][0];
				outPix[0] = blue; // blue
				outPix[1] = 0;	  //green
				double red = 255*fullImage[i][j][1];
				outPix[2] = red;  // red
				if(red > 150) //red > 50 || red > blue
					redElement += (int)(red);

				//if(red > 200)
				//	cout << red << endl;
			}
			/*//old
			Vec3b& outPix = outputMat.at<Vec3b>(i,j);
			int maxEle = getMaxElementIndex(fullImage[i][j]);
			if(allElementsEquals(fullImage[i][j]))
			{
				outPix[0] = 0; outPix[1] = 255; outPix[2] = 0; // green
			}
			else if(maxEle == 0)
			{
				outPix[0] = 255; outPix[1] = 0; outPix[2] = 0; // blue
			}
			else if(maxEle == 1)
			{
				outPix[0] = 0; outPix[1] = 0; outPix[2] = 255; // red
			}*/
		}
	}
	__momentRed = .8*__momentRed + .8*redElement;

	outVideo.write(outputMat);
	outcsv << __momentRed << "," << __frameNum/10.0 << "\n";
	//outVideo << outputMat;
}

void breakUpVideo(const char* videoName, Net& net, unsigned int startFrame = 0)
{
	VideoCapture video(videoName);
	if(!video.isOpened())
	{
		cout << "Could not open video: " << videoName << endl;
		return;
	}

	if(video.get(CV_CAP_PROP_FRAME_WIDTH) < 32 || video.get(CV_CAP_PROP_FRAME_HEIGHT) < 32)
	{
		printf("The video %s is too small in at least one dimension. Minimum size is 32x32.\n",videoName);
		return;
	}

	__momentRed = 0;

	char outName[255], outNameCSV[255];
	string origName(videoName);
	size_t dot = origName.rfind('.');
	const char *noExtension = origName.substr(0,dot).c_str();
	//const char *extension = origName.substr(dot).c_str();

	sprintf(outName,"%s_prediction%s",noExtension,".avi");//extension);
	sprintf(outNameCSV,"%s_prediction.csv",noExtension);
	
	//cout << "writing " << outName << endl;

	VideoWriter outVideo(outName, 
	 CV_FOURCC('M', 'J', 'P', 'G'),//-1,//video.get(CV_CAP_PROP_FOURCC),
	 10,//video.get(CV_CAP_PROP_FPS), 
	 Size(video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT)));

	ofstream outcsv;
	outcsv.open(outNameCSV);

	//cout << "FPS = " << video.get(CV_CAP_PROP_FPS) << endl;

	__width = video.get(CV_CAP_PROP_FRAME_WIDTH);
	__height = video.get(CV_CAP_PROP_FRAME_HEIGHT);

	if(!outVideo.isOpened())
	{
		cout << "Could not open out video" << endl;
		return;
	}

	Mat frame;
	for(unsigned int i = 0; i < startFrame; i++)
	{
		if(!video.grab()) // takes us to point in video. grab is quicker than read b/c it doesn't return data
			return;
	}
	//unsigned long count = 0;
	//boinc_fraction_done(0);
	while(video.read(frame))
	{
		//printf("Frame %ld of %.0lf\n", ++count, video.get(CV_CAP_PROP_FRAME_COUNT));
		//printf("Frame %ld. \t%3.4lf%%\n", ++count, video.get(CV_CAP_PROP_POS_AVI_RATIO) * 100.0);
		breakUpImage(frame, net, outVideo, outcsv);
		__frameNum++;
		boinc_fraction_done(video.get(CV_CAP_PROP_POS_AVI_RATIO));
		if(boinc_time_to_checkpoint())
		{
			writeCheckpoint();
			boinc_checkpoint_completed();
		}
	}
	
	outcsv.close();
}

int checkExtensions(char* filename)
{
	string name = filename;
	if(name.rfind(".avi")  == name.length() - 4) return 1;
	if(name.rfind(".mpeg") == name.length() - 5) return 1;
	if(name.rfind(".mp4")  == name.length() - 4) return 1;
	return 0;
}

int main(int argc, char** argv)
{
	if(argc < 3 || 5 < argc)
	{
		printf("use format: ./ConvNetVideoDriverCL cnnConfig.txt VideoPath stride=<1> gpu=<true/false> device=<device#>\n");
		return -1;
	}

	#ifdef _BOINC_APP_
	boinc_init();
	BOINC_OPTIONS options;
	boinc_options_defaults(options);
	options.multi_thread = true;  // for multiple threads in OpenCL
	options.multi_process = true; // for multiple processes in OpenCL?
	options.normal_thread_priority = true; // so GPUs will run at full speed
	boinc_init_options(&options);
	#endif

	inPath = argv[2];

	time_t starttime, endtime;
	int device = -1;

	if(argc > 3)
	{
		for(int i = 3; i < argc; i++)
		{
			string arg(argv[i]);
			if(arg.find("stride=") != string::npos)
			{
				stride = stoi(arg.substr(arg.find("=")+1));
			}
			else if(arg.find("gpu=") != string::npos)
			{
				if(arg.find("false") != string::npos || arg.find("False") != string::npos)
				{
					__useGPU = false;
				}
			}
			else if(arg.find("device=") != string::npos)
			{
				device = stoi(arg.substr(arg.find("=")+1));
			}
		}
	}

	string resolved_CNN_name = getBoincFilename(string(argv[1]));

	//set up net
	Net net(resolved_CNN_name);
	net.setDevice(device); //should it exit if this fails?
	if(!net.finalize()) // failure due to unable to finalize CNN
	{
		boinc_finish(-1); 
		return -1;
	}

	//go through all images in the folder
	string resolved_Video_name = getBoincFilename(string(argv[2]));
	
	#ifdef _BOINC_APP_	
	if(readCheckpoint())
		printf("Continuing from Checkpoint\n");
	else
		printf("No Checkpoint found. Starting from beginning\n");
	#endif

	unsigned int startFrame = __frameNum;

	starttime = time(NULL);
	breakUpVideo(resolved_Video_name.c_str(),net,startFrame);
	endtime = time(NULL);
	cout << "Time for video " << filenames[i] << ": " << secondsToString(endtime - starttime) << endl;

	boinc_finish(0);
	return 0;
	
}