
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <vector>
#include "ConvNet.h"
#include <ctype.h>
#include <fstream>
#include <time.h>
#include <thread>
#include <pthread.h>
#ifdef __APPLE__
 	#include "OpenCL/opencl.h"
#else
 	#include "CL/cl.h"
#endif

#ifdef WIN32
# include <io.h>
#else
# include <unistd.h>
#endif

using namespace cv;
using namespace std;

pthread_mutex_t frameMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t submitMutex = PTHREAD_MUTEX_INITIALIZER;

int curFrame = 0;
int curSubmittedFrame = 0;

typedef vector<vector<vector<double> > > imVector;

struct Frame
{
	int frameNum = -1;
	Mat* mat;
	int redElement = 0;
};

vector<Frame*> waitingFrames(0);

char *inPath, *outPath;
int stride = 1;
bool __useGPU = true;

VideoCapture __video;
VideoWriter __outVideo;
ofstream __outcsv;
char* __netName;

int __rows, __cols;

bool done = false;

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

//puts frame and frameNum in parameters
void getNextFrame(Mat& frame, unsigned int& frameNum)
{
	pthread_mutex_lock(&frameMutex);
	bool val = __video.read(frame);
	frameNum = curFrame++;
	//printf("got frame %d\n",curFrame-1);
	if(val == false) //this means the last frame was grabbed.
		done = true;
	pthread_mutex_unlock(&frameMutex);
}

void submitFrame(Mat* frame, unsigned int frameNum, int red)//, int device)
{
	pthread_mutex_lock(&submitMutex);
	//printf("frame %d submitted by thread %d\n", frameNum, device);
	if(frameNum == curSubmittedFrame)
	{
		//printf("sub if\n");
		__outVideo << *frame;
		__outcsv << red << "," << (frameNum/10.0) << "\n";
		curSubmittedFrame++;
		delete frame;
		printf("Frame %d completed.\n",frameNum);
		int i=0; 
		//printf("starting while\n");
		while(i < waitingFrames.size() && waitingFrames[i]->frameNum == curSubmittedFrame)
		{
			//printf("in while\n");
			__outVideo << (*(waitingFrames[i]->mat));
			__outcsv << waitingFrames[i]->redElement << "," << (waitingFrames[i]->frameNum/10.0) << "\n";
			//printf("pushed\n");
			//printf("++\n");
			printf("Frame %d completed.\n",waitingFrames[i]->frameNum);
			delete waitingFrames[i]->mat;
			delete waitingFrames[i];

			curSubmittedFrame++;
			i++;
		}
		//printf("after while\n");
		if(i != 0) //if we took any away from the waitingFrames
		{
			for(int j=i; j < waitingFrames.size(); j++)
			{
				waitingFrames[j-i] = waitingFrames[j];
			}
			waitingFrames.resize(waitingFrames.size() - i);
		}
		//printf("end sub if\n");
	}
	else
	{
		//printf("sub else\n");
		Frame *newframe = new Frame;
		newframe->mat = frame; 
		newframe->frameNum = frameNum;
		newframe->redElement = red;
		waitingFrames.resize(waitingFrames.size() + 1);
		waitingFrames.back() = nullptr;

		int i=0;
		//sorted insertion into list
		if(waitingFrames[0] != nullptr) //make sure not first element in list
		{
			while(i < waitingFrames.size() -1 && waitingFrames[i]->frameNum < frameNum)
				i++;
			for(int j=waitingFrames.size()-1; j >= i+1; j--)
			{
				waitingFrames[j] = waitingFrames[j-1];
			}
		}
		waitingFrames[i] = newframe;
		//printf("frame %d is first frame waiting\n",waitingFrames[0]->frameNum);
		//printf("end sub else\n");
	}
	pthread_mutex_unlock(&submitMutex);
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

int getNumDevices()
{
	//get num of platforms and we will use the first one
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);
	vector<cl_platform_id> platformIds(platformIdCount);
	clGetPlatformIDs(platformIdCount,platformIds.data(), nullptr);
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platformIds[0],CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

	return (int)deviceIdCount;
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
void breakUpImage(Mat& image, Net& net, Mat& outputMat, int& inred)
{
	int numrows = image.rows;
	int numcols = image.cols;

	//printf("%d %d\n",numrows, numcols);

	vector<vector< vector<double> > > fullImage; //2 dims for width and height, last dim for each possible category
	resize3DVector(fullImage,numrows,numcols,net.getNumCategories());
	setAll3DVector(fullImage,0);
	vector<imVector> imageRow(0); // this will hold all subimages from one row
	vector<int> calcedClasses(0);
	vector<vector<double> > confidences(0);//for the confidence for each category for each image
		//the outer vector is the image, the inner vector is the category, the double is output(confidence) of the softmax

	int numrowsm32 = numrows-32;
	int numcolsm32 = numcols-32;

	for(int i=0; i <= numrowsm32; i+=stride)
	{
		imageRow.resize(0);

		//get all subimages from a row
		for(int j=0; j<= numcolsm32; j+=stride) //NOTE: each j is a different subimage
		{
			const Mat out = image(Range(i,i+32),Range(j,j+32));
			imageRow.resize(imageRow.size()+1);
			convertColorMatToVector(out,imageRow.back());
		}
		//set them as the data in the net
		preprocess(imageRow);
		net.setData(imageRow);
		net.newRun(calcedClasses, __useGPU);
		net.getConfidences(confidences); //gets the confidence for each category for each image

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
	//printf("starting red element\n");
	squareElements(fullImage);

	int redElement = 0;
	for(int i=0; i < numrows; i++)
	{
		for(int j=0; j < numcols; j++)
		{
			//printf("pixel %d %d - %d %d\n",i,j, outputMat.rows, outputMat.cols );
			double sumsq = vectorSum(fullImage[i][j]);
			for(int n=0; n < fullImage[i][j].size(); n++)
			{
				//fullImage[i][j][n] = fullImage[i][j][n] * fullImage[i][j][n] / sumsq;
				fullImage[i][j][n] /= sumsq;
			}

			//write the pixel
			Vec3b& outPix = outputMat.at<Vec3b>(i,j);
			//int maxEle = getMaxElementIndex(fullImage[i][j]);
			//printf("writing\n");
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

			}
		}
	}
	inred = redElement;
}

void __parallelVideoProcessor(int device)
{
	Net net(__netName);
	if(!net.isActive() || !net.setDevice(device))
		return;
	printf("Thread using device %d\n",device);

	unsigned int frameNum=0;
	Mat frame;
	int red;
	getNextFrame(frame, frameNum);
	while(!done)
	{
		//printf("thread %d got frame %d\n", device, frameNum);
		Mat* outFrame = new Mat(__rows,__cols,CV_8UC3);
		//printf("thread %d: starting breakUpImage %d\n",device,frameNum);
		breakUpImage(frame, net, *outFrame, red);
		//printf("thread %d: submitting frame %d\n",device,frameNum);
		submitFrame(outFrame, frameNum, red);//, device);

		getNextFrame(frame,frameNum);
	}
}

void breakUpVideo(const char* videoName)
{
	__video.open(videoName);
	if(!__video.isOpened())
	{
		cout << "Could not open video: " << videoName << endl;
		return;
	}

	if(__video.get(CV_CAP_PROP_FRAME_WIDTH) < 32 || __video.get(CV_CAP_PROP_FRAME_HEIGHT) < 32)
	{
		printf("The video %s is too small in at least one dimension. Minimum size is 32x32.\n",videoName);
		return;
	}



	char outName[255], outNameCSV[255];
	string origName(videoName);
	size_t dot = origName.rfind('.');
	const char *noExtension = origName.substr(0,dot).c_str();

	sprintf(outName,"%s_prediction%s",noExtension,".avi");
	sprintf(outNameCSV,"%s_prediction.csv",noExtension);
	//cout << "writing " << outName << endl;

	__outcsv.open(outNameCSV);

	__outVideo.open(outName, 
	 CV_FOURCC('M', 'J', 'P', 'G'),//-1,//video.get(CV_CAP_PROP_FOURCC),
	 10,//video.get(CV_CAP_PROP_FPS), 
	 Size(__video.get(CV_CAP_PROP_FRAME_WIDTH), __video.get(CV_CAP_PROP_FRAME_HEIGHT)));

	__rows = __video.get(CV_CAP_PROP_FRAME_HEIGHT);
	__cols = __video.get(CV_CAP_PROP_FRAME_WIDTH);
	int numDevices = getNumDevices();
	thread* t = new thread[numDevices];
	for(int i=0; i < numDevices; i++)
	{
		//start new thread for each device. Any thread for a device that does not support double will return early.
		t[i] = thread(__parallelVideoProcessor, i);
	}

	for(int i=0; i < numDevices; i++)
	{
		t[i].join();
	}

	__outcsv.close();
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
		printf("use format: ./ConvNetVideoDriver cnnConfig.txt VideoOrFolderPath (stride=1) (gpu=true)\n");
		return -1;
	}
	time_t starttime, endtime;
	inPath = argv[2];

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
		}
	}

	__netName = argv[1];
	//set up net
	//Net net(argv[1]);
	//if(!net.isActive())
		//return 0;



	//go through all images in the folder
	bool isDirectory;
	struct stat s;
	if(stat(inPath,&s) == 0)
	{
		if(s.st_mode & S_IFDIR) // directory
		{
			isDirectory = true;
		}
		else if (s.st_mode & S_IFREG) // file
		{
			isDirectory = false;
		}
		else
		{
			printf("We're not sure what the file you inputted was.\nExiting\n");
			return -1;
		}
	}
	else
	{
		printf("Error getting status of folder or file.\nExiting\n");
		return -1;
	}

	vector<string> filenames(0);
	
	if(isDirectory)
	{
		DIR *directory;
		struct dirent *file;
		if((directory = opendir(inPath)))// != NULL)
		{
			string pathName = inPath;
			if(pathName.rfind("/") != pathName.length()-1)
			{
				pathName.append(1,'/');
			}
			char inPathandName[250];
			while((file = readdir(directory)))// != NULL)
			{
				if(strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") != 0)
				{
					if(checkExtensions(file->d_name))
					{
						sprintf(inPathandName,"%s%s",pathName.c_str(),file->d_name);
						string ipan(inPathandName);
						if(ipan.find("_prediction") == string::npos)
						{
							//breakUpImage(inPathandName, net);
							filenames.push_back(ipan);
						}
					}
				}
			}
			//cout << "closing directory" << endl;
			closedir(directory);
			//cout << "directory closed" << endl;
		}
	}
	else
	{
		if(checkExtensions(inPath))
		{
			//breakUpImage(inPath, net);
			string ip(inPath);
			filenames.push_back(ip);
		}
	}

	for(int i=0; i < filenames.size(); i++)
	{
		starttime = time(NULL);
		curFrame=0;
		breakUpVideo(filenames[i].c_str());
		endtime = time(NULL);
		cout << "Time for video " << filenames[i] << ": " << secondsToString(endtime - starttime) << endl;
	}

	//cout << "returning" << endl;
	return 0;
	
}