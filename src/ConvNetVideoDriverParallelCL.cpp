/**
 * TODO: Convert to C++ standard <thread> then refactor.
 */
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <vector>
#include "ConvNetCL.h"
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
	long frameNum = -1;
	Mat* mat;
	size_t red = 0;
	size_t flyingElement;
	double percentDone;
};

vector<Frame*> waitingFrames(0);

Net* masternet;

char *inPath, *outPath;
int stride = 1;
int jump = 1;
vector<int> excludes;
bool firstGot = false;

imVector *fullImages;

VideoCapture __video;
VideoWriter __outVideo;
ofstream __outcsv;
char* __netName;

int inputWidth, inputHeight;
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
//void getNextFrame(Mat& frame, unsigned int& frameNum, double& percentDone)
void getNextFrame(Frame& frame)
{
	pthread_mutex_lock(&frameMutex);
	//bool val = __video.read(frame);
	if(firstGot)
	{
		bool val1 = true;
		for(int i=0; i < jump; i++)
			if(!__video.grab())
				val1 = false;
		bool val2 = __video.retrieve(*(frame.mat));
		frame.percentDone = __video.get(CV_CAP_PROP_POS_AVI_RATIO) * 100.0;
		frame.frameNum = curFrame;
		curFrame += jump;
		// printf("got frame %d\n",curFrame);
		if(!val1 || !val2) //this means the last frame was grabbed.
		{
			done = true;
		}
	}
	else // first frame only
	{
		bool val = __video.read(*(frame.mat));
		frame.percentDone = __video.get(CV_CAP_PROP_POS_AVI_RATIO) * 100.0;
		frame.frameNum = curFrame;
		curFrame+=jump;
		if(!val)
			done = true;
		firstGot = true;
		// printf("got first frame %d\n",curFrame-jump);
	}
	pthread_mutex_unlock(&frameMutex);
}

//void submitFrame(Mat* frame, unsigned int frameNum, int red, double percentDone)//, int device)
void submitFrame(Frame& frame)
{
	pthread_mutex_lock(&submitMutex);
	//printf("frame %d submitted by thread %d\n", frameNum, device);
	if(frame.frameNum == curSubmittedFrame)
	{
		//printf("sub if\n");
		__outVideo << *(frame.mat);
		// __outcsv << frame.red << "," << (frame.frameNum/10.0) << "\n";
		__outcsv << frame.frameNum/10.0 << ',' << frame.red << ',' << frame.flyingElement << '\n';
		curSubmittedFrame+=jump;
		delete frame.mat;
		printf("Frame %ld completed. %.2lf%% complete.\n",frame.frameNum,frame.percentDone);
		int i=0; 
		//printf("starting while\n");
		while(i < waitingFrames.size() && waitingFrames[i]->frameNum == curSubmittedFrame)
		{
			//printf("in while\n");
			__outVideo << (*(waitingFrames[i]->mat));
			// __outcsv << waitingFrames[i]->red << "," << (waitingFrames[i]->frameNum/10.0) << "\n";
			__outcsv << (waitingFrames[i]->frameNum/10.0) << ',' << waitingFrames[i]->red << ',' << waitingFrames[i]->flyingElement << '\n';
			printf("Frame %ld completed. %.2lf%% complete.\n",waitingFrames[i]->frameNum,waitingFrames[i]->percentDone);
			delete waitingFrames[i]->mat;
			delete waitingFrames[i];

			curSubmittedFrame+=jump;
			i++;
		}
		__outcsv.flush();
		// printf("after while\n");
		if(i != 0) //if we took any away from the waitingFrames
		{
			for(int j=i; j < waitingFrames.size(); j++)
			{
				waitingFrames[j-i] = waitingFrames[j];
			}
			waitingFrames.resize(waitingFrames.size() - i);
		}
		// printf("end sub if\n");
	}
	else
	{
		// printf("Frame %d is waiting\n", frameNum);
		//printf("sub else\n");
		Frame *newframe = new Frame;
		newframe->mat = frame.mat; 
		newframe->frameNum = frame.frameNum;
		newframe->red = frame.red;
		newframe->flyingElement = frame.flyingElement;
		newframe->percentDone = frame.percentDone;
		waitingFrames.resize(waitingFrames.size() + 1);
		waitingFrames.back() = nullptr;

		int i=0;
		//sorted insertion into list
		if(waitingFrames[0] != nullptr) //make sure not first element in list
		{
			while(i < waitingFrames.size() -1 && waitingFrames[i]->frameNum < frame.frameNum)
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


/*
 * The inner for loop gets the confidences for each pixel in the image. If a pixel is in more than one subimage
 * (i.e. the stride is less than the subimage size), then the confidences from each subimage is added.
 */
//void breakUpImage(Mat& image, Net& net, Mat& outputMat, int& inred)
void breakUpImage(Frame& frame, Net& net, int device)
{
	// int numrows = image.rows;
	// int numcols = image.cols;
	int numrows = frame.mat->rows;
	int numcols = frame.mat->cols;

	//printf("%d %d\n",numrows, numcols);

	// vector<vector< vector<double> > > fullImage; //2 dims for width and height, last dim for each possible category
	// resize3DVector(fullImage,numrows,numcols,net.getNumClasses());
	setAll3DVector(fullImages[device],0);
	// vector<imVector> imageRow(0); // this will hold all subimages from one row
	vector<Mat> imageRow(0);
	vector<int> calcedClasses(0);
	vector<vector<double> > confidences(0);//for the confidence for each category for each image
		//the outer vector is the image, the inner vector is the category, the double is output(confidence) of the softmax

	int numrowsm32 = numrows-inputHeight;
	int numcolsm32 = numcols-inputWidth;

	//loop where the cnn is run over all subimages
	for(int i=0; i <= numrowsm32; i+=stride) 
	{
		imageRow.resize(0);

		//get all subimages from a row
		for(int j=6; j<= numcolsm32; j+=stride) //NOTE: each j is a different subimage //j starts at 6 b/c black line on left
		{
			imageRow.push_back((*(frame.mat))(Range(i,i+inputHeight),Range(j,j+inputWidth)));

			// const Mat out = (*(frame.mat))(Range(i,i+inputHeight),Range(j,j+inputWidth));
			// imageRow.push_back(out);

			// imageRow.resize(imageRow.size()+1);
			// convertColorMatToVector(out,imageRow.back());
		}
		//cout << imageRow[0] << endl;
		//set them as the data in the net
		//preprocess(imageRow);
		net.setData(imageRow);
		net.run();
		net.getConfidences(confidences); //gets the confidence for each category for each image

		int curImage = 0;
		for(int j=6; j<= numcolsm32; j+=stride) //NOTE: each iteration of this loop is a different subimage
		{
			for(int ii=i; ii < i+inputHeight && ii < numrows; ii++)
			{
				for(int jj=j; jj < j+inputWidth && jj < numcols; jj++)
				{
					if(ii > 417 && jj > 525)
						break;
					for(int cat = 0; cat < confidences[curImage].size(); cat++)
					{
						//printf("%d %d %d %d\n",i,j,jj,cat);
						fullImages[device][ii][jj][cat] += confidences[curImage][cat];
					}
				}
			}
			curImage++;
		}
	}
	//printf("starting red element\n");
	squareElements(fullImages[device]);

	Mat* outputMat = new Mat(numrows, numcols, CV_8UC3);

	int numClasses = fullImages[device][0][0].size();

	//calculate what output image should look like and csv file should be
	size_t redElement = 0;
	size_t flyingElement = 0;
	for(int i=0; i < numrows; i++)
	{
		for(int j=0; j < numcols; j++)
		{
			double sumsq = vectorSum(fullImages[device][i][j]);
			for(int n=0; n < fullImages[device][i][j].size(); n++)
			{
				//fullImage[i][j][n] = fullImage[i][j][n] * fullImage[i][j][n] / sumsq;
				fullImages[device][i][j][n] /= sumsq;
			}

			//write the pixel
			Vec3b& outPix = outputMat->at<Vec3b>(i,j);
			if(allElementsEquals(fullImages[device][i][j]))
			{
				outPix[0] = 255; outPix[1] = 255; outPix[2] = 255; // white
			}
			else
			{
				double noBird = 255 * fullImages[device][i][j][0];
				double onGround = 255 * fullImages[device][i][j][1];
				double flying;


				if(numClasses > 2)
					flying = 255 * fullImages[device][i][j][2];
				else
					flying = 0;
				// printf("%lf %lf %lf\n", fullImages[device][i][j][0], fullImages[device][i][j][2], fullImages[device][i][j][1]);

				outPix[0] = noBird; // blue
				// outputMat->at<unsigned char>(i,j,0) = (unsigned char)noBird;
				outPix[1] = flying;	  //green
				outPix[2] = onGround;  // red
				if(onGround > 150) //red > 50 || red > blue
					redElement += (size_t)onGround;
				if(flying > 150)
					flyingElement += (size_t)flying;

			}
		}
	}
	//replace original image with prediction
	delete frame.mat;
	frame.mat = outputMat;

	//put in sums
	frame.red = redElement;
	frame.flyingElement = flyingElement;
	// printf("flyingElement: %lu\n", flyingElement);
}

void __parallelVideoProcessor(int device)
{
	if(device == 0)
		printf("Loading Nets. This could take a while for larger nets.\n");

	for(int i = 0; i < excludes.size(); i++)
	{
		if(excludes[i] == device)
			return;
	}

	// Net net(__netName); //0
	Net net = *masternet; //1
	// Net net;
	// net = *masternet;

	printf("Device %d loc %p\n",device, (void *)&net);

	// net.setConstantMem(true);
	if(!net.setDevice(device) || !net.finalize())
		return;
	printf("Thread using device %d\n",device);

	resize3DVector(fullImages[device], __rows, __cols, net.getNumClasses());

	Frame frame;
	frame.mat = new Mat(__rows, __cols, CV_8UC3);
	getNextFrame(frame);

	while(!done)
	{		
		breakUpImage(frame, net, device);
		submitFrame(frame);
		frame.mat = new Mat(__rows, __cols, CV_8UC3);
		getNextFrame(frame);
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

	if(__video.get(CV_CAP_PROP_FRAME_WIDTH) < inputWidth || __video.get(CV_CAP_PROP_FRAME_HEIGHT) < inputHeight)
	{
		printf("The video %s is too small in at least one dimension. Minimum size is %dx%d.\n",videoName,inputWidth,inputHeight);
		return;
	}

	//Get the names for the output files.
	char outName[255], outNameCSV[255];
	string origName(videoName);
	size_t dot = origName.rfind('.');
	const char *noExtension = origName.substr(0,dot).c_str();
	sprintf(outName,"%s_prediction%s",noExtension,".avi");
	sprintf(outNameCSV,"%s_prediction.csv",noExtension);

	//Open csv
	__outcsv.open(outNameCSV);

	//adjust FPS to account for jump
	int fps = 10;
	if(jump <= 10)
		fps /= jump;
	else
		fps = 1;

	//open video and get width and height
	__outVideo.open(outName, 
	 CV_FOURCC('m','p','4','v'),//CV_FOURCC('M', 'J', 'P', 'G'),//-1,//video.get(CV_CAP_PROP_FOURCC),
	 fps,//video.get(CV_CAP_PROP_FPS), 
	 Size(__video.get(CV_CAP_PROP_FRAME_WIDTH), __video.get(CV_CAP_PROP_FRAME_HEIGHT)));
	__rows = __video.get(CV_CAP_PROP_FRAME_HEIGHT);
	__cols = __video.get(CV_CAP_PROP_FRAME_WIDTH);

	//get all the devices and make threads.
	int numDevices = getNumDevices();
	thread* t = new thread[numDevices];
	fullImages = new imVector[numDevices];
	for(int i=0; i < numDevices; i++)
	{
		//start new thread for each device. Any thread for a device that does not support double or is excluded will return early.
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
	if(argc < 3)
	{
		printf("Usage (Required to come first):\n ./ConvNetVideoDriverParallelCL cnnConfig.txt VideoOrFolderPath\n");
		printf("Optional args (must come after required args. Case sensitive.):\n");
		printf("   stride=<int>        Stride across image. Defaults to 1.\n");
		printf("   jump=<int>          How many frames to jump between computations. If jump=10,\n");
		printf("                       it will calc on frames 0, 10, 20, etc. Defaults to 1.\n");
		printf("   ex=<int>            Device to exclude from running. Can be used multiple times.\n");
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
				stride = stoi(arg.substr(arg.find("=")+1));
			else if(arg.find("jump=") != string::npos)
				jump = stoi(arg.substr(arg.find("=")+1));
			else if(arg.find("ex=") != string::npos)
				excludes.push_back(stoi(arg.substr(arg.find('=')+1)));
			else
			{
				printf("Unknown arg \"%s\". Aborting.\n", argv[i]);
				return 0;
			}
		}
	}

	// printf("jump %d stride %d\n", jump, stride);

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
			isDirectory = true;
		else if (s.st_mode & S_IFREG) // file
			isDirectory = false;
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
				pathName.append(1,'/');
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

	printf("Loading Net. This could take a while for larger nets.\n");
	masternet = new Net(__netName);
	inputWidth = masternet->getInputWidth();
	inputHeight = masternet->getInputHeight();

	masternet->printLayerDims();

	for(int i=0; i < filenames.size(); i++)
	{
		starttime = time(NULL);
		curFrame=0;
		breakUpVideo(filenames[i].c_str());
		endtime = time(NULL);
		cout << "Time for video " << filenames[i] << ": " << secondsToString(endtime - starttime) << endl;
	}
	delete masternet;

	//cout << "returning" << endl;
	return 0;
	
}