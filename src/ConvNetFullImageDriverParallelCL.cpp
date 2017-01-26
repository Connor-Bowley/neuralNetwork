/**
 * TODO: Convert to <thread> from <pthread.h> and then refactor
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
int curRow = 0;
int curSubmittedFrame = 0;

typedef vector<vector<vector<double> > > imVector;

char *inPath;
int stride = 1;

Mat fullMat;

int numrowsmin, numcolsmin;
int numClasses;

vector<imVector> fullImages;
vector<Net*> nets;
vector<bool> deviceActive;

char* __netName;

vector<Net::ClassInfo> infos;

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

int getNextRow()
{
	pthread_mutex_lock(&frameMutex);
	int out;
	if(curRow < fullMat.rows - inputHeight)
	{
		out = curRow;
		curRow += stride;
		printf("Giving row %d of %d (%d)\n", out,__rows - inputHeight, __rows);
	}
	else
	{
		done = true;
		out = -1;
	}
	pthread_mutex_unlock(&frameMutex);
	return out;
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

void combineImages() // can we do this threaded too?
{
	for(int d = 1; d < fullImages.size(); d++)
	{
		if(!deviceActive[d])
			continue;
		for(int i = 0; i < fullImages[d].size(); i++)
		{
			for(int j = 0; j < fullImages[d][i].size(); j++)
			{
				for(int k = 0; k < fullImages[d][i][j].size(); k++)
				{
					fullImages[0][i][j][k] += fullImages[d][i][j][k];
				}
			}
		}
	}
}

void breakUpRow(int row, int device)
{
	int i = row;
	vector<Mat> imageRow(0);
	vector<int> calcedClasses(0);
	vector<vector<double> > confidences(0);//for the confidence for each category for each image
		//the outer vector is the image, the inner vector is the category, the double is output(confidence) of the softmax

	//get all subimages from a row
	for(int j=0; j<= numcolsmin; j+=stride) //NOTE: each j is a different subimage //j starts at 6 b/c black line on left
	{
		imageRow.push_back((fullMat)(Range(i,i+inputHeight),Range(j,j+inputWidth)));
	}
	//set them as the data in the net
	nets[device]->setData(imageRow);
	nets[device]->run();
	nets[device]->getConfidences(confidences); //gets the confidence for each category for each image

	int curImage = 0;
	for(int j=0; j<= numcolsmin; j+=stride) //NOTE: each iteration of this loop is a different subimage
	{
		for(int ii=i; ii < i+inputHeight && ii < __rows; ii++)
		{
			for(int jj=j; jj < j+inputWidth && jj < __cols; jj++)
			{
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

void __parallelImageRowProcessor(int device)
{
	if(!deviceActive[device])
		return;

	int row = getNextRow();

	while(!done)
	{	
		if(row == -1)
			break;
		breakUpRow(row, device);
		row = getNextRow();
	}
}

string getNameForVal(int trueVal)
{
	for(int i = 0; i < infos.size(); i++)
	{
		if(infos[i].trueVal == trueVal)
			return infos[i].name;
	}
	char buf[100];
	sprintf(buf,"class%d",trueVal);
	return string(buf);
}


/*
 * The inner for loop gets the confidences for each pixel in the image. If a pixel is in more than one subimage
 * (i.e. the stride is less than the subimage size), then the confidences from each subimage is added.
 */
//void breakUpImage(Mat& image, Net& net, Mat& outputMat, int& inred)
void breakUpImage(const char* imageName)
{ 
	//reset stuff
	done = false;
	curRow = 0;

	fullMat = imread(imageName,1);

	__rows = fullMat.rows;
	__cols = fullMat.cols;

	for(int i = 0; i < nets.size(); i++)
	{
		resize3DVector(fullImages[i],__rows,__cols,numClasses);
		setAll3DVector(fullImages[i],0);
	}

	numrowsmin = __rows-inputHeight;
	numcolsmin = __cols-inputWidth;

	//calculate the rows in parallel.
	thread* t = new thread[nets.size()];
	for(int i = 0; i < nets.size(); i++)
		t[i] = thread(__parallelImageRowProcessor, i);

	for(int i = 0; i < nets.size(); i++)
		t[i].join();

	combineImages(); // combines into fullImages[0]

	// int numClasses = fullImages[0][0][0].size(); //fullImages[device][row][col]

	//process the data
	double sumsq;
	squareElements(fullImages[0]);
	for(int i=0; i < __rows; i++)
	{
		for(int j=0; j < __cols; j++)
		{
			sumsq = vectorSum(fullImages[0][i][j]);
			for (int k = 0; k < numClasses; k++)
			{
				fullImages[0][i][j][k] /= sumsq;
			}
		}
	}

	//make the output Mats
	vector<Mat*> outputMats(numClasses);
	for(int k = 0; k < numClasses; k++)
	{
		outputMats[k] = new Mat(__rows,__cols,CV_8UC3);
	}
	//calculate what output images should look like
	for(int k = 0; k < numClasses; k++)
	{
		for(int i=0; i < __rows; i++)
		{
			for(int j=0; j < __cols; j++)
			{
				//write the pixel
				Vec3b& outPix = outputMats[k]->at<Vec3b>(i,j);
				if(allElementsEquals(fullImages[0][i][j]))
				{
					outPix[0] = 0; outPix[1] = 255; outPix[2] = 0; // green
				}
				else
				{
					double pix = 255 * fullImages[0][i][j][k];
					outPix[0] = pix;  // blue
					outPix[1] = pix;  //green
					outPix[2] = pix;  // red
				}
			}
		}
	}

	//output the mats
	char outName[500];
	string origName(imageName);
	size_t dot = origName.rfind('.');
	const string noExtension = origName.substr(0,dot);
	const string extension = origName.substr(dot+1);

	for(int k = 0; k < numClasses; k++)
	{
		sprintf(outName,"%s_prediction_%s.%s",noExtension.c_str(), getNameForVal(k).c_str(), extension.c_str());
		// if(infos.size() > k)
		// 	sprintf(outName,"%s_prediction_%s.%s",noExtension.c_str(), getNameForVal(k).c_str(), extension.c_str());
		// else
		// 	sprintf(outName,"%s_prediction_class%d.%s",noExtension.c_str(),k,extension.c_str());
		printf("Writing %s\n", outName);
		imwrite(outName,*(outputMats[k]));
	}

	//cleanup memory
	for(int m = 0; m < numClasses; m++)
		delete outputMats[m];
}

int checkExtensions(char* filename)
{
	string name = filename;
	if(name.rfind(".jpg")  == name.length() - 4) return 1;
	if(name.rfind(".jpeg") == name.length() - 5) return 1;
	if(name.rfind(".png")  == name.length() - 4) return 1;
	if(name.rfind(".JPG")  == name.length() - 4) return 1;
	if(name.rfind(".JPEG") == name.length() - 5) return 1;
	if(name.rfind(".PNG")  == name.length() - 4) return 1;
	return 0;
}

int main(int argc, char** argv)
{
	if(argc < 3 || 5 < argc)
	{
		printf("Usage (Required to come first):\n ./ConvNetFullImageDriverParallelCL cnnFile.txt ImageOrFolderPath\n");
		printf("Optional args (must come after required args. Case sensitive.):\n");
		printf("   stride=<int>        Stride across image. Defaults to 1.\n");
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

	printf("Found %lu image(s).\n", filenames.size());

	//init all nets
	for(int i = 0; i < getNumDevices(); i++)
	{
		// printf("%d\n", i);
		nets.push_back(new Net(__netName));	
	}
	fullImages.resize(nets.size());
	deviceActive.resize(nets.size());

	inputHeight = nets[0]->getInputHeight();
	inputWidth = nets[0]->getInputWidth();
	numClasses = nets[0]->getNumClasses();

	nets[0]->getClassNames(infos);

	printf("Getting devices\n");
	//get the ones that work
	for(int i = 0 ; i < nets.size(); i++)
	{
		// nets[i]->setConstantMem(true);
		if(nets[i]->setDevice(i) && nets[i]->finalize())
		{
			deviceActive[i] = true;
			printf("Thread using device %d\n",i);
		}
		else
		{
			deviceActive[i] = false;
			delete nets[i];
		}
	}

	for(int i=0; i < filenames.size(); i++)
	{
		starttime = time(NULL);
		breakUpImage(filenames[i].c_str());
		endtime = time(NULL);
		cout << "Time for image: " << filenames[i] << ": " << secondsToString(endtime - starttime) << endl;
	}

	//cout << "returning" << endl;
	return 0;
	
}