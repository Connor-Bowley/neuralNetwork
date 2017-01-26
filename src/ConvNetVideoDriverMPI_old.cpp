#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include "ConvNetCL.h"
#include <ctype.h>
#include <fstream>
#include <time.h>
#include <thread>
#include <pthread.h>
#include <mpi.h>
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

#define FRAME_BLOCK_SIZE 10

#define GET_FRAMENUM 0
#define SEND_FRAMENUM 1
#define SUBMIT_FRAME 2

pthread_mutex_t frameMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t globalFrameMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t submitMutex = PTHREAD_MUTEX_INITIALIZER;

unsigned int curFrame = 0;
unsigned int myCurFrame = 0;
unsigned int myReadFrames = 0;
unsigned int myTakenFrames = 0;
unsigned int myFrameAmount = 0;
unsigned int curSubmittedFrame = 0;
unsigned int lastFrameNum = 0;


thread *getThread, *submitThread;
bool stopThread = false;

int my_rank; // process rank
int comm_sz; // num processes

typedef vector<vector<vector<double> > > imVector;

struct Frame
{
	int frameNum = -1;
	Mat* mat;
	int red = 0;
};

std::vector<Frame*> doneFrames(0);


vector<Frame*> waitingFrames(0);

char *inPath, *outPath;
int stride = 1;
int jump = 1;
int blockSize = FRAME_BLOCK_SIZE;

VideoCapture __globalVideo;
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

//used only by thread 0
unsigned int getNextFrameGlobal(unsigned int& frameNum)
{
	pthread_mutex_lock(&globalFrameMutex);
	Mat frame;
	unsigned int amount = 0; //amount of frames validated
	double damount = 0;
	for(int i = 0; i < blockSize; i++)
	{
		bool val = __globalVideo.read(frame); // different copy of video so no conflict with process 0
		frameNum = curFrame++;
		if(val == false) //this means the last frame was grabbed.
		{
			done = true;
			break;
		}
		//amount++;
		damount += 1/jump;
	}
	amount = (unsigned int)damount;
	pthread_mutex_unlock(&globalFrameMutex);
	return amount;
}

void getFirstFrameInfo()
{
	myTakenFrames = 0;
	//talk to thread running getThreadMPI
	if(my_rank != 0)
	{
		unsigned int buf[2];
		MPI_Send(&my_rank, 1, MPI_INT, 0, GET_FRAMENUM, MPI_COMM_WORLD);
		MPI_Recv(buf, 2, MPI_UNSIGNED, 0, SEND_FRAMENUM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		myCurFrame = buf[0];
		myFrameAmount = buf[1];
	}
	else
	{
		myFrameAmount = getNextFrameGlobal(myCurFrame);
	}
}

//returns next frame num. used by all
bool getNextFrame(Mat& frame, unsigned int& frameNum)
{
	// printf("getting next frame %d\n", my_rank);
	pthread_mutex_lock(&frameMutex);
	// printf("next frame in mutex %d\n", my_rank);
	while(myReadFrames != myCurFrame) // the my video will need to go past the frames others have done.
	{
		bool val = __video.read(frame);
		if(val == false) // this should never happend
			break;
		myReadFrames++;
	}
	myCurFrame += jump;
	myTakenFrames++;
	if(myTakenFrames == myFrameAmount)
	{
		myTakenFrames = 0;
		//talk to thread running getThreadMPI
		if(my_rank != 0)
		{
			unsigned int buf[2];
			MPI_Send(&my_rank, 1, MPI_INT, 0, GET_FRAMENUM, MPI_COMM_WORLD);
			MPI_Recv(buf, 2, MPI_UNSIGNED, 0, SEND_FRAMENUM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			myCurFrame = buf[0];
			myFrameAmount = buf[1];
		}
		else
		{
			myFrameAmount = getNextFrameGlobal(myCurFrame);
		}
	}
	//printf("got frame %d\n",curFrame-1);
	pthread_mutex_unlock(&frameMutex);
	// printf("past mutex\n");
	if(myFrameAmount > 0)
		return true;
	else
		return false;
}


void getThreadMPI() // thread opened by 0 to allocate frames to processes
{
	int rank;
	unsigned int frameNum[2];
	blockSize = FRAME_BLOCK_SIZE * jump;
	while(!done)
	{
		//get next mpi msg from any source. They send their rank over in the message
		MPI_Recv(&rank, 1, MPI_INT, MPI_ANY_SOURCE, GET_FRAMENUM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		//push to getNextFrame which has a mutex
		frameNum[1] = getNextFrameGlobal(frameNum[0]);

		//send mpi msg back saying [start frame, amount of frames]
		MPI_Send(frameNum, 2, MPI_UNSIGNED, rank, SEND_FRAMENUM, MPI_COMM_WORLD);
	}
}

void submitFrames()
{
	//submit all stored frames to 0
}

void storeFrame(Frame& frame)
{
	Frame* store = new Frame;
	store->mat = frame.mat;
	store->red = frame.red;
	store->frameNum = frame.frameNum;

	doneFrames.push_back(store);
}

void submitFrame(unsigned int frameNum, unsigned int red)
{
	unsigned int buf[2];
	buf[0] = frameNum;
	buf[1] = red;
	MPI_Send(buf, 2, MPI_UNSIGNED, 0, SEND_FRAMENUM, MPI_COMM_WORLD);
}

//returns whether or not all frames have been submitted
bool submitFrameGlobal(unsigned int frameNum, unsigned int red)//, int device)
{
	bool returnVal = false;
	pthread_mutex_lock(&submitMutex);
	//printf("frame %d submitted by thread %d\n", frameNum, device);
	if(frameNum == curSubmittedFrame)
	{
		//printf("sub if\n");
		//__outVideo << *frame;
		__outcsv << red << "," << (frameNum/10.0) << "\n";
		curSubmittedFrame += jump;
		//delete frame;
		printf("Frame %d completed.\n",frameNum);
		int i=0; 
		//printf("starting while\n");
		while(i < waitingFrames.size() && waitingFrames[i]->frameNum == curSubmittedFrame)
		{
			//printf("in while\n");
			//__outVideo << (*(waitingFrames[i]->mat));
			__outcsv << waitingFrames[i]->red << "," << (waitingFrames[i]->frameNum/10.0) << "\n";
			//printf("pushed\n");
			//printf("++\n");
			printf("Frame %d completed.\n",waitingFrames[i]->frameNum);
			//delete waitingFrames[i]->mat;
			delete waitingFrames[i];

			curSubmittedFrame += jump;
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
		if(done && curSubmittedFrame == curFrame)
			returnVal = true;
	}
	else
	{
		//printf("sub else\n");
		Frame *newframe = new Frame;
		//newframe->mat = frame; 
		newframe->frameNum = frameNum;
		newframe->red = red;
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
	return returnVal;
}

void submitThreadMPI()
{
	unsigned int buffer[2]; //[frameNumber, red]
	while(!stopThread)
	{
		MPI_Recv(buffer, 2, MPI_UNSIGNED, MPI_ANY_SOURCE, SUBMIT_FRAME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		stopThread = submitFrameGlobal(buffer[0], buffer[1]);
	}
	exit(0);
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
	// printf("breakUpImage\n");
	int numrows = image.rows;
	int numcols = image.cols;

	//printf("%d %d\n",numrows, numcols);

	vector<vector< vector<double> > > fullImage; //2 dims for width and height, last dim for each possible category
	resize3DVector(fullImage,numrows,numcols,net.getNumClasses());
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
		//preprocess(imageRow);
		net.setData(imageRow);
		net.run();
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
	//if(device != 2) return;
	Net net(__netName);
	net.setConstantMem(true);
	if(!net.setDevice(device) || !net.finalize())
		return;
	printf("Thread using rank %d, device %d\n", my_rank, device);

	unsigned int frameNum=0;
	Mat frame;
	bool valid;
	int red;
	// printf("gonna get next frame\n");
	getFirstFrameInfo();
	// printf("past next frame\n");
	// printf("past 2\n");
	while(true)
	{
		valid = getNextFrame(frame,frameNum);
		// cout << "Valid: " << valid << endl;
		if(!valid)
			break;
		// printf("while loop\n");
		//printf("thread %d got frame %d\n", device, frameNum);
		Mat* outFrame = new Mat(__rows,__cols,CV_8UC3);
		//printf("thread %d: starting breakUpImage %d\n",device,frameNum);
		breakUpImage(frame, net, *outFrame, red);
		//printf("thread %d: submitting frame %d\n",device,frameNum);
		//submitFrame(frameNum, red);//, device);	
		//storeFrame(frameNum,red);
	}
	submitFrames();
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

	if(my_rank == 0)
	{
		__globalVideo.open(videoName);
		//make threads for getting frame num and submitting
		// getThread = new thread(getThreadMPI);
		// submitThread = new thread(submitThreadMPI);
		getThreadMPI();
		submitThreadMPI();
		return;
	}


	char outNameCSV[255];
	string origName(videoName);
	size_t dot = origName.rfind('.');
	const char *noExtension = origName.substr(0,dot).c_str();

	//sprintf(outName,"%s_prediction%s",noExtension,".avi");
	sprintf(outNameCSV,"%s_prediction.csv",noExtension);
	//cout << "writing " << outName << endl;

	__outcsv.open(outNameCSV);

	// __outVideo.open(outName, 
	//  CV_FOURCC('M', 'J', 'P', 'G'),//-1,//video.get(CV_CAP_PROP_FOURCC),
	//  10,//video.get(CV_CAP_PROP_FPS), 
	//  Size(__video.get(CV_CAP_PROP_FRAME_WIDTH), __video.get(CV_CAP_PROP_FRAME_HEIGHT)));

	__rows = __video.get(CV_CAP_PROP_FRAME_HEIGHT);
	__cols = __video.get(CV_CAP_PROP_FRAME_WIDTH);
	// int numDevices = getNumDevices();
	// thread* t = new thread[numDevices];
	// for(int i=0; i < numDevices; i++)
	// {
	// 	//start new thread for each device. Any thread for a device that does not support double will return early.
	// 	t[i] = thread(__parallelVideoProcessor, i);
	// }

	// for(int i=0; i < numDevices; i++)
	// {
	// 	t[i].join();
	// }

	__parallelVideoProcessor(0);

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
	
	time_t starttime, endtime;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if(argc < 3 || 5 < argc)
	{
		if(my_rank == 0)
			printf("use format: ./ConvNetVideoDriver cnnConfig.txt VideoOrFolderPath stride=<1> jump=<1>\n");
		MPI_Finalize();
		return 0;
	}
	if(comm_sz < 2)
	{
		if(my_rank == 0)
			printf("Must have at least 2 processes. Exiting\n");
		MPI_Finalize();
		return 0;
	}


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
			else
			{
				printf("Unknown arg \"%s\". Aborting.\n", argv[i]);
				return 0;
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
	MPI_Finalize();
	return 0;
	
}