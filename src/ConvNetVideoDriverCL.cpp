#include "ConvNetCL.h"
#include "ConvNetCommon.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cassert>
#include <cctype>
#include <ctime>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <thread>
#include <vector>

#ifdef WIN32
# include <io.h>
#else
# include <unistd.h>
#endif

using namespace cv;
using namespace std;
using namespace convnet;

char *inPath, *outPath;
int imageNum = 0;
int stride = 1;
bool __useGPU = true;

unsigned int __frameNum = 0;

int __width;
int __height;

int __momentRed = 0;

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
	size_t length = 0;
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

	squareElements3DVector(fullImage);
	
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
				uchar blue = (uchar)(255.0*fullImage[i][j][0]);
				outPix[0] = blue; // blue
				outPix[1] = 0;	  //green
				uchar red = (uchar)(255.0*fullImage[i][j][1]);
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
	__momentRed = (int)(.8*__momentRed + .8*redElement);

	outVideo.write(outputMat);
	outcsv << __momentRed << "," << __frameNum/10.0 << "\n";
	//outVideo << outputMat;
}

void breakUpVideo(const char* videoName, Net& net)
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

	__width = (int)video.get(CV_CAP_PROP_FRAME_WIDTH);
	__height = (int)video.get(CV_CAP_PROP_FRAME_HEIGHT);

	if(!outVideo.isOpened())
	{
		cout << "Could not open out video" << endl;
		return;
	}

	Mat frame;
	unsigned long count = 0;

	if(video.read(frame))
	{
		printf("Frame %ld. \t%3.4lf%%\n", ++count, video.get(CV_CAP_PROP_POS_AVI_RATIO) * 100.0);
		breakUpImage(frame, net, outVideo, outcsv);
		__frameNum++;
		while(video.read(frame))
		{
			//printf("Frame %ld of %.0lf\n", ++count, video.get(CV_CAP_PROP_FRAME_COUNT));
			printf("Frame %ld. \t%3.4lf%%\n", ++count, video.get(CV_CAP_PROP_POS_AVI_RATIO) * 100.0);
			breakUpImage(frame, net, outVideo, outcsv);
			__frameNum++;
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
		printf("use format: ./ConvNetVideoDriverCL cnnConfig.txt VideoOrFolderPath stride=<1> gpu=<true/false> device=<device#>\n");
		return -1;
	}

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


	//set up net
	Net net(argv[1]);
	net.setDevice(device);
	if(!net.finalize())
		return 0;


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
		breakUpVideo(filenames[i].c_str(),net);
		endtime = time(NULL);
		cout << "Time for video " << filenames[i] << ": " << secondsToString(endtime - starttime) << endl;
	}

	//cout << "returning" << endl;
	return 0;
	
}