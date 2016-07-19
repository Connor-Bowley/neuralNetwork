
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
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
// #include <thread>


using namespace cv;
using namespace std;

typedef vector<vector<vector<double> > > imVector;

char *inPath, *outPath;
int imageNum = 0;
int stride = 1;

int inputWidth, inputHeight;

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

double vectorSumSq(const vector<double>& vect)
{
	double sum=0;
	for(int i=0; i<vect.size(); i++)
		sum += vect[i] * vect[i];
	return sum;
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

// void _t_convertColorMatToVector(const Mat& m , vector<vector<vector<double> > > &dest, int row)
// {
// 	for(int j=0; j< m.cols; j++)
// 	{
// 		const Vec3b& curPixel = m.at<Vec3b>(row,j);
// 		dest[row][j][0] = curPixel[0];
// 		dest[row][j][1] = curPixel[1];
// 		dest[row][j][2] = curPixel[2];
// 	}
// }

// void convertColorMatToVector(const Mat& m, vector<vector<vector<double> > > &dest)
// {
// 	if(m.type() != CV_8UC3)
// 	{
// 		throw "Incorrect Mat type. Must be CV_8UC3.";
// 	}

// 	int width2 = m.rows;
// 	int height2 = m.cols;
// 	int depth2 = 3;
// 	//resize dest vector
// 	resize3DVector(dest,width2,height2,depth2);
// 	thread *t = new thread[width2];
	
// 	for(int i=0; i< width2; i++)
// 	{
// 		t[i] = thread(_t_convertColorMatToVector,ref(m),ref(dest),i);
// 	}

// 	for(int i=0; i< width2; i++)
// 	{
// 		t[i].join();
// 	}

// 	//delete t;
// }

/*
 * The inner for loop gets the confidences for each pixel in the image. If a pixel is in more than one subimage
 * (i.e. the stride is less than the subimage size), then the confidences from each subimage is added.
 */
void breakUpImage(const char* imageName, Net& net)
{
	//cout << "starting breakUpImage" << endl;
	Mat image = imread(imageName,1);
	int numrows = image.rows;
	int numcols = image.cols;
	printf("%s rows: %d, cols: %d\n",imageName, numrows,numcols);


	vector<vector< vector<double> > > fullImage; //2 dims for width and height, last dim for each possible category
	resize3DVector(fullImage,numrows,numcols,net.getNumClasses());
	setAll3DVector(fullImage,0);
	// vector<imVector> imageRow(0); // this will hold all subimages from one row
	vector<Mat> imageRow(0);
	vector<int> calcedClasses(0);
	vector<vector<double> > confidences(0);//for the confidence for each category for each image
		//the outer vector is the image, the inner vector is the category, the double is output(confidence) of the softmax

	int numrowsm32 = numrows - inputHeight;
	int numcolsm32 = numcols - inputWidth;
	if(numrows < inputHeight || numcols < inputWidth)
	{
		printf("The image %s is too small in at least one dimension. Minimum size is %dx%d.\n",imageName,inputHeight,inputWidth);
		return;
	}
	for(int i=0; i <= numrowsm32; i+=stride)
	{
		imageRow.resize(0);
		printf("row %d of %d (%d)\n",i,numrowsm32,numrows);
		//cout << "row " << i << " of " << numrows << endl;
		//get all subimages from a row
		for(int j=0; j<= numcolsm32; j+=stride) //NOTE: each j is a different subimage
		{
			const Mat out = image(Range(i,i+inputHeight),Range(j,j+inputWidth));
			imageRow.push_back(out);
			// imageRow.resize(imageRow.size()+1);
			// convertColorMatToVector(out,imageRow.back());
		}
		//set them as the data in the net
		//preprocess(imageRow);
		net.setData(imageRow);
		net.run();
		net.getConfidences(confidences); //gets the confidence for each category for each image
		//if((i == 0 || i == numrows-32))
		//cout << "row: " << i << " ";
		//printVector(confidences[0]);
		//cout << "conf got" << endl;
		int curImage = 0;
		for(int j=0; j<= numcolsm32; j+=stride) //NOTE: each iteration of this loop is a different subimage
		{
			for(int ii=i; ii < i+inputHeight && ii < numrows; ii++)
			{
				for(int jj=j; jj < j+inputHeight && jj < numcols; jj++)
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
	


	//now we have the confidences for every pixel in the image
	//so get the category for each pixel and make a new image from it
	Mat outputMat(numrows,numcols,CV_8UC3);
	for(int i=0; i < numrows; i++)
	{
		for(int j=0; j < numcols; j++)
		{
			/*//straight ratios
			double sum = vectorSum(fullImage[i][j]);
			for(int n=0; n < fullImage[i][j].size(); n++)
			{
				fullImage[i][j][n] /= sum;
			}*/

			//square ratios
			double sumsq = vectorSumSq(fullImage[i][j]);
			for(int n=0; n < fullImage[i][j].size(); n++)
			{
				fullImage[i][j][n] = fullImage[i][j][n] * fullImage[i][j][n] / sumsq;
			}

			//write the pixel
			Vec3b& outPix = outputMat.at<Vec3b>(i,j);
			//int maxEle = getMaxElementIndex(fullImage[i][j]);
			if(allElementsEquals(fullImage[i][j]))
			{
				outPix[0] = 0; outPix[1] = 0; outPix[2] = 0; // black
			}
			else
			{
				outPix[0] = 255*fullImage[i][j][0]; // blue
				outPix[1] = 255*fullImage[i][j][2]; // green
				outPix[2] = 255*fullImage[i][j][1]; // red
			}
			/*//write only blue and red, no in between
			else if(maxEle == 0)
			{
				outPix[0] = 255; outPix[1] = 0; outPix[2] = 0; // blue
			}
			else if(maxEle == 1)
			{
				outPix[0] = 0; outPix[1] = 0; outPix[2] = 255; // red
			}
			*/
		}
	}
	char outName[255];
	string origName(imageName);
	size_t dot = origName.rfind('.');
	const char *noExtension = origName.substr(0,dot).c_str();
	const char *extension = origName.substr(dot).c_str();

	sprintf(outName,"%s_prediction%s",noExtension,extension);
	cout << "writing " << outName << endl;
	imwrite(outName, outputMat);

	
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
		printf("use format: ./ConvNetFullImageDriverCL cnnConfig.txt imageOrFolderPath (stride=1) (device=0)\n");
		return 0;
	}

	inPath = argv[2];

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
			else if(arg.find("device") != string::npos)
			{
				device = stoi(arg.substr(arg.find("=")+1));
			}
		}
	}


	//set up net
	Net net(argv[1]);
	if(device != -1)
		net.setDevice(device);

	inputWidth = net.getInputWidth();
	inputHeight = net.getInputHeight();
	
	//cout << "net loaded" << endl;
	if(!net.finalize())
		return 0;
	//cout << "net finalized" << endl;



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
	time_t starttime, endtime;

	for(int i=0; i < filenames.size(); i++)
	{
		starttime = time(NULL);
		breakUpImage(filenames[i].c_str(),net);
		endtime = time(NULL);
		cout << "Time for image " << filenames[i] << ": " << secondsToString(endtime - starttime) << endl;
	}

	//cout << "returning" << endl;
	return 0;
	
}