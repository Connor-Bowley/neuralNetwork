/*********************************************
 *
 *	ConvNetTest
 *
 * 	Created by Connor Bowley on 3/16/16
 *
 *
 *	Get the image vectors made without using Mat
 *
 *********************************************/

#include "ConvNet.h"
#include "ConvNetCommon.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cassert>
#include <ctime>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <thread>
#include <vector>

using namespace std;
using namespace cv;
using namespace convnet;

void _t_convertColorMatToVector(const Mat& m , imVector& dest, int row)
{
	for(int j=0; j< m.cols; j++)
	{
		const Vec3b curPixel = m.at<Vec3b>(row,j);
		dest[row][j][0] = curPixel[0];
		dest[row][j][1] = curPixel[1];
		dest[row][j][2] = curPixel[2];
	}
}

void threadTest(imVector& dest)
{
	cout << dest.size() << endl;
}

void convertColorMatToVector(const Mat& m, imVector& dest)
{
	if(m.type() != CV_8UC3)
	{
		throw "Incorrect Mat type. Must be CV_8UC3.";
	}

	int width2 = m.rows;
	int height2 = m.cols;
	int depth2 = 3;
	//resize dest vector
	resize3DVector<double>(dest,width2,height2,depth2);
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
	dest.resize(width2);
	for(int i=0; i < width2; i++)
	{
		dest[i].resize(height2);
		for(int j=0; j < height2; j++)
		{
			dest[i][j].resize(depth2);
		}
	}
	
	for(int i=0; i< m.rows; i++)
	{
		for(int j=0; j< m.cols; j++)
		{
			Vec3b curPixel = m.at<Vec3b>(i,j);
			dest[i][j][0] = curPixel[0];
			dest[i][j][1] = curPixel[1];
			dest[i][j][2] = curPixel[2];
		}
	}
}
*/

void getImageInVector(const char* filename, vector<vector<vector<double> > >& dest)
{
	Mat image = imread(filename,1);
	convertColorMatToVector(image,dest);
}

int checkExtensions(const char* filename)
{
	const string name = filename;
	if(name.rfind(".jpg")  == name.length() - 4) return 1;
	if(name.rfind(".jpeg") == name.length() - 5) return 1;
	if(name.rfind(".png")  == name.length() - 4) return 1;
	return 0;
}

void oldestMain()
{
	string weights = "1,0,1,-1,0,0,-1,1,0,0,1,0,1,0,0,1,-1,-1,-1,0,0,1,1,-1,-1,1,0,0,1,-1,-1,-1,-1,-1,0,-1,1,1,-1,0,0,-1,0,1,-1,1,-1,1,1,1,-1,-1,0,1,_1,0";

	double testArray[] = {
		0,1,2, 1,2,1, 0,0,0, 2,2,2, 2,0,1,
		2,2,1, 1,0,0, 2,1,1, 2,2,0, 1,2,0,
		1,1,0, 1,0,1, 2,2,0, 0,1,0, 2,2,2,
		1,1,0, 0,2,1, 0,1,0, 1,2,2, 2,0,2,
		2,0,0, 2,0,2, 0,1,1, 2,1,0, 2,2,0};

	imVector testVectors(1);
	vector<double> trueVals(1,1);
	
	convert1DArrayTo3DVector<double>(testArray,5,5,3,testVectors[0]);


	cout << "Making Net" << endl;
	Net *testNet = new Net(5,5,3);
	testNet->addTrainingData(testVectors,trueVals);
	//cout << "Adding ConvLayer" << endl;
	testNet->addConvLayer(2,2,3,0,weights);
	//const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad)
	//testNet->addActivLayer();
	//testNet->addMaxPoolLayer(2,1);

	testNet->run(false);


	cout << "Done" << endl;
	//delete testNet;
}

void getTrainingImages(const char* folder, int trueVal, imVector& images, vector<double>& trueVals)
{
	const char* inPath = folder;
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
			return;
		}
	}
	else
	{
		cout << "Error getting status of folder or file. \"" << folder << "\"\nExiting\n";
		return;
	}
	
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
						//images.push_back();
						images.resize(images.size() + 1);
						//cout << "getImageInVector(" << inPathandName << ",images[" << images.size()-1 << "])" << endl;
						getImageInVector(inPathandName,images.back());
						trueVals.push_back(trueVal);
					}
				}
			}
			closedir(directory);
		}
	}
	else
	{
		if(checkExtensions(inPath))
		{
			//images.push_back();
			images.resize(images.size() + 1);
			getImageInVector(inPath,images.back());
			trueVals.push_back(trueVal);
		}

	}
}

int runTrainedCNN(int argc, char** argv)
{
	if(argc != 3)
	{
		cout << "Use: ./ConvNetTest trainingImagesConfig.txt nnConfig.txt. A trainingImagesConfig file is needed. See the readMe for format." << endl;
		return 0;
	}
	cout << "Running a trained NeuralNet: " << argv[2] << endl;
	cout << "Building NeuralNet" << endl;
	//set up CNN
	
	Net net(argv[2]);

	cout << "NeuralNet set up" << endl;
	
	cout << "Getting training images" << endl;

	vector<vector<vector<vector<double> > > > trainingImages;
	vector<double> trueVals;

	ifstream tiConfig;
	string line;
	tiConfig.open(argv[1]);
	if(!tiConfig.is_open())
	{
		cout << "Could not open the trainingImagesConfig file " << argv[1];
		return -1;
	}
	while(getline(tiConfig,line))
	{
		int loc = line.find(",");
		if(loc != string::npos)
		{
			cout << "Adding folder" << line.substr(0,loc) << endl;
			getTrainingImages(line.substr(0,loc).c_str(),stoi(line.substr(loc+1)),trainingImages,trueVals);
		}
		else
			cout << "Error in trainingImagesConfig at line:\n" << line << endl;
	}
	tiConfig.close();

	cout << trainingImages.size() << " training images added." << endl;

	assert(trainingImages.size() == trueVals.size());
	if(trainingImages.size() == 0)
	{
		cout << "No training images found. Exiting." << endl;
		return 0;
	}
	// do (val - mean)/stdDeviation

	//cout << "Doing image preprocessing (compress vals from -1 to 1)" << endl;
	//compressImage(trainingImages,-1,1);
	
	cout << "Doing image preprocessing." << endl;
	preprocess(trainingImages);

	cout << "Adding training images to Network" << endl;
	net.addTrainingData(trainingImages,trueVals);

	//shuffling shouldnt matter if there is no backprop
	//net.shuffleTrainingData(10);

	cout << "Doing a run without learning on training images" << endl;
	vector<int> calcedClasses;
	//net.run(false);
	time_t starttime = time(NULL);
	net.newRun(calcedClasses,false);
	time_t endtime = time(NULL);
	cout << "Time for OpenCL code: " << secondsToString(endtime - starttime) << endl;


	cout << "Done" << endl;
	return 0;
}

int continueTrainingCNN(int argc, char** argv)
{
	if(argc != 3 && argc != 4)
	{
		cout << "Use: ./ConvNetTest trainingImagesConfig.txt nnConfig.txt.\n./ConvNetTest trainingImagesConfig.txt nnConfig.txt newOutfile.txt." << endl;
		cout <<  "If a new outFile is not specified any new training will be saved over the old nnConfig file.\nA trainingImagesConfig file is needed. See the readMe for format." << endl;
		return 0;
	}
	cout << "Running a trained NeuralNet: " << argv[2] << endl;
	cout << "Building NeuralNet" << endl;
	//set up CNN
	
	Net net(argv[2]);
	Net net2(argv[2]);

	cout << "NeuralNet set up" << endl;
	
	cout << "Getting training images" << endl;

	vector<vector<vector<vector<double> > > > trainingImages;
	vector<double> trueVals;

	ifstream tiConfig;
	string line;
	tiConfig.open(argv[1]);
	if(!tiConfig.is_open())
	{
		cout << "Could not open the trainingImagesConfig file " << argv[1];
		return -1;
	}
	while(getline(tiConfig,line))
	{
		int loc = line.find(",");
		if(loc != string::npos)
		{
			cout << "Adding folder" << line.substr(0,loc) << endl;
			getTrainingImages(line.substr(0,loc).c_str(),stoi(line.substr(loc+1)),trainingImages,trueVals);
		}
		else
			cout << "Error in trainingImagesConfig at line:\n" << line << endl;
	}
	tiConfig.close();

	cout << trainingImages.size() << " training images added." << endl;

	assert(trainingImages.size() == trueVals.size());
	if(trainingImages.size() == 0)
	{
		cout << "No training images found. Exiting." << endl;
		return 0;
	}
	// do (val - mean)/stdDeviation

	//cout << "Doing image preprocessing (compress vals from -1 to 1)" << endl;
	//compressImage(trainingImages,-1,1);
	
	cout << "Doing image preprocessing." << endl;
	//preprocess(trainingImages);
	meanSubtraction(trainingImages);

	cout << "Adding training images to Network" << endl;
	net.addTrainingData(trainingImages,trueVals);
	net2.addTrainingData(trainingImages,trueVals);

	//shuffling shouldnt matter if there is no backprop
	//net.shuffleTrainingData(10);

	time_t starttime, endtime;

	cout << "Continuing training net" << endl;
	int epochs = 5;
	/*
	starttime = time(NULL);
	net.train(epochs);
	endtime = time(NULL);
	cout << "Time for non OpenCL code: " << endtime - starttime << " seconds" << endl;
	*/
	starttime = time(NULL);
	net2.OpenCLTrain(epochs, false);
	endtime = time(NULL);
	cout << "Time for OpenCL code: " << secondsToString(endtime - starttime) << endl;
	

	//net.save("origTrain.txt");
	net2.save("clTrain.txt");

	/*
	if(argc == 4)
	{
		cout << "Saving CNN to " << argv[3] << endl;
		net.save(argv[3]);
	}
	else
	{
		cout << "Saving CNN back to " << argv[2] << endl;
		net.save(argv[2]);
	}
	*/

	cout << "Done" << endl;
	return 0;
} // end continueTrainingCNN

int trainCNN(int argc, char** argv)
{
	cout << "Training a new Neural Net" << endl;
	if(argc != 2 && argc != 3)
	{
		cout << "Uses: ./ConvNetTest trainingImagesConfig.txt\n"<<
		"./ConvNetTest trainingImagesConfig.txt outputFilename.txt\nA trainingImagesConfig file is needed. See the readMe for format." << endl;
		return -1;
	}
	cout << "Building NeuralNet" << endl;

	Net net(32,32,3);           //32x32x3
	net.setActivType(ActivLayer::LEAKY_RELU);
	net.addConvLayer(6,1,5,0);	//28x28x6
	net.addActivLayer();
	net.addMaxPoolLayer(2,2);	//14x14x6
	net.addConvLayer(10,1,3,0);	//12x12x10
	net.addActivLayer();
	net.addMaxPoolLayer(3,3);	//4x4x10
	//net.addConvLayer(5,1,3,1);	//4x4x5
	net.addActivLayer();
	net.addConvLayer(2,1,4,0);	//1x1x2
	net.addActivLayer();
	net.addSoftmaxLayer();

	//set up CNN
	/*
	Net net(32,32,3);
	net.setActivType(ActivLayer::LEAKY_RELU);
	net.addConvLayer(20, 1, 3, 1); //numfilters, stride, filtersize, padding
	net.addActivLayer(); 			//32x32x20
	net.addConvLayer(10,1,3,1);		//32x32x10
	net.addActivLayer();			
	net.addMaxPoolLayer(2,2);		//16x16x10
	net.addConvLayer(20,1,3,1);		//16x16x20
	net.addActivLayer();
	net.addMaxPoolLayer(2,2);		//8x8x20
	net.addConvLayer(40,1,3,1);		//8x8x40
	net.addActivLayer();
	net.addMaxPoolLayer(2,2);		//4x4x40
	net.addConvLayer(30,1,3,1);		//4x4x30
	net.addActivLayer();
	net.addMaxPoolLayer(2,2);		//2x2x30
	net.addConvLayer(20,1,3,1);		//2x2x20
	net.addActivLayer();
	net.addMaxPoolLayer(2,2);		//1x1x20
	net.addConvLayer(4,1,3,1);		//1x1x4
	net.addSoftmaxLayer();
	*/

	
	/*
	Net net(32,32,3);
	net.setActivType(ActivLayer::LEAKY_RELU);
	net.addConvLayer(6,1,5,0);
	net.addActivLayer();
	net.addMaxPoolLayer(2,2);
	net.addConvLayer(10,1,3,0);
	net.addActivLayer();
	net.addMaxPoolLayer(3,3);
	net.addConvLayer(2,1,4,0);
	net.addActivLayer();
	net.addSoftmaxLayer();
	*/
	
	//Net net("oneEpoch.txt");

	cout << "NeuralNet set up" << endl;

	net.printLayerDims();
	//getchar();
	
	time_t imageStart = time(NULL);
	cout << "Getting training images" << endl;

	vector<vector<vector<vector<double> > > > trainingImages;
	vector<double> trueVals;

	ifstream tiConfig;
	string line;
	tiConfig.open(argv[1]);
	if(!tiConfig.is_open())
	{
		cout << "Could not open the trainingImagesConfig file " << argv[1];
		return -1;
	}
	while(getline(tiConfig,line))
	{
		int loc = line.find(",");
		if(loc != string::npos)
		{
			cout << "Adding folder" << line.substr(0,loc) << endl;
			getTrainingImages(line.substr(0,loc).c_str(),stoi(line.substr(loc+1)),trainingImages,trueVals);
		}
		else
			cout << "Error in trainingImagesConfig at line:\n" << line << endl;
	}
	tiConfig.close();

	cout << trainingImages.size() << " training images added." << endl;
	time_t imageEnd = time(NULL);
	cout << "Time needed for getting training images: " << secondsToString(imageEnd-imageStart) << endl;

	assert(trainingImages.size() == trueVals.size());
	if(trainingImages.size() == 0)
	{
		cout << "No training images found. Exiting." << endl;
		return 0;
	}
	// do (val - mean)/stdDeviation

	//cout << "Doing image preprocessing (compress vals from -1 to 1)" << endl;
	//compressImage(trainingImages,-1,1);
	
	cout << "Doing image preprocessing." << endl;
	//preprocess(trainingImages);
	meanSubtraction(trainingImages);

	cout << "Adding training images to Network" << endl;
	net.addTrainingData(trainingImages,trueVals);

	net.shuffleTrainingData(10);

	//cout << "Doing gradient check" << endl;
	//net.gradientCheck();

	//cout << "Split Training Neural Network" << endl;
	//net.splitTrain(1);

	cout << "Training Neural Network using OpenCL" << endl;
	time_t trainStart = time(NULL);
	net.OpenCLTrain(5, false);
	//net.train(5);
	time_t trainEnd = time(NULL);
	cout << "Time for training 5 epochs: " << secondsToString(trainEnd - trainStart) << endl;


	//int batchSize = 10;
	//cout << "MiniBatch Training Neural Network. Batch Size = "<< batchSize << endl;
	//net.miniBatchTrain(100, batchSize);

	//cout << "Doing a run without learning on training images" << endl;
	//net.runTrainingData();

	if(argc == 3)
	{
		cout << "Saving CNN to " << argv[2] << endl;
		net.save(argv[2]);
	}

	cout << "Done" << endl;
	return 0;
}

int main(int argc, char** argv)
{
	trainCNN(argc, argv);
	//continueTrainingCNN(argc, argv);
	//runTrainedCNN(argc, argv);
	//cout << "back in main" << endl;

	//oldestMain();
	return 0;
}


















