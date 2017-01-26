/************************************
*
*	ConvNetTester
*
*	Created by Connor Bowley on 5/2/16
*
*	This program is for running trained CNNs that were trained using ConvNet.
*
*	Usage:
*		./ConvNetTester CNNConfigFile.txt binaryTrainingImagesFile <gpu=true>
*
*************************************/

#include "ConvNet.h"
#include "ConvNetCommon.h"

#include <cctype>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace convnet;

int main(int argc, char** argv)
{
	if(argc != 3 && argc != 4)
	{
		cout << "Usage: ./ConvNetTester CNNConfigFile.txt binaryTrainingImagesFile <gpu=true>" << endl;
		return 0;
	}

	bool useGPU = true;

	if(argc == 4)
	{
		string gpu(argv[3]);
		if(gpu.find("gpu=") != string::npos)
		{
			if(gpu.find("false") != string::npos || gpu.find("False") != string::npos)
			{
				useGPU = false;
			}
		}
	}

	time_t starttime,endtime;

	//set up net
	Net net(argv[1]);
	if(!net.isActive())
		return 0;

	//get images and preprocess them
	vector<imVector> images(0);
	vector<double> trueVals(0);

	starttime = time(NULL);
	if (!convertBinaryToVector(argv[2], images, trueVals)) {
		cout << "Exiting." << endl;
		return 0;
	}
	endtime = time(NULL);
	cout << "Time to bring in training data: " << secondsToString(endtime - starttime) << endl;

	in.close();
	//preprocessByFeature(images);
	//preprocessCollective(images);
	preprocess(images);
	//meanSubtraction(images);

	net.addTrainingData(images,trueVals);

	//net.printTrainingDistribution();

	vector<int> calced(0);

	starttime = time(NULL);
	net.newRun(calced,useGPU);
	endtime = time(NULL);
	cout << "Time for OpenCL code: " << secondsToString(endtime - starttime) << ". " << endl;
}