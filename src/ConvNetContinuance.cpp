/************************************
*
*	ConvNetTrainer
*
*	Created by Connor Bowley on 5/2/16
*
*	This program is for running trained CNNs that were trained using ConvNet.
*
*	Usage:
*		./ConvNetTrainer binaryTrainingImagesFile outname=<outputName.txt> epochs=<#epochs> gpu=<true/false>
*
* 		The outputName and epochs are optional keyword arguments. If no outname is specified the weights are not saved. Epochs defaults to 1. Defaults to using GPU
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
	if(argc < 3)
	{
		cout << "Usage: ./ConvNetContinuance CNNConfigFile.txt binaryTrainingImagesFile outname=<outputName.txt> epochs=<#epochs> gpu=<true/false> device=<device#>";
		cout << "\n\tThe outputName and epochs are optional keyword arguments. If no outname is specified the new weights will be saved over the old weights. Epochs defaults to 1. Defaults to using GPU" << endl;
		return 0;
	}

	time_t starttime,endtime;

	string outputName;
	bool saveNewWeights = false;
	int epochs = 1;
	int device = -1;
	bool useGPU = true;

	if(argc > 3)
	{
		for(int i = 3; i < argc; i++)
		{
			string arg(argv[i]);
			if(arg.find("outname=") != string::npos)
			{
				outputName = arg.substr(arg.find("=") + 1);
				saveNewWeights = true;
			}
			else if(arg.find("epochs=") != string::npos || arg.find("epoch=") != string::npos)
			{
				epochs = stoi(arg.substr(arg.find("=")+1));
			}
			else if(arg.find("gpu=") != string::npos)
			{
				if(arg.find("false") != string::npos || arg.find("False") != string::npos)
				{
					useGPU = false;
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
	if(!net.isActive())
	{
		cout << "Net did not load correctly. Exiting." << endl;
		return 0;
	}
	

	//get images and preprocess them
	vector<imVector> images(0);
	vector<double> trueVals(0);

	starttime = time(NULL);
	if (!convertBinaryToVector(argv[2], images, &trueVals)) {
		cout << "Exiting." << endl;
		return 0;
	}
	endtime = time(NULL);
	cout << "Time to bring in training data: " << secondsToString(endtime - starttime) << endl;

	//preprocessByFeature(images);
	//preprocessCollective(images);
	preprocess(images);
	//meanSubtraction(images);

	net.shuffleTrainingData(10);

	cout << "Adding training data" << endl;
	net.addTrainingData(images,trueVals);

	net.printTrainingDistribution();

	net.setDevice(device);

	starttime = time(NULL);
	net.OpenCLTrain(epochs, useGPU);
	endtime = time(NULL);
	cout << "Time for OpenCL code: " << secondsToString(endtime - starttime) << ". - " << secondsToString((endtime-starttime)/(float)epochs) << " per epoch." << endl;

	if(saveNewWeights)
	{
		net.save(outputName.c_str());
	}
	else
	{
		net.save(argv[1]);
	}
}