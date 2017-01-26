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
		cout << "Usage: ./ConvNetTrainer newCNNConfigFile.txt binaryTrainingImagesFile outname=<outputName.txt> epochs=<#epochs> gpu=<true/false> device=<device#>";
		cout << "\n\tThe outputName and epochs are optional keyword arguments. If no outname is specified the weights are not saved. Epochs defaults to 1. Defaults to using GPU" << endl;
		return 0;
	}

	time_t starttime,endtime;

	string outputName;
	bool saveWeights = false;
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
				saveWeights = true;
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
	Net net(xSize,ySize,zSize);

	//64x64x3 net
//	net.setActivType(ActivLayer::LEAKY_RELU);
//	net.addConvLayer(10,1,3,1); //64x64x10
//	net.addActivLayer();
//	net.addMaxPoolLayer(2,2); 	//32x32x10
//	net.addConvLayer(6,1,5,0);	//28x28x6
//	net.addActivLayer();
//	net.addMaxPoolLayer(2,2);	//14x14x6
//	net.addConvLayer(7,1,3,1);	//14x14x7
//	net.addActivLayer();
//	net.addConvLayer(10,1,3,0);	//12x12x10
//	net.addActivLayer();
//	net.addMaxPoolLayer(3,3);	//4x4x10
//	net.addConvLayer(5,1,3,1);	//4x4x5
//	net.addActivLayer();
//	net.addConvLayer(4,1,4,0);	//1x1x4
//	net.addActivLayer();
//	net.addSoftmaxLayer();
	//*/

	// failed fully connected net
	/*
	net.setActivType(ActivLayer::RELU);
	net.addConvLayer(10,1,3,0); 		//30x30x10
	net.addActivLayer();
	net.addMaxPoolLayer(2,2);		//15x15x10
	net.addConvLayer(2,1,15,0);		//1x1x2
	net.addActivLayer();
	net.addSoftmaxLayer();
	//*/
	
	/* large net
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
	net.addConvLayer(2,1,3,1);		//1x1x4
	net.addSoftmaxLayer();
	//*/
	
	
	//small net
	net.setActivType(ActivLayer::LEAKY_RELU);
	net.addConvLayer(6,1,5,0);
	net.addActivLayer();
	net.addMaxPoolLayer(2,2);

	// net.addConvLayer(7,1,3,1);
	// net.addActivLayer();

	net.addConvLayer(10,1,3,0);
	net.addActivLayer();
	net.addMaxPoolLayer(3,3);

	// net.addConvLayer(5,1,3,1);	//4x4x5
	// net.addActivLayer();

	net.addConvLayer(2,1,4,0);
	net.addActivLayer();
	net.addSoftmaxLayer();
	//*/

	if(!net.isActive())
	{
		cout << "Something went wrong making the net. Exiting." << endl;
		return 0;
	}
	

	//get images and preprocess them
	vector<imVector> images(0);
	vector<double> trueVals(0);

	starttime = time(NULL);
	if (!convertBinaryToVector(in, images, &trueVals)) {
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

	//net.shuffleTrainingData(10);

	cout << "Adding training data" << endl;
	net.addTrainingData(images,trueVals);

	net.printTrainingDistribution();

	net.setDevice(device);

	starttime = time(NULL);
	net.OpenCLTrain(epochs, useGPU);
	endtime = time(NULL);
	cout << "Time for OpenCL code: " << secondsToString(endtime - starttime) << ". - " << secondsToString((endtime-starttime)/(float)epochs) << " per epoch." << endl;

	if(saveWeights)
	{
		net.save(outputName.c_str());
	}
}