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

#include "ConvNetCL.h"
#include "ConvNetCommon.h"

#include <cctype>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace convnet;

int main(int argc, char** argv)
{
	if(argc < 3)
	{
		cout << "Usage (Required to come first):\n   ./ConvNetContinuanceCL CNNConfigFile.txt binaryTrainingImagesFile";
		cout << "\nOptional arguments (must come after required args, everything before equals sign is case sensitive):\n";
		cout << "   outname=<outname.txt>    Sets the name for the outputted trained CNN. If not specified new weights will not be saved.\n";
		cout << "   testSet=<name.txt>       A binary training file to be used as a test/validation set. Never trained on.\n";
		cout << "   epochs=<#epochs>         Number of epochs to train for. Defaults to 1.\n";
		cout << "   device=<device#>         Which OpenCL device on to use. Integer. Defaults to GPU supporting doubles if present, else defaults to CPU.\n";
		cout << "   -train_as_is             Causes CNN to train using all images for every epoch. On by default. Can only use one train method at a time\n";
		cout << "   -train_equal_prop        Causes CNN to train using equal amounts of each class for each epoch. For classes with larger amounts of images,\n";
		cout << "                            the ones used will be randomly chosen each epoch. Can only use one train method at a time\n";
		cout << "   learningRate=<rate>        Sets the learningRate for the CNN.\n";
		cout << "   RELU_CAP=<cap>             Sets max value that can pass through the RELU\n";
		cout << "   LEAKY_RELU_CONST=<const>   Sets the constant for LEAKY_RELU\n";
		cout << "   l2Lambda=<lambda>          Sets the lambda for L2 Regularization\n";
		cout << "   MOMENT_CONST=<const>       Sets the constant decay for the Nesterov Momentum\n";
		cout << "   MAX_NORM_CAP=<cap>         Sets the max value a weight can be\n";
		return 0;
	}

	time_t starttime, endtime;

	string outputName;
	bool saveNewWeights = false;
	int epochs = -1;
	int device = -1;
	string testSetName;
	bool haveTest = false;
	int haveTrainMethod = 0;
	int trainMethod = TRAIN_AS_IS;
	double learningRate = -1; //-1s are sentinel values
	double reluCap = -1;
	double leaky = -1;
	double l2 = -1;
	double moment = -1;
	double maxNorm = -1;
	bool preprocessIndividual = false;
	bool preprocessCollective = false;


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
				epochs = stoi(arg.substr(arg.find("=")+1));
			else if(arg.find("device=") != string::npos)
				device = stoi(arg.substr(arg.find("=")+1));
			else if(arg.find("testSet=") != string::npos)
			{
				testSetName = arg.substr(arg.find("=") + 1);
				haveTest = true;
			}
			else if(arg.find("-train_equal_prop") != string::npos)
			{
				trainMethod = TRAIN_EQUAL_PROP;
				haveTrainMethod++;
			}
			else if(arg.find("-train_as_is") != string::npos)
				haveTrainMethod++; // this is on by default
			else if(arg.find("-preprocessCollective") != string::npos)
				preprocessCollective = true;
			else if(arg.find("-preprocessIndividual") != string::npos)
				preprocessIndividual = true;
			else if(arg.find("learningRate=") != string::npos)
				learningRate = stod(arg.substr(arg.find("=")+1));
			else if(arg.find("RELU_CAP=") != string::npos)
				reluCap = stod(arg.substr(arg.find("=")+1));
			else if(arg.find("LEAKY_RELU_CONST=") != string::npos)
				leaky = stod(arg.substr(arg.find("=")+1));
			else if(arg.find("l2Lambda=") != string::npos)
				l2 = stod(arg.substr(arg.find("=")+1));
			else if(arg.find("MOMENT_CONST=") != string::npos)
				moment = stod(arg.substr(arg.find("=")+1));
			else if(arg.find("MAX_NORM_CAP=") != string::npos)
				maxNorm = stod(arg.substr(arg.find("=")+1));
			else
			{
				printf("Unknown arg %s. Aborting.\n", argv[i]);
				return 0;
			}
		}
	}

	if(haveTrainMethod > 1)
	{
		printf("You cannot have multiple training methods simultaneously.\n");
		return 0;
	}

	if(preprocessIndividual && preprocessCollective)
	{
		printf("You can only have one preprocessing method.\n");
		return 0;
	}

	if(!saveNewWeights)
	{
		printf("If you continue no weights will be saved. Would you like to continue? (y/n)\n");
		char cont = getchar();
		if(cont != 'y' && cont != 'Y')
			return 0;
	}

	//set up net
	Net net(argv[1]);
	if(learningRate != -1)
    	net.set_learningRate(learningRate);
    if(reluCap != -1)
    	net.set_RELU_CAP(reluCap);
    if(leaky != -1)
    	net.set_LEAKY_RELU_CONST(leaky);
    if(l2 != -1)
    	net.set_l2Lambda(l2);
    if(moment != -1)
    	net.set_MOMENT_CONST(moment);
    if(maxNorm != -1)
    	net.set_MAX_NORM_CAP(maxNorm);
    if(device != -1)
    	net.setDevice(device);
    if(preprocessCollective)
    	net.preprocessCollectively();
    if(preprocessIndividual)
    	net.preprocessIndividually();
    if(saveNewWeights)
    	net.setSaveName(outputName);
    if(names.size() != 0)
    	net.setClassNames(names, trues);	
    net.setTrainingType(trainMethod);
	if(!net.finalize())
	{
		cout << net.getErrorLog() << endl;
		cout << "Net did not load correctly. Exiting." << endl;
		return 0;
	}

	printf("CNN Layer Dimensions\n");
	net.printLayerDims();
	

	//get images and add them
	vector<imVector> images(0);
	vector<double> trueVals(0);
	vector<string> names;
	vector<int> trues;
	short sizeByte, xSize, ySize, zSize;

	starttime = time(NULL);
	if (!convertBinaryToVector(argv[2], images, &trueVals, &names, &trues, sizeByte, xSize, ySize, zSize)) {
		cout << "Exiting." << endl;
		return 0;
	}
	endtime = time(NULL);
	cout << "Time to bring in training data: " << secondsToString(endtime - starttime) << endl;

	cout << "Adding training data" << endl;
	net.setTrainingData(images,trueVals);

	printf("Training Data Distribution:\n");
	net.printTrainingDistribution();
	printf("\n");

	//get test images if needed
	if(haveTest) {
		vector<imVector> testImages(0);
		vector<double> testTrueVals(0);

		cout << "Bringing in testing data from file: " << testSetName.c_str()) << endl;
		starttime = time(NULL);
		if (!convertBinaryToVectorTest(testSetName.c_str(), testImages, &testTrueVals, sizeByte, xSize, ySize, zSize)) {
			cout << "Exiting." << endl;
			return 0;
		}
		endtime = time(NULL);
		cout << "Time to bring in test data: " << secondsToString(endtime - starttime) << endl;

		net.addTestData(testImages, testTrueVals);

		cout << "Test Set Distribution:" << endl;
		net.printTestDistribution();
		cout << endl;
	}


	starttime = time(NULL);
	net.train(epochs);
	endtime = time(NULL);
	cout << "Time for OpenCL code: " << secondsToString(endtime - starttime) << ". - " << secondsToString((endtime-starttime)/(float)epochs) << " per epoch." << endl;

	// if(saveNewWeights)
	// {
	// 	net.save(outputName.c_str());
	// }
	// else
	// {
	// 	net.save(argv[1]);
	// }
}