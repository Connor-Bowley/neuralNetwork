/************************************
*
*	ConvNetTrainer
*
*	Created by Connor Bowley on 5/2/16
*
*	This program is for running trained CNNs that were trained using ConvNet.
*
*	Usage:
*		See usage statement in main.
*
*************************************/

#include "ConvNetCL.h"
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
	if(argc < 2)
	{
		cout << "Usage (Required to come first):\n   ./ConvNetTrainerCL binaryTrainingImagesFile";
		cout << "\nOptional arguments (must come after required args, everything before equals sign is case sensitive):\n";
		cout << "   cnn=<cnn_config.txt>       Sets the layout of the CNN to what is in the config. If not specified will use whatever CNN is compiled.\n";
		cout << "   outname=<outname.txt>      Sets the name for the outputted trained CNN. If not specified weights will not be saved.\n";
		cout << "   testSet=<name.txt>         A binary training file to be used as a test/validation set. Never trained on.\n";
		cout << "   epochs=<#epochs>           Number of epochs to train for. Defaults to \"How long it takes\"\n";
		cout << "   device=<device#>           Which OpenCL device on to use. Integer. Defaults to GPU supporting doubles if present, else defaults to CPU.\n";
		cout << "   -train_as_is               Causes CNN to train using all images for every epoch. On by default. Can only use one train method at a time\n";
		cout << "   -train_equal_prop          Causes CNN to train using equal amounts of each class for each epoch. For classes with larger amounts of images,\n";
		cout << "                                 the ones used will be randomly chosen each epoch. Can only use one train method at a time\n";
		cout << "   -preprocessIndividual      Preprocesses training data and test data individually by image. Not recommended.\n";
		cout << "   -preprocessCollective      Preprocesses training data collectively and preprocesses test data based on the training data. Default.\n";
		// cout << "   miniBatch=<int>            Sets the miniBatch size for training. Defaults to 1 (Stochastic gradient descent).\n";
		cout << "   learningRate=<rate>        Sets the learningRate for the CNN.\n";
		cout << "   RELU_CAP=<cap>             Sets max value that can pass through the RELU\n";
		cout << "   LEAKY_RELU_CONST=<const>   Sets the constant for LEAKY_RELU\n";
		cout << "   l2Lambda=<lambda>          Sets the lambda for L2 Regularization\n";
		cout << "   MOMENT_CONST=<const>       Sets the constant decay for the Nesterov Momentum\n";
		cout << "   MAX_NORM_CAP=<cap>         Sets the max value a weight can be\n";
		return 0;
	}

	time_t starttime,endtime;

	string outputName;
	bool saveWeights = false;
	string cnn_name = "";
	int epochs = -1;
	int device = -1;
	int haveTrainMethod = 0;
	int trainMethod = TRAIN_AS_IS;
	string testSetName;
	bool haveTest = false;
	double learningRate = -1; //-1s are sentinel values
	double reluCap = -1;
	double leaky = -1;
	double l2 = -1;
	double moment = -1;
	double maxNorm = -1;
	bool preprocessIndividual = false;
	bool preprocessCollective = false;

	int miniBatchSize = 1;

	if(argc > 2)
	{
		for(int i = 2; i < argc; i++)
		{
			string arg(argv[i]);
			if(arg.find("outname=") != string::npos)
			{
				outputName = arg.substr(arg.find("=") + 1);
				saveWeights = true;
			}
			else if(arg.find("cnn=") != string::npos)
				cnn_name = arg.substr(arg.find('=') + 1);
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
			// else if(arg.find("miniBatch=") != string::npos)
			// 	miniBatchSize = stoi(arg.substr(arg.find("=")+1));
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
				printf("Unknown arg \"%s\". Aborting.\n", argv[i]);
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

	if(miniBatchSize <= 0)
	{
		printf("MiniBatch size must be positive.\n");
		return 0;
	}

	if(!saveWeights)
	{
		printf("If you continue no weights will be saved. Would you like to continue? (y/n)\n");
		char cont = getchar();
		if(cont != 'y' && cont != 'Y')
			return 0;
	}

	vector<imVector> images(0);
	vector<double> trueVals(0);
	vector<string> names(0);
	vector<int> trues(0);
	short sizeByte, xSize, ySize, zSize;

	//get training images
	printf("Bringing in training data from file: %s\n", argv[1]);
	starttime = time(NULL);
	if (!convertBinaryToVector(argv[1], images, &trueVals, &names, &trues, sizeByte, xSize, ySize, zSize)) {
		cout << "Exiting." << endl;
		return 0;
	}
	endtime = time(NULL);
	cout << "Time to bring in training data: " << secondsToString(endtime - starttime) << endl;

	//set up net
	Net net;
	if(cnn_name != "")
	{
		bool success = net.load(cnn_name.c_str());
		if(!success)
			return 0;
	}
	else
	{
		net.init(xSize,ySize,zSize);

		//small 128x128x3
		// net.setActivType(LEAKY_RELU);
		// net.addConvLayer(6,1,5,0); //28x28x6
		// net.addMaxPoolLayer(2,2);  //14x14x6
		// net.addConvLayer(10,1,3,0);	//12x12x10
		// net.addMaxPoolLayer(3,3);   //4x4x10
		// net.addFullyConnectedLayer(2);

		//large 128x128x3
		// net.setActivType(LEAKY_RELU);	//128x128x3 
		// net.addConvLayer(32,1,3,1);		//128x128x32
		// net.addMaxPoolLayer(2,2);		//64x64x32 
		// net.addConvLayer(32,1,5,0);     //60x60x32
		// net.addMaxPoolLayer(2,2); 	    //30x30x32
		// net.addConvLayer(64,1,3,0);	  	//28x28x32
		// net.addMaxPoolLayer(2,2); 		//14x14x32
		// net.addConvLayer(128,1,3,0);	//12x12x64
		// net.addMaxPoolLayer(3,3);		//4x4x64
		// net.addFullyConnectedLayer(4);	//1x1x4	 	

		//64x64x3 net
		// net.setActivType(RELU);
		// net.addConvLayer(10,1,3,1);     //64x64x10
		// net.addMaxPoolLayer(2,2); 	    //32x32x10
		// net.addConvLayer(10,1,3,1);	    //32x32x10
		// net.addMaxPoolLayer(2,2);	    //16x16x10
		// net.addConvLayer(10,1,3,1);	    //16x16x10
		// net.addMaxPoolLayer(2,2);	    //8x8x10
		// net.addConvLayer(10,1,3,1);	    //8x8x10
		// net.addFullyConnectedLayer(10); //1x1x10
		// net.addFullyConnectedLayer(4);  //1x1x4

		/* shallow 64x64x3 net */
		// net.setActivType(LEAKY_RELU);		//64x64x3   //32x32x3
		// net.addConvLayer(20,1,5,0);     	//60x60x20	//28x28x20
		// net.addMaxPoolLayer(2,2); 	    	//30x30x20	//14x14x20
		// net.addConvLayer(20,1,3,0);	  	//28x28x20	//12x12x20
		// net.addMaxPoolLayer(2,2); 			//14x14x20	//6x6x20
		// net.addConvLayer(20,1,3,0);		//12x12x20	//4x4x20
		// net.addMaxPoolLayer(3,3);			//4x4x  20 	//fails for 32x32 start
		// net.addFullyConnectedLayer(4);		//1x1x4	 	//1x1x4
		
		///large net 32
		// net.setActivType(LEAKY_RELU);
		// net.addConvLayer(20, 1, 3, 1);  //32x32x20 //numfilters, stride, filtersize, padding
		// net.addConvLayer(20,1,3,1);		//32x32x10
		// net.addMaxPoolLayer(2,2);		//16x16x20
		// net.addConvLayer(30,1,3,1);		//16x16x30
		// net.addConvLayer(40,1,3,1);		//16x16x40
		// net.addMaxPoolLayer(2,2);		//8x8x40
		// net.addConvLayer(50,1,3,1);		//8x8x50
		// net.addMaxPoolLayer(2,2);		//4x4x50
		// net.addFullyConnectedLayer(128);//1x1x128
		// net.addFullyConnectedLayer(4);  //1x1x4
		//*/
		
		
		//small net 32 also 128
		net.setActivType(LEAKY_RELU);
		net.addConvLayer(6,1,5,0); //28x28x6
		net.addMaxPoolLayer(2,2);  //14x14x6
		//net.addConvLayer(7,1,3,1);
		net.addConvLayer(10,1,3,0);	//12x12x10
		net.addMaxPoolLayer(3,3);   //4x4x10
		//net.addConvLayer(5,1,3,1);	//4x4x5
		//net.addConvLayer(4,1,4,0);  //1x1x2
		net.addFullyConnectedLayer(4);

		//big small net 32
		// net.setActivType(LEAKY_RELU);
		// net.addConvLayer(20,1,5,0); //28x28x6
		// net.addMaxPoolLayer(2,2);  //14x14x6
		// net.addConvLayer(20,1,3,0);	//12x12x10
		// net.addMaxPoolLayer(3,3);   //4x4x10
		// net.addFullyConnectedLayer(1024);
		// net.addFullyConnectedLayer(3);
	}
    

	//set hyperparameters
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
    if(saveWeights)
    	net.setSaveName(outputName);
    if(names.size() != 0)
    	net.setClassNames(names,trues);
    net.setTrainingType(trainMethod);
	if(!net.finalize())
	{
		cout << net.getErrorLog() << endl;
		cout << "Something went wrong making the net. Exiting." << endl;
		return 0;
	}
	
	cout << "CNN Layer Sizes" << endl;
	net.printLayerDims();

	//add images to net
	net.addTrainingData(images, trueVals);
//    net.addTrainingDataShallow(dest,trueVals);
    
    cout << "Training Distribution:" << endl;
	net.printTrainingDistribution();
	cout << endl;

	if(haveTest)
	{
		vector<imVector> testImages(0);
		vector<double> testTrueVals(0);

		cout << "Bringing in testing data from file: " << testSetName.c_str() << endl;
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
	if(miniBatchSize == 1)
		net.train(epochs);
	else
		net.miniBatchTrain(miniBatchSize, epochs);
	endtime = time(NULL);
	cout << "Time for OpenCL code: " << secondsToString(endtime - starttime) << ".";
	if(epochs != -1)
		cout << " - " << secondsToString((endtime-starttime)/(float)epochs) << " per epoch." << endl;
	else
		cout << endl;


	//now train will save the weights
	// if(saveWeights)
	// {
	// 	net.save(outputName.c_str());
	// }
}