/*************************************************************************************************
 *
 *  ConvNet.cpp
 *  
 *
 *  Created by Connor Bowley on 3/15/16.
 *
 *	Classes
 *		Net
 *		Layer - Abstract
 *			ConvLayer
 *			MaxPoolLayer
 *			ActivLayer - does the activation function. Defaults to RELU.
 *			InputLayer
 *			FullyConnectedLayer??? - not currently implemented.
 *
 *	Each Layer needs forwardprop and backprop
 *		Can we build in local dneurons in forwardprop? No?
 *		Multiply(and add) by forward derivatives in backprop
 * 		
 *		each layer finishes the derivatives for the layer before it
 *		layers never need to know what layer comes after??? even in backprop??? I think so.
 *
 *
 *	If you want to add a new layer, in Net you need to update:
 *		add a new layer type in the .h file. static const int NEW_LAYER = 1+whatever the last one is
 *		search for that layer in the save and load functions. Prob make a getHyperparameters function
 *		make a new add function. bool addNewLayer(Layer&, hyperparameters needed)
 *
 *
 *
 *	Todo: Fix batch gradient descent?
 *
 *	Todo: fix miniBatchTrain?
 *
 *	Todo: implement a gradient check using numerical gradients.
 *
 * 	Todo: make a semi-shallow copy of net (and layers?) that has it's own neurons but all point to the 
 *		  same weights?
 *
 *	Todo: Check build status for each device and run accordingly
 *
 *************************************************************************************************/

#include "ConvNet.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>
#include <random>

#ifdef __APPLE__
 	#include "OpenCL/opencl.h"
#else
 	#include "CL/cl.h"
#endif

using namespace std;

/***********************************
 * Class Implementations
 ***********************************/

/**********************
 * Net
 **********************/

double Net::stepSize = 1e-3;

int Net::n_activationType = 0;

const double Net::GRADCHECK_H = .01;

bool Net::gradCheck = false;


void Net::debug()
{
	
}

Net::Net(const char* filename)
{
	load(filename);
	initOpenCL();
}

bool Net::isActive() const
{
	return (n_layers.size() > 0) && (n_layers.back()->getType() == Net::SOFTMAX_LAYER);
}

Net::Net(int inputWidth, int inputHeight, int inputDepth)
{
	init(inputWidth,inputHeight,inputDepth);
}

void Net::init(int inputWidth, int inputHeight, int inputDepth)
{
	n_hasConvLayer = false;
	n_hasMaxPoolLayer =false;
	n_hasRELULayer = false;
	n_hasLeakyRELULayer = false;
	n_hasSoftmax  = false;
	//Normal Net init stuff
	resize3DVector(n_blankVector,inputWidth,inputHeight,inputDepth);
	n_blankInput.setImage(&n_blankVector, &n_blankVector);
	n_layers.push_back(&n_blankInput);

	__device = -1;

	initOpenCL();
}

void Net::initOpenCL()
{
	cl_int error;
	//get num of platforms and we will use the first one
	platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);
	platformIds.resize(platformIdCount);
	clGetPlatformIDs(platformIdCount,platformIds.data(), nullptr);
	deviceIdCount = 0;
	clGetDeviceIDs(platformIds[0],CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

	deviceIds.resize(deviceIdCount);
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

	const cl_context_properties contextProperties[] = 
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties>(platformIds[0]),
		0,0
	};
	context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), 
	nullptr, nullptr, &error);
	CheckError(error);
}

Net::~Net()
{
	//do I need to run delete on the vectors in the layers????????
	Layer *point;
	
	for(int i=0; i< n_layers.size(); i++)
	{
		point = n_layers.back();
		n_layers.pop_back();
		delete point;
	}
	/*
	for(int i=0; i< n_trainingData.size(); i++)
	{
		delete n_trainingData[i];
	}
	*/
	clReleaseContext(context);
}

void Net::printLayerDims()
{
	for(int i=0; i < n_layers.size(); i++)
	{
		cout << n_layers[i]->getNeuronDims();
	}
}

void Net::forwardprop()
{
	for(int i=1; i< n_layers.size(); i++)
	{
		n_layers[i]->forwardprop(*n_layers[i-1]);
	}
}

void Net::backprop()
{
	for(int i= n_layers.size()-1; i > 0; i--)
	{
		n_layers[i]->backprop(*n_layers[i-1]);
	}
}

void Net::runTrainingData()
{
	int numCorrect = 0;
	SoftmaxLayer* soft = (SoftmaxLayer*)n_layers.back();
	//set the next training image as the InputLayer for the net
	for(int t=0; t< n_trainingData.size(); t++)
	{
		n_layers[0] = n_trainingData[t];

		//run forward pass
		forwardprop();

		//set trueVal
		soft->setTrueVal(n_trainingDataTrueVals[t]);

		int predictedClass = soft->getPredictedClass();
		if(predictedClass == n_trainingDataTrueVals[t])
			numCorrect++;
	}
	cout << "Run on Training data: " <<  "Accuracy: " << (double)numCorrect/n_trainingData.size()*100 << "%, " << numCorrect << " out of " << n_trainingData.size() << endl;
}

void Net::splitTrain(int epochs, bool useGPU)
{
	useGPU = false;

	if(n_trainingData.size() != n_trainingDataTrueVals.size())
	{
		cout << "All the training data needs true values for training" << endl;
		return;
	}

	string ep = to_string(epochs);

	int startValidationIndex = n_trainingData.size() * 0.9;

	gradCheck = false;
	//vector<vector<vector<double> > >& lastLayerGradients = n_layers.back()->getdNeurons();
	//setAll3DVector(lastLayerGradients,1);
	int numCorrect;
	SoftmaxLayer* soft = (SoftmaxLayer*)n_layers.back();
	for(int e=0; e< epochs; e++)
	{
		numCorrect = 0;
		//set the next training image as the InputLayer for the net
		for(int t=0; t< startValidationIndex; t++)
		{
			n_layers[0] = n_trainingData[t];

			//run forward pass
			forwardprop();

			//set trueVal
			soft->setTrueVal(n_trainingDataTrueVals[t]);

			int predictedClass = soft->getPredictedClass();
			if(predictedClass == n_trainingDataTrueVals[t])
				numCorrect++;

			//get prediction and see if we are right. add up the amount of rights and wrongs get get accuracy
			//and print for each epoch?

			//run backward pass beside the weight no
			backprop();

		}
		cout << "Epoch: " ;
		cout << setw(ep.size()) << e+1;
		cout << ", Accuracy: " << (double)numCorrect/(startValidationIndex)*100 << "%, " << numCorrect << " out of " << startValidationIndex << endl;

		shuffleTrainingData();
	}

	cout << "Running Validation set" << endl;
	numCorrect = 0;
	for(int t=startValidationIndex; t< n_trainingData.size(); t++)
	{
		n_layers[0] = n_trainingData[t];

		//run forward pass
		forwardprop();

		//set trueVal
		soft->setTrueVal(n_trainingDataTrueVals[t]);

		int predictedClass = soft->getPredictedClass();
		if(predictedClass == n_trainingDataTrueVals[t])
			numCorrect++;
	}
	cout << "Validation run on Training data: " <<  "Accuracy: " << (double)numCorrect/(n_trainingData.size()-startValidationIndex)*100 << "%, " << numCorrect << " out of " << (n_trainingData.size()-startValidationIndex) << endl;
	
}

void Net::train(int epochs)
{
	if(n_trainingData.size() != n_trainingDataTrueVals.size())
	{
		cout << "All the training data needs true values for training" << endl;
		return;
	}

	//set 1 in the d_neurons in the last layer
			//or do we need to set it to the error? -> I don't think so.
	string ep = to_string(epochs);

	gradCheck = false;
	//vector<vector<vector<double> > >& lastLayerGradients = n_layers.back()->getdNeurons();
	//setAll3DVector(lastLayerGradients,1);
	int numCorrect;
	SoftmaxLayer* soft = (SoftmaxLayer*)n_layers.back();
	for(int e=0; e< epochs; e++)
	{
		numCorrect = 0;
		//set the next training image as the InputLayer for the net
		for(int t=0; t< n_trainingData.size(); t++)
		{
			n_layers[0] = n_trainingData[t];

			//run forward pass
			forwardprop();

			//set trueVal
			soft->setTrueVal(n_trainingDataTrueVals[t]);

			int predictedClass = soft->getPredictedClass();
			if(predictedClass == n_trainingDataTrueVals[t])
				numCorrect++;

			//get prediction and see if we are right. add up the amount of rights and wrongs get get accuracy
			//and print for each epoch?

			//run backward pass beside the weight no
			backprop();
		}
		cout << "Epoch: " ;
		cout << setw(ep.size()) << e+1;
		cout << ", Accuracy: " << (double)numCorrect/n_trainingData.size()*100 << "%, " << numCorrect << " out of " << n_trainingData.size() << endl;

	}

}

void Net::miniBatchTrain(int epochs, int batchSize)
{
	if(n_trainingData.size() != n_trainingDataTrueVals.size())
	{
		cout << "All the training data needs true values for training" << endl;
		return;
	}

	string ep = to_string(epochs);
	int origBatchSize = batchSize;

	gradCheck = false;
	//vector<vector<vector<double> > >& lastLayerGradients = n_layers.back()->getdNeurons();
	//setAll3DVector(lastLayerGradients,1);
	int numCorrect;
	vector<double> errors(2);
	SoftmaxLayer* soft = (SoftmaxLayer*)n_layers.back();
	int numFullBatches = n_trainingData.size()/batchSize;
	int remain = n_trainingData.size() % batchSize;
	int curBatch = 0;
	for(int e=0; e< epochs; e++)
	{
		numCorrect = 0;
		batchSize = origBatchSize;
		//set the next training image as the InputLayer for the net
		for(int t=0; t< n_trainingData.size(); t+=batchSize)
		{
			if(curBatch >= numFullBatches)
				batchSize = remain;
			for(int b=0; b<batchSize; b++)
			{
				n_layers[0] = n_trainingData[t+b];

				//run forward pass
				forwardprop();

				//set trueVal
				soft->setTrueVal(n_trainingDataTrueVals[t]);

				//get error
				vector<double> curError = soft->getError();
				for(int i=0; i< curError.size(); i++)
				{
					errors[i] += curError[i];
				}

				int predictedClass = soft->getPredictedClass();
				//cout << "Pred: "<<predictedClass << " True: "<< n_trainingDataTrueVals[t] << "\n";
				if(predictedClass == n_trainingDataTrueVals[t])
					numCorrect++;
			}

			for(int i=0; i< errors.size(); i++)
			{
				errors[i] /= n_trainingData.size();
			}
			soft->setError(errors);
			backprop();
			for(int i=0; i< errors.size(); i++)
				errors[i] = 0;

		}
		cout << "Epoch: " ;
		cout << setw(ep.size()) << e+1;
		cout << ", Accuracy: " << (double)numCorrect/n_trainingData.size()*100 << "%, " << numCorrect << " out of " << n_trainingData.size() << endl;
	}
}

void Net::gradientCheck()
{
	gradCheck = true;
	SoftmaxLayer* soft = (SoftmaxLayer*)n_layers.back();
	for(int t=0; t< n_trainingData.size(); t++)
	{
		n_layers[0] = n_trainingData[t];

		forwardprop();

		soft->setTrueVal(n_trainingDataTrueVals[t]);

		backprop();
		cout << "Image "<<t << endl;
		soft->gradientCheck(*n_layers[n_layers.size()-2]);
	}
	

	//for layers other than softmax do
	//vector<...> prevNeurons = prevLayer.getNeurons
	//without putting an & before prevNeurons. then you can change it.
}

void Net::OpenCLTrain(int epochs, bool useGPU)
{
	if(n_layers.back()->getType() != Net::SOFTMAX_LAYER)
	{
		cout << "Last layer must be softmax to train. Aborting training." << endl;
		return;
	}

	cl_int error = CL_SUCCESS;

	vector<int> calculatedClasses(n_trainingData.size());

	//get num of platforms and we will use the first one
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);
	vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs(platformIdCount,platformIds.data(), nullptr);

	cl_uint deviceIdCount = 0;
	//cl_uint gpudeviceIdCount = 0;
	clGetDeviceIDs(platformIds[0],CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

	vector<cl_device_id> deviceIds(deviceIdCount);
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

	//decide which one to use.
	// if gpu available that can hold the mem, use it. 
	unsigned long memNeeded = getMem();
	int q = 0; //device to use
	if(__device != -1)
		q = __device;
	else if(useGPU)
	{
		for(int i = 1; i < deviceIdCount; i++)
		{
			cl_device_type type;
			CheckError(clGetDeviceInfo(deviceIds[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, nullptr));
			if(type == CL_DEVICE_TYPE_GPU)
			{
				cl_ulong memAmount;
				CheckError(clGetDeviceInfo(deviceIds[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memAmount, nullptr));
				if(memAmount > memNeeded + 200)
				{
					q = i;
				}
			}
		}
	}
	size_t valueSize;
	CheckError(clGetDeviceInfo(deviceIds[q], CL_DEVICE_NAME, 0, NULL, &valueSize));
	char* name = new char[valueSize];
	cout << "Using device " << q << ": ";
	CheckError(clGetDeviceInfo(deviceIds[q], CL_DEVICE_NAME, valueSize, name, nullptr));
	cout << name << endl;


	//make the context
	const cl_context_properties contextProperties[] = 
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties>(platformIds[0]),
		0,0
	};
	cl_context context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), 
	nullptr, nullptr, &error);
	CheckError(error);

	//build the program
	cl_program CNTraining = CreateProgram(LoadKernel("../kernels/ConvNetTraining_kernel.cl"), context);
	const char* options = "-cl-single-precision-constant";
	//cout << "Build Program " << clBuildProgram(CNTraining, gpudeviceIdCount, gpudeviceIds.data(), options, nullptr, nullptr) << endl;
	const cl_device_id* deviceToBuild = &(deviceIds[q]);
	CheckError(clBuildProgram(CNTraining, 1, deviceToBuild, options, nullptr, nullptr));
	cl_kernel reluKernel, leakyReluKernel, convKernel, maxPoolKernel, softmaxKernel, zeroPadKernel, reluBackKernel,
		zeroPadBackKernel, softmaxBackKernel, maxPoolBackKernel, leakyReluBackKernel, convBackNeuronsKernel, 
		convBackBiasesKernel, convBackWeightsKernel, copyArrayKernel, convBackWeightsMomentKernel;

	//Create the kernels

	reluKernel = clCreateKernel(CNTraining, "relu", &error);
	CheckError(error);
	reluBackKernel = clCreateKernel(CNTraining, "relu_back", &error);
	CheckError(error);

	leakyReluKernel = clCreateKernel(CNTraining, "leakyRelu", &error);
	CheckError(error);
	leakyReluBackKernel = clCreateKernel(CNTraining, "leakyRelu_back", &error);
	CheckError(error);

	convKernel = clCreateKernel(CNTraining, "convolve", &error);
	CheckError(error);
	convBackNeuronsKernel = clCreateKernel(CNTraining, "convolve_back_neurons", &error);
	CheckError(error);
	convBackBiasesKernel = clCreateKernel(CNTraining, "convolve_back_biases", &error);
	CheckError(error);
	convBackWeightsKernel = clCreateKernel(CNTraining, "convolve_back_weights", &error);
	CheckError(error);
	convBackWeightsMomentKernel = clCreateKernel(CNTraining, "convolve_back_weights_moment", &error);
	CheckError(error);

	zeroPadKernel = clCreateKernel(CNTraining, "zeroPad", &error);
	CheckError(error);
	zeroPadBackKernel = clCreateKernel(CNTraining, "zeroPad_back", &error);
	CheckError(error);

	maxPoolKernel = clCreateKernel(CNTraining, "maxPool", &error);
	CheckError(error);
	maxPoolBackKernel = clCreateKernel(CNTraining, "maxPool_back", &error);
	CheckError(error);

	softmaxKernel = clCreateKernel(CNTraining, "softmax", &error);
	CheckError(error);
	softmaxBackKernel = clCreateKernel(CNTraining, "softmax_back", &error);
	CheckError(error);

	copyArrayKernel = clCreateKernel(CNTraining, "copyArray", &error);
	CheckError(error);


	//cout << "Kernels created" << endl;
		
	//make the command queue
	cl_command_queue queue = clCreateCommandQueue(context, deviceIds[q], 0, &error);
		CheckError(error);

	//make a vector of pointers to kernels that is parallel to n_layers. Use this to make memory
	//on gpu for the weights and biases. get maxSize of every set of neurons and make two of
	//those on gpu
	int numConvLayers = 0;

	int maxNeuronSize = n_layers[0]->getNumNeurons();

	vector<cl_kernel*> layers(n_layers.size());
	layers[0] = nullptr; //because input layer
	vector<cl_mem> weights;
	vector<int> weightSizes;
	vector<cl_mem> biases;
	vector<int> biasSizes;
	vector<cl_mem> layerNeeds(n_layers.size() - 1); //minus softmax. layerNeeds[0] will be empty for input layer

	int maxWeightSize = 0;

	////////////////////////////////////////////////
	// get order of layers and weights and biases
	////////////////////////////////////////////////
	for(int i=1; i< n_layers.size(); i++)
	{
		int type = n_layers[i]->getType();
		if(type == Net::CONV_LAYER)
		{
			//cout << "conv " << numConvLayers << "i " << i << endl;
			//cout << "Conv" << endl;
			layers[i] = &convKernel;
			numConvLayers++;
			ConvLayer *conv = (ConvLayer*) n_layers[i];
			int numWeights = conv->getNumWeights();
			if(numWeights > maxWeightSize)
			{
				maxWeightSize = numWeights;
			}
			//cout << "get some weights" << endl;
			double* w = conv->getWeights();
			//cout << "weights got" << endl;
			weights.push_back(clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeof(double) * numWeights, w, &error));
			weightSizes.push_back(numWeights);
			CheckError(error);

			//get biases too
			int numBiases = conv->getNumBiases();
			double *b = conv->getBiases();
			biases.push_back(clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeof(double) * numBiases, b, &error));
			biasSizes.push_back(numBiases);
			CheckError(error);

			int maxSizeNeeded = conv->getMaxSizeNeeded();
			if(maxSizeNeeded > maxNeuronSize)
			{
				maxNeuronSize = maxSizeNeeded;
			}

			delete w;
			delete b;

			vector<int> hyper = conv->getKernelHyperParameters();

							//		    (prevWidth+ 2 * padding ) * (prevHeight+2 * padding ) * depth
			size_t paddedArraySize = (size_t) ((hyper[2] + 2 * hyper[5]) * (hyper[6] + 2 * hyper[5]) * hyper[3]);
			layerNeeds[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * paddedArraySize, nullptr, &error);

			//cout << "End Conv" << endl;
		}
		else if (type == Net::MAX_POOL_LAYER)
		{
			layers[i] = &maxPoolKernel;
			size_t dneuronInfoSize = n_layers[i]->getNumNeurons();
			layerNeeds[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * dneuronInfoSize, nullptr, &error);
		}
		else if (type == Net::SOFTMAX_LAYER)
		{
			layers[i] = &softmaxKernel;
		}
		else if (type == Net::ACTIV_LAYER)
		{
			ActivLayer* act = (ActivLayer*)n_layers[i];
			if(act->getActivationType() == ActivLayer::RELU)
				layers[i] = &reluKernel;
			else if (act->getActivationType() == ActivLayer::LEAKY_RELU)
				layers[i] = &leakyReluKernel;
			size_t dneuronInfoSize = n_layers[i]->getNumNeurons();
			layerNeeds[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * dneuronInfoSize, nullptr, &error);
		}
		else
		{
			cout << "Unknown layer type for the GPU implemenation. Defaulting to CPU." << endl;
			gradCheck = false;
			for(int r=0; r < n_trainingData.size(); r++)
			{
				n_layers[0] = n_trainingData[r];

				//run forward pass
				forwardprop();

				//get the results and save them into n_results
			}
			return;
		}
		if(n_layers[i]->getNumNeurons() > maxNeuronSize)
		{
			maxNeuronSize = n_layers[i]->getNumNeurons();
		}
	}

	//set up velocity stuff
	vector<cl_mem> velocities;
	double *zeroVels = new double[maxWeightSize];
	for(int i=0; i< maxWeightSize; i++)
		zeroVels[i] = 0.0;
	for(int l=0; l < n_layers.size(); l++)
	{
		if(n_layers[l]->getType() == Net::CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)n_layers[l];
			velocities.push_back(clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeof(double) * conv->getNumWeights(), zeroVels, &error));
			CheckError(error);
		}
	}
	delete zeroVels;
	zeroVels = NULL;

	
	cl_mem n = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * maxNeuronSize,
			nullptr, &error);
	CheckError(error);
	cl_mem *neurons = &n;
	cl_mem p = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * maxNeuronSize,
			nullptr, &error);
	CheckError(error);
	cl_mem *prevNeurons = &p;


	//double* testHold = new double[maxNeuronSize];

	//cout << "Max neuron size: " << maxNeuronSize << endl;
	
	int imageSize = ((InputLayer*)n_layers[0])->getImageSize();
	double* image = new double[imageSize];

	/**************************************
	 *
	 *	Start going through the images
	 *
	 * 	forwardprop and backprop
	 *
	 **************************************/
	int softSize = n_layers.back()->getNumNeurons();
    cout << "softSize = " << softSize << endl;
	vector<double> neur(softSize);
	vector<double> test(maxNeuronSize);
	cl_mem *temp;
	//cout << "Starting run on GPU(s)" << endl;
	string ep = to_string(epochs);
	for(int e=1; e <= epochs; e++)
	{
		shuffleTrainingData();

		if(e%5 == 0)
		{
			Net::stepSize *= .5;
			printf("Epoch %d: Changed learning rate from %e to %e\n",e,Net::stepSize*2,Net::stepSize);
		}
		
		for(int r=0; r < n_trainingData.size(); r++)
		{
			//set the new image(s)
			int startForThisRound = r;
			if(r < n_trainingData.size())
				n_trainingData[r]->getImage(image, imageSize);
			

			CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0,
					sizeof(double) * imageSize,
					image, 0, nullptr, nullptr));
			clFinish(queue);

			//run kernels from index = 1 to index = end - 1. special stuff needs to be done on 
			//data before softmax is run
			int curConvLayer = 0;
			// this is initialized for the case the first real layer is a non-padded conv layer and it needs
			// to save the neurons into the layerNeeds
			size_t globalWorkSize[] = {(size_t)n_layers[0]->getNumNeurons(),0,0};
			
			

			////////////////////////////////////
			// forwardprop
			////////////////////////////////////
			//cout << "forwardprop epoch: " << e << " image: " << r << endl;
			for(int i=1; i< layers.size() - 1; i++)
			{	
				//cout << "start for" << endl;
				if(layers[i] == &convKernel)
				{	/* just here to show what the hyper parameters are
					vector<int> hyper(7);
					hyper[0] = c_weights[0].size(); // filterSize
					hyper[1] = c_stride;
					hyper[2] = c_prevNeuronWidth;
					hyper[3] = c_prevNeuronDepth;
					hyper[4] = c_biases.size();  // num filters
					hyper[5] = c_padding;
					hyper[6] = c_prevNeuronHeight;
					*/

					//cout << "running convKernel " << curConvLayer << endl;
					vector<int> hyper = ((ConvLayer*)n_layers[i])->getKernelHyperParameters();

					if(hyper[5] != 0) //if padding != 0
					{
						//run the zeroPad Kernel
						clSetKernelArg(zeroPadKernel, 0, sizeof(cl_mem), prevNeurons);
						clSetKernelArg(zeroPadKernel, 1, sizeof(cl_mem), neurons);
						clSetKernelArg(zeroPadKernel, 2, sizeof(int), &(hyper[5])); // padding
						clSetKernelArg(zeroPadKernel, 3, sizeof(int), &(hyper[2])); // prevWidth
						clSetKernelArg(zeroPadKernel, 4, sizeof(int), &(hyper[6])); // prevHeight
						clSetKernelArg(zeroPadKernel, 5, sizeof(int), &(hyper[3])); // depth (before and after zero pad)

						// run it for the size of the new array
						//							  (prevWidth+ 2 * padding ) * (prevHeight+2 * padding ) * depth
						globalWorkSize[0] = (size_t) ((hyper[2] + 2 * hyper[5]) * (hyper[6] + 2 * hyper[5]) * hyper[3]);
						CheckError(clEnqueueNDRangeKernel(queue, zeroPadKernel, 1,
							nullptr,
							globalWorkSize,
							nullptr, 0, nullptr, nullptr));
						

						//swap the buffers so prevNeurons holds the zero padded data
						temp = neurons;
						neurons = prevNeurons;
						prevNeurons = temp;
						clFinish(queue);

						if(Net::walkthrough)
						{
							printf("pwid %d, phei %d, dep %d, pad %d, size %ld\n",hyper[2],hyper[6],hyper[3],hyper[5],globalWorkSize[0]);
							cout << "Forward Layer Padded " << i << endl;
							CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * globalWorkSize[0],
								test.data(), 0, nullptr, nullptr));
							printArray(test.data(), globalWorkSize[0]);
							getchar();
						}
					}

					//save the source array into the layerNeeds for this conv layer
					clSetKernelArg(copyArrayKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(copyArrayKernel, 1, sizeof(cl_mem), &(layerNeeds[i]));
					// global work size will either have been set by the zeroPadKernel or the previous layer if no pad
					// globalWorkSize is initialized so that if this convKernel is the first layer then it will work
					CheckError(clEnqueueNDRangeKernel(queue, copyArrayKernel, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					//clFinish(queue); shouldn't need to finish yet because convKernel doesn't need to use the copy

					if(Net::walkthrough)
					{
						clFinish(queue);
						cout << "conv layerNeeds " << curConvLayer << endl;
						CheckError(clEnqueueReadBuffer(queue, layerNeeds[i], CL_TRUE, 0, sizeof(double) * globalWorkSize[0],
							test.data(), 0, nullptr, nullptr));
						printArray(test.data(), globalWorkSize[0]);
						getchar();
					}

					
					clSetKernelArg(convKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(convKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(convKernel, 2, sizeof(cl_mem), &weights[curConvLayer]);
					clSetKernelArg(convKernel, 3, sizeof(cl_mem), &biases[curConvLayer]);
					
					int numFilters = hyper[4];
					int paddedWidth = hyper[2] + 2*hyper[5]; // adjust the width to account for padding
					clSetKernelArg(convKernel, 4, sizeof(int), &numFilters);
					clSetKernelArg(convKernel, 5, sizeof(int), &(hyper[0])); // filterSize
					clSetKernelArg(convKernel, 6, sizeof(int), &(hyper[1])); // stride
					clSetKernelArg(convKernel, 7, sizeof(int), &paddedWidth); // prevWidth
					clSetKernelArg(convKernel, 8, sizeof(int), &(hyper[3])); // prevDepth

					if(Net::walkthrough)
					{
						printf("Conv %d, numFil %d, filSize %d, str %d, padWid %d\n", 
							curConvLayer, numFilters, hyper[0], hyper[1], paddedWidth);
					}

					globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();
					
					CheckError(clEnqueueNDRangeKernel(queue, convKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
					curConvLayer++;
				}
				else if(layers[i] == &maxPoolKernel)
				{
					//cout << "running maxPoolKernel " << endl;
					vector<int> hyper = ((MaxPoolLayer*)n_layers[i])->getKernelHyperParameters();

					clSetKernelArg(maxPoolKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(maxPoolKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(maxPoolKernel, 2, sizeof(int), &(hyper[0]));
					clSetKernelArg(maxPoolKernel, 3, sizeof(int), &(hyper[1]));
					clSetKernelArg(maxPoolKernel, 4, sizeof(int), &(hyper[2]));
					clSetKernelArg(maxPoolKernel, 5, sizeof(int), &(hyper[3]));
					clSetKernelArg(maxPoolKernel, 6, sizeof(cl_mem), &(layerNeeds[i])); //maxIndexes
					
					globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();
					
					CheckError(clEnqueueNDRangeKernel(queue, maxPoolKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
				}
				else if(layers[i] == &reluKernel)
				{
					//cout << "running reluKernel " << endl;
					clSetKernelArg(reluKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(reluKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(reluKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo

					globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();
					
					CheckError(clEnqueueNDRangeKernel(queue, reluKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
				}
				else if(layers[i] == &leakyReluKernel)
				{
					//cout << "running leakyReluKernel " << endl;
					clSetKernelArg(leakyReluKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(leakyReluKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(leakyReluKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo

					globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();

					CheckError(clEnqueueNDRangeKernel(queue, leakyReluKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
				}
				clFinish(queue);
				if(Net::walkthrough)
				{
					cout << "Forward Layer " << i << endl;
					CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * n_layers[i]->getNumNeurons(),
						test.data(), 0, nullptr, nullptr));
					printArray(test.data(), n_layers[i]->getNumNeurons());
					getchar();
				}
				

				//swap prev and cur neuron pointers
				temp = neurons;
				neurons = prevNeurons;
				prevNeurons = temp;
			}

			//end of forward through normal layers. now softmax is all that is left
			CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * softSize, 
				neur.data(), 0, nullptr, nullptr));
			//clFinish(queue[q]);
			maxSubtraction(neur);
			double denom = vectorESum(neur);
			CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * softSize,
				neur.data(), 0, nullptr, nullptr));

			clFinish(queue);

			//set kernel args
			//cout << "running softmaxKernel " << r <<  endl;
			clSetKernelArg(softmaxKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(softmaxKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(softmaxKernel, 2, sizeof(double), &denom);
			globalWorkSize[0] = (size_t)softSize;
			CheckError(clEnqueueNDRangeKernel(queue, softmaxKernel, 1,
				nullptr,
				globalWorkSize,
				nullptr, 0, nullptr, nullptr));

			clFinish(queue);

			
			CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * softSize, 
				neur.data(), 0, nullptr, nullptr));
			calculatedClasses[startForThisRound++] = getMaxElementIndex(neur);

			if(Net::showErrors || Net::walkthrough)
			{
				cout << "max element was " << getMaxElementIndex(neur)  << ". True: " << n_trainingDataTrueVals[r] << endl;
				cout << "neur" << endl;
				printVector(neur);
				if(Net::walkthrough)
					getchar();
			}
			/*
			cout << "Forward softmax" << endl;
			printVector(neur);*/
			//cout << getMaxElementIndex(neur) << " | " << neur[getMaxElementIndex(neur)] << endl;


			////////////////////////////
			// start backprop
			////////////////////////////


			//cout << "backprop epoch: " << e << " image: " << r << endl;
			int trueVal = n_trainingDataTrueVals[r];
			clSetKernelArg(softmaxBackKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(softmaxBackKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(softmaxBackKernel, 2, sizeof(int), &trueVal);
			//globalWorkSize[0] = (size_t)softSize; // this is still true
			CheckError(clEnqueueNDRangeKernel(queue,softmaxBackKernel, 1,
				nullptr,
				globalWorkSize,
				nullptr, 0, nullptr, nullptr));
			clFinish(queue);
			
			if(Net::showErrors && !Net::walkthrough)
			{
				CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * softSize, 
					neur.data(), 0, nullptr, nullptr));
				cout << "Backprop Softmax" << endl;
				printVector(neur);
			}
			

			//swap prev and cur neuron pointers
			temp = neurons;
			neurons = prevNeurons;
			prevNeurons = temp;

			//prevNeurons has become prevdNeurons
			//neurons has become dneurons

			curConvLayer--;
			for(int i = n_layers.size() - 2; i >= 1; i--) // start at end layer beside softmax. End at first layer beside input
			{
				if(layers[i] == &reluKernel)
				{
					//cout << "running reluBackKernel " << endl;
					clSetKernelArg(reluBackKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(reluBackKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(reluBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo

					globalWorkSize[0] = (size_t)n_layers[i-1]->getNumNeurons();
					
					CheckError(clEnqueueNDRangeKernel(queue, reluBackKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
				}
				else if(layers[i] == &leakyReluKernel)
				{
					//cout << "running leakyReluBackKernel " << endl;
					clSetKernelArg(leakyReluBackKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(leakyReluBackKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(leakyReluBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i])); // dneuronInfo

					globalWorkSize[0] = (size_t)n_layers[i-1]->getNumNeurons();

					CheckError(clEnqueueNDRangeKernel(queue, leakyReluBackKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
				}
				else if(layers[i] ==  &maxPoolKernel)
				{
					//cout << "running maxPoolBackKernel " << endl;
					int numIndexes = n_layers[i]->getNumNeurons();
					clSetKernelArg(maxPoolBackKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(maxPoolBackKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(maxPoolBackKernel, 2, sizeof(cl_mem), &(layerNeeds[i]));
					clSetKernelArg(maxPoolBackKernel, 3, sizeof(int), &numIndexes);

					globalWorkSize[0] = (size_t)n_layers[i-1]->getNumNeurons();

					CheckError(clEnqueueNDRangeKernel(queue, maxPoolBackKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
				}
				else if(layers[i] == &convKernel)
				{
					//cout << "running convKernel back " << endl;
					/* just here to show what the hyper parameters are
					vector<int> hyper(7);
					hyper[0] = c_weights[0].size(); // filterSize
					hyper[1] = c_stride;
					hyper[2] = c_prevNeuronWidth;
					hyper[3] = c_prevNeuronDepth;
					hyper[4] = c_biases.size();  // num filters
					hyper[5] = c_padding;
					hyper[6] = c_prevNeuronHeight;
					*/

					//backprop neurons					
					//cout << "running convBackNeuronsKernel" << endl;
					ConvLayer* conv = (ConvLayer*)n_layers[i];
					vector<int> hyper = conv->getKernelHyperParameters();
					int paddedWidth = hyper[2] + 2*hyper[5]; // adjust the width to account for padding
					
					clSetKernelArg(convBackNeuronsKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(convBackNeuronsKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(convBackNeuronsKernel, 2, sizeof(cl_mem), &(weights[curConvLayer]));
					clSetKernelArg(convBackNeuronsKernel, 3, sizeof(int), &(hyper[4])); //numFilters
					clSetKernelArg(convBackNeuronsKernel, 4, sizeof(int), &(hyper[0])); //filterSize
					clSetKernelArg(convBackNeuronsKernel, 5, sizeof(int), &(hyper[1])); //stride
					clSetKernelArg(convBackNeuronsKernel, 6, sizeof(int), &paddedWidth);
					clSetKernelArg(convBackNeuronsKernel, 7, sizeof(int), &(hyper[3])); //depth

					//cout << "padded width " << paddedWidth << " numFilters " << hyper[4] << " filterSize " << hyper[0] << endl;

												//(prevWidth+ 2 * padding ) * (prevHeight+2 * padding ) * depth
					globalWorkSize[0] = (size_t) ((hyper[2] + 2 * hyper[5]) * (hyper[6] + 2 * hyper[5]) * hyper[3]);
					
					CheckError(clEnqueueNDRangeKernel(queue, convBackNeuronsKernel, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

					clFinish(queue); //MUST finish before weights start getting updated

					if(Net::walkthrough)
					{
						//clFinish(queue);
						cout << "conv layerNeeds back" << curConvLayer << endl;
						CheckError(clEnqueueReadBuffer(queue, layerNeeds[i], CL_TRUE, 0, sizeof(double) * globalWorkSize[0],
							test.data(), 0, nullptr, nullptr));
						printArray(test.data(), globalWorkSize[0]);
						getchar();
					}

					//backprop biases
					
					int sizeOfNeurons = n_layers[i]->getNumNeurons();
					double mystepSize = Net::stepSize;// * i;
					//cout << "running convBackBiasesKernel " << curConvLayer << ": " << sizeOfNeurons << "/" << hyper[4] << endl;
					clSetKernelArg(convBackBiasesKernel, 0, sizeof(cl_mem), &(biases[curConvLayer]));
					clSetKernelArg(convBackBiasesKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(convBackBiasesKernel, 2, sizeof(int), &sizeOfNeurons);
					clSetKernelArg(convBackBiasesKernel, 3, sizeof(int), &(hyper[4])); // numFilters = dneuronsDepth
					clSetKernelArg(convBackBiasesKernel, 4, sizeof(double), &(mystepSize));

					globalWorkSize[0] = (size_t) hyper[4];//numFilters = numBiases
					CheckError(clEnqueueNDRangeKernel(queue, convBackBiasesKernel, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));


					if(Net::walkthrough)
					{
						clFinish(queue);
						cout << "ConvLayer back biases " << curConvLayer << endl;
						CheckError(clEnqueueReadBuffer(queue, biases[curConvLayer], CL_TRUE, 0, sizeof(double) * globalWorkSize[0],
							test.data(), 0, nullptr, nullptr));
						printArray(test.data(), globalWorkSize[0]);
						getchar();
					}
					
					//backprop weights
					//cout << "stepSize host " << Net::stepSize << endl;
					
					//cout << "running convBackWeightsKernel" << endl;
					if(!Net::useMomentum) // no momentum
					{
						clSetKernelArg(convBackWeightsKernel, 0, sizeof(cl_mem), &(weights[curConvLayer]));
						clSetKernelArg(convBackWeightsKernel, 1, sizeof(cl_mem), &(layerNeeds[i]));
						clSetKernelArg(convBackWeightsKernel, 2, sizeof(cl_mem), neurons);
						clSetKernelArg(convBackWeightsKernel, 3, sizeof(int), &(hyper[3])); // depth
						clSetKernelArg(convBackWeightsKernel, 4, sizeof(int), &(hyper[1])); // stride
						clSetKernelArg(convBackWeightsKernel, 5, sizeof(int), &paddedWidth);
						clSetKernelArg(convBackWeightsKernel, 6, sizeof(int), &(hyper[0])); // filterSize
						clSetKernelArg(convBackWeightsKernel, 7, sizeof(int), &(hyper[4])); // numFilters
						clSetKernelArg(convBackWeightsKernel, 8, sizeof(double), &(mystepSize));

						globalWorkSize[0] = (size_t)conv->getNumWeights();
						//cout << "starting kernel " << endl;
						CheckError(clEnqueueNDRangeKernel(queue, convBackWeightsKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					}
					else // with momentum
					{
						clSetKernelArg(convBackWeightsMomentKernel, 0, sizeof(cl_mem), &(weights[curConvLayer]));
						clSetKernelArg(convBackWeightsMomentKernel, 1, sizeof(cl_mem), &(layerNeeds[i]));
						clSetKernelArg(convBackWeightsMomentKernel, 2, sizeof(cl_mem), neurons);
						clSetKernelArg(convBackWeightsMomentKernel, 3, sizeof(int), &(hyper[3])); // depth
						clSetKernelArg(convBackWeightsMomentKernel, 4, sizeof(int), &(hyper[1])); // stride
						clSetKernelArg(convBackWeightsMomentKernel, 5, sizeof(int), &paddedWidth);
						clSetKernelArg(convBackWeightsMomentKernel, 6, sizeof(int), &(hyper[0])); // filterSize
						clSetKernelArg(convBackWeightsMomentKernel, 7, sizeof(int), &(hyper[4])); // numFilters
						clSetKernelArg(convBackWeightsMomentKernel, 8, sizeof(double), &(mystepSize));
						clSetKernelArg(convBackWeightsMomentKernel, 9, sizeof(cl_mem), &(velocities[curConvLayer]));

						globalWorkSize[0] = (size_t)conv->getNumWeights();
						//cout << "starting kernel " << endl;
						CheckError(clEnqueueNDRangeKernel(queue, convBackWeightsMomentKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					}

					if(Net::walkthrough)
					{
						clFinish(queue);
						cout << "ConvLayer back weights " << curConvLayer << endl;
						cout << "numWeights " << conv->getNumWeights() << endl;
						CheckError(clEnqueueReadBuffer(queue, weights[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->getNumWeights(),
							test.data(), 0, nullptr, nullptr));
						printArray(test.data(), conv->getNumWeights());
						getchar();
					}

					//backprop the padding if necessary
					if(hyper[5] != 0)
					{
						globalWorkSize[0] = (size_t)conv->getPaddedNeuronSize();
						if(Net::walkthrough)
						{
							printf("pwid %d, phei %d, dep %d, pad %d, size %ld\n",hyper[2],hyper[6],hyper[3],hyper[5],globalWorkSize[0]);
							cout << "Back Layer Padded " << i << " ConvLayer " << curConvLayer << endl;
							CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * globalWorkSize[0],
								test.data(), 0, nullptr, nullptr));
							printArray(test.data(), globalWorkSize[0]);
							getchar();
						}

						//cout << "Running zeroPadBackKernel" << endl;
						clFinish(queue); //so it finishes before zeroPadBackKernel starts changing prevNeurons and neurons

						//swap prev and cur neuron pointers

						temp = neurons;
						neurons = prevNeurons;
						prevNeurons = temp;

						clSetKernelArg(zeroPadBackKernel, 0, sizeof(cl_mem), prevNeurons);
						clSetKernelArg(zeroPadBackKernel, 1, sizeof(cl_mem), neurons);
						clSetKernelArg(zeroPadBackKernel, 2, sizeof(int), &(hyper[5])); //padding
						clSetKernelArg(zeroPadBackKernel, 3, sizeof(int), &(hyper[2])); //prevWidth (non-padded)
						clSetKernelArg(zeroPadBackKernel, 4, sizeof(int), &(hyper[6])); //prevHeight(non-padded)
						clSetKernelArg(zeroPadBackKernel, 5, sizeof(int), &(hyper[3])); //depth

						
						CheckError(clEnqueueNDRangeKernel(queue,zeroPadBackKernel, 1,
							nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

						if(Net::walkthrough)
						{
							clFinish(queue);
							cout << "Backprop Layer post de-pad " << i << " conv " << curConvLayer << endl;
							CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * n_layers[i-1]->getNumNeurons(),
								test.data(), 0, nullptr, nullptr));
							printArray(test.data(), n_layers[i-1]->getNumNeurons());
							getchar();
						}
					}
					curConvLayer--;
				}

				//cout << "before finish" << endl;
				clFinish(queue);
				//cout << "after finish" << endl;
				if(Net::walkthrough)
				{
					cout << "Backprop Layer " << i << endl;
					CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * n_layers[i]->getNumNeurons(),
						test.data(), 0, nullptr, nullptr));
					printArray(test.data(), n_layers[i]->getNumNeurons());
					getchar();
				}

				//swap prev and cur neuron pointers
				temp = neurons;
				neurons = prevNeurons;
				prevNeurons = temp;
			}
		}

		//cout << "calculatedClasses" << endl;
		if(calculatedClasses.size() == n_trainingDataTrueVals.size())
		{
			int numCorrect = 0;
			for(int i=0; i< calculatedClasses.size(); i++)
			{
				if(calculatedClasses[i] == n_trainingDataTrueVals[i])
					numCorrect++;
			}
			cout << "Epoch: " ;
			cout << setw(ep.size()) << e;
			cout << ", Accuracy on training data run: " << numCorrect << " out of " << n_trainingDataTrueVals.size() << ". " << numCorrect/(double)calculatedClasses.size()*100 << "%" << endl;
		}
	}

	delete image;
	double* tempWeight = new double[maxWeightSize];
	int curConvLayer = 0;
	for(int i = 1; i < n_layers.size() - 1; i++)
	{
		if(n_layers[i]->getType() == Net::CONV_LAYER)
		{
			//cout << "Setting weights for cur " << curConvLayer << " layer " << i << endl;
			ConvLayer* conv = (ConvLayer*)n_layers[i];

			CheckError(clEnqueueReadBuffer(queue, weights[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->getNumWeights(),
				tempWeight, 0, nullptr, nullptr));

			conv->setWeights(tempWeight);

			CheckError(clEnqueueReadBuffer(queue, biases[curConvLayer], CL_TRUE, 0, sizeof(double) * conv->getNumBiases(),
				tempWeight, 0, nullptr, nullptr));	

			conv->setBiases(tempWeight);

			curConvLayer++;	
		}

	}


	clReleaseCommandQueue(queue);
	clReleaseMemObject(*neurons);
	clReleaseMemObject(*prevNeurons);
	for(int w = 0; w < weights.size(); w++)
	{
		clReleaseMemObject(weights[w]);
		clReleaseMemObject(biases[w]);
	}
	for(int n = 0; n < layerNeeds.size(); n++)
	{
		clReleaseMemObject(layerNeeds[n]);
	}

	clReleaseKernel(convKernel);
	clReleaseKernel(zeroPadKernel);
	clReleaseKernel(maxPoolKernel);
	clReleaseKernel(reluKernel);
	clReleaseKernel(leakyReluKernel);
	clReleaseKernel(softmaxKernel);

	clReleaseKernel(convBackNeuronsKernel);
	clReleaseKernel(convBackBiasesKernel);
	clReleaseKernel(convBackWeightsKernel);
	clReleaseKernel(zeroPadBackKernel);
	clReleaseKernel(maxPoolBackKernel);
	clReleaseKernel(reluBackKernel);
	clReleaseKernel(leakyReluBackKernel);
	clReleaseKernel(softmaxBackKernel);

	clReleaseProgram(CNTraining);

	clReleaseContext(context);
}

bool Net::setDevice(unsigned int device)
{
	if(device >= deviceIds.size())
		return false;
	cl_uint canDouble;
	CheckError(clGetDeviceInfo(deviceIds[device], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &canDouble, nullptr));
	if(canDouble == 0)
		return false;
	__device = device;
	return true;
}

void Net::newRun(vector<int>& calculatedClasses, bool useGPU)
{
	//cout << "start newrun" << endl;
	cl_int error = CL_SUCCESS;
	calculatedClasses.resize(n_trainingData.size());
	n_confidences.resize(n_trainingData.size());
	/*
	//get num of platforms and we will use the first one
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);
	vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs(platformIdCount,platformIds.data(), nullptr);
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platformIds[0],CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

	vector<cl_device_id> deviceIds(deviceIdCount);
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);
	*/
	//decide which one to use.
	// if gpu available that can hold the mem, use it. 
	unsigned long forwardMemNeeded = getMemForward();
	int q = 0;
	if(__device != -1)
		q = __device;
	 //device to use
	else if(useGPU)
	{
		for(int i = 1; i < deviceIdCount; i++)
		{
			cl_device_type type;
			CheckError(clGetDeviceInfo(deviceIds[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, nullptr));
			if(type == CL_DEVICE_TYPE_GPU)
			{
				cl_ulong memAmount;
				CheckError(clGetDeviceInfo(deviceIds[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memAmount, nullptr));
				if(memAmount > forwardMemNeeded)
				{
					q = i;
				}
			}
		}
	}
	size_t valueSize;
	CheckError(clGetDeviceInfo(deviceIds[q], CL_DEVICE_NAME, 0, NULL, &valueSize));
	char* name = new char[valueSize];
	//cout << "Using device " << q << ": ";
	CheckError(clGetDeviceInfo(deviceIds[q], CL_DEVICE_NAME, valueSize, name, nullptr));
	//cout << name << endl;

	//make the context
	/*
	const cl_context_properties contextProperties[] = 
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties>(platformIds[0]),
		0,0
	};
	cl_context context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), 
	nullptr, nullptr, &error);
	CheckError(error);
	*/

	//build the program
	cl_program CNForward = CreateProgram(LoadKernel("../kernels/ConvNetForward_kernel.cl"), context);
	//cl_program CNForwardT = CreateProgram(LoadKernel("../kernels/ConvNetTraining_kernel.cl"), context);
	//const char* options = "-cl-single-precision-constant";
	//cout << "Build Program " << clBuildProgram(CNForward, gpudeviceIdCount, gpudeviceIds.data(), options, nullptr, nullptr) << endl;
	const cl_device_id* deviceToBuild = &(deviceIds[q]);
	CheckError(clBuildProgram(CNForward, 1, deviceToBuild, nullptr, nullptr, nullptr));
	//CheckError(clBuildProgram(CNForwardT, 1, deviceToBuild, options, nullptr, nullptr));
	cl_kernel reluKernel, leakyReluKernel, convKernel, maxPoolKernel, softmaxKernel, zeroPadKernel;

	//Create the kernels

	reluKernel = clCreateKernel(CNForward, "relu", &error);
	CheckError(error);

	leakyReluKernel = clCreateKernel(CNForward, "leakyRelu", &error);
	CheckError(error);

	convKernel = clCreateKernel(CNForward, "convolve", &error);
	CheckError(error);
	zeroPadKernel = clCreateKernel(CNForward, "zeroPad", &error);
	CheckError(error);

	maxPoolKernel = clCreateKernel(CNForward, "maxPool", &error);
	CheckError(error);

	softmaxKernel = clCreateKernel(CNForward, "softmax", &error);
	CheckError(error);


		
	//make the command queue
	cl_command_queue queue = clCreateCommandQueue(context, deviceIds[q], 0, &error);
	CheckError(error);

	//make a vector of pointers to kernels that is parallel to n_layers. Use this to make memory
	//on gpu for the weights and biases. get maxSize of every set of neurons and make two of
	//those on gpu
	int numConvLayers = 0;

	int maxNeuronSize = n_layers[0]->getNumNeurons();


	vector<cl_kernel*> layers(n_layers.size());
	layers[0] = nullptr; //because input layer
	vector<cl_mem> weights;
	vector<int> weightSizes;
	vector<cl_mem> biases;
	vector<int> biasSizes;

	for(int i=1; i< n_layers.size(); i++)
	{
		int type = n_layers[i]->getType();
		if(type == Net::CONV_LAYER)
		{
			//cout << "conv " << numConvLayers << "i " << i << endl;
			//cout << "Conv" << endl;
			layers[i] = &convKernel;
			numConvLayers++;
			ConvLayer *conv = (ConvLayer*) n_layers[i];
			int numWeights = conv->getNumWeights();
			//cout << "get some weights" << endl;
			double* w = conv->getWeights();
			//cout << "weights got" << endl;
			weights.push_back(clCreateBuffer(context, /*CL_MEM_READ_ONLY |*/ CL_MEM_COPY_HOST_PTR,
				sizeof(double) * numWeights, w, &error));
			weightSizes.push_back(numWeights);
			CheckError(error);

			//get biases too
			int numBiases = conv->getNumBiases();
			double *b = conv->getBiases();
			biases.push_back(clCreateBuffer(context, /*CL_MEM_READ_ONLY |*/ CL_MEM_COPY_HOST_PTR,
				sizeof(double) * numBiases, b, &error));
			biasSizes.push_back(numBiases);
			CheckError(error);

			int maxSizeNeeded = conv->getMaxSizeNeeded();
			if(maxSizeNeeded > maxNeuronSize)
			{
				maxNeuronSize = maxSizeNeeded;
			}

			delete w;
			delete b;
			//cout << "End Conv" << endl;
		}
		else if (type == Net::MAX_POOL_LAYER)
			layers[i] = &maxPoolKernel;
		else if (type == Net::SOFTMAX_LAYER)
			layers[i] = &softmaxKernel;
		else if (type == Net::ACTIV_LAYER)
		{
			ActivLayer* act = (ActivLayer*)n_layers[i];
			if(act->getActivationType() == ActivLayer::RELU)
				layers[i] = &reluKernel;
			else if (act->getActivationType() == ActivLayer::LEAKY_RELU)
				layers[i] = &leakyReluKernel;
		}
		else
		{
			cout << "Unknown layer type for the GPU implemenation. Defaulting to CPU." << endl;
			gradCheck = false;
			for(int r=0; r < n_trainingData.size(); r++)
			{
				n_layers[0] = n_trainingData[r];

				//run forward pass
				forwardprop();

				//get the results and save them into n_results
			}
			return;
		}
		if(n_layers[i]->getNumNeurons() > maxNeuronSize)
		{
			maxNeuronSize = n_layers[i]->getNumNeurons();
		}
	}

	
	cl_mem n = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * maxNeuronSize,
			nullptr, &error);
	CheckError(error);
	cl_mem *neurons = &n;
	cl_mem p = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * maxNeuronSize,
			nullptr, &error);
	CheckError(error);
	cl_mem *prevNeurons = &p;

	//double* testHold = new double[maxNeuronSize];

	
	int imageSize = ((InputLayer*)n_layers[0])->getImageSize();
	double* image = new double[imageSize];

	/**************************************
	 *
	 *	Start going through the images
	 *
	 **************************************/
	int softSize = n_layers.back()->getNumNeurons();
	vector<double> neur(softSize);
	vector<double> test(maxNeuronSize);

	//cout << "Starting going through the images" << endl;
	for(int r=0; r < n_trainingData.size(); r++)
	{
		//cout << "image " << r << endl;
		//set the new image(s)
		int startForThisRound = r;
		if(r < n_trainingData.size())
			n_trainingData[r]->getImage(image, imageSize);		

		CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0,
				sizeof(double) * imageSize,
				image, 0, nullptr, nullptr));
		clFinish(queue);

		//run kernels from index = 1 to index = end - 1. special stuff needs to be done on 
		//data before softmax is run
		int curConvLayer = 0;
		size_t globalWorkSize[] = {0,0,0};
		cl_mem *temp;
		
		for(int i=1; i< layers.size() - 1; i++)
		{	
			//cout << "start for" << endl;
			if(layers[i] == &convKernel)
			{	/*
				vector<int> hyper(7);
				hyper[0] = c_weights[0].size(); // filterSize
				hyper[1] = c_stride;
				hyper[2] = c_prevNeuronWidth;
				hyper[3] = c_prevNeuronDepth;
				hyper[4] = c_biases.size();  // num filters
				hyper[5] = c_padding;
				hyper[6] = c_prevNeuronHeight;
				*/

				//cout << "running convKernel " << curConvLayer << endl;
				vector<int> hyper = ((ConvLayer*)n_layers[i])->getKernelHyperParameters();

				if(hyper[5] != 0) //if padding != 0
				{
					//cout << "running zeroPad" << endl;
					//run the zeroPad Kernel
					clSetKernelArg(zeroPadKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(zeroPadKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(zeroPadKernel, 2, sizeof(int), &(hyper[5])); // padding
					clSetKernelArg(zeroPadKernel, 3, sizeof(int), &(hyper[2])); // prevWidth
					clSetKernelArg(zeroPadKernel, 4, sizeof(int), &(hyper[6])); // prevHeight
					clSetKernelArg(zeroPadKernel, 5, sizeof(int), &(hyper[3])); // depth (before and after zero pad)

					// run it for the size of the new array
					//							  (prevWidth+ 2 * padding ) * (prevHeight+2 * padding ) * depth
					globalWorkSize[0] = (size_t) ((hyper[2] + 2 * hyper[5]) * (hyper[6] + 2 * hyper[5]) * hyper[3]);
					CheckError(clEnqueueNDRangeKernel(queue, zeroPadKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
					clFinish(queue);

					//swap the buffers so prevNeurons holds the zero padded data
					temp = neurons;
					neurons = prevNeurons;
					prevNeurons = temp;

					//cout << "end zero pad" << endl;
				}
				
				clSetKernelArg(convKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(convKernel, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(convKernel, 2, sizeof(cl_mem), &weights[curConvLayer]);
				clSetKernelArg(convKernel, 3, sizeof(cl_mem), &biases[curConvLayer]);
				int numFilters = hyper[4];
				hyper[2] += 2*hyper[5]; // adjust the width to account for padding
				clSetKernelArg(convKernel, 4, sizeof(int), &numFilters);
				clSetKernelArg(convKernel, 5, sizeof(int), &(hyper[0])); // filterSize
				clSetKernelArg(convKernel, 6, sizeof(int), &(hyper[1])); // stride
				clSetKernelArg(convKernel, 7, sizeof(int), &(hyper[2])); // prevWidth
				clSetKernelArg(convKernel, 8, sizeof(int), &(hyper[3])); // prevDepth

				globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();
				
				//cout << "start kernel" << endl;
				CheckError(clEnqueueNDRangeKernel(queue, convKernel, 1,
					nullptr,
					globalWorkSize,
					nullptr, 0, nullptr, nullptr));
				curConvLayer++;
				//cout << "end kernel" << endl;
			}
			else if(layers[i] == &maxPoolKernel)
			{
				//cout << "running maxPoolKernel " << endl;
				vector<int> hyper = ((MaxPoolLayer*)n_layers[i])->getKernelHyperParameters();

				clSetKernelArg(maxPoolKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(maxPoolKernel, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(maxPoolKernel, 2, sizeof(int), &(hyper[0]));
				clSetKernelArg(maxPoolKernel, 3, sizeof(int), &(hyper[1]));
				clSetKernelArg(maxPoolKernel, 4, sizeof(int), &(hyper[2]));
				clSetKernelArg(maxPoolKernel, 5, sizeof(int), &(hyper[3]));
				
				globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();
				
				CheckError(clEnqueueNDRangeKernel(queue, maxPoolKernel, 1,
					nullptr,
					globalWorkSize,
					nullptr, 0, nullptr, nullptr));
			}
			else if(layers[i] == &reluKernel)
			{
				//cout << "running reluKernel " << endl;
				clSetKernelArg(reluKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(reluKernel, 1, sizeof(cl_mem), neurons);

				globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();
				
				CheckError(clEnqueueNDRangeKernel(queue, reluKernel, 1,
					nullptr,
					globalWorkSize,
					nullptr, 0, nullptr, nullptr));
			}
			else if(layers[i] == &leakyReluKernel)
			{
				//cout << "running leakyReluKernel " << endl;
				clSetKernelArg(leakyReluKernel, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(leakyReluKernel, 1, sizeof(cl_mem), neurons);

				globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();

				CheckError(clEnqueueNDRangeKernel(queue, leakyReluKernel, 1,
					nullptr,
					globalWorkSize,
					nullptr, 0, nullptr, nullptr));
			}
			clFinish(queue);

			if(Net::walkthrough)
			{
				cout << "Layer " << i << endl;
				CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * n_layers[i]->getNumNeurons(),
					test.data(), 0, nullptr, nullptr));
				printArray(test.data(), n_layers[i]->getNumNeurons());
				getchar();
			}

			//swap prev and cur neuron pointers
			temp = neurons;
			neurons = prevNeurons;
			prevNeurons = temp;
		}

		//end of forward through normal layers. now softmax is all that is left
		CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * softSize, 
			neur.data(), 0, nullptr, nullptr));
		//clFinish(queue[q]);
		maxSubtraction(neur);
		double denom = vectorESum(neur);
		CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * softSize,
			neur.data(), 0, nullptr, nullptr));

		clFinish(queue);

		//set kernel args
		//cout << "running softmaxKernel " << r <<  endl;
		clSetKernelArg(softmaxKernel, 0, sizeof(cl_mem), prevNeurons);
		clSetKernelArg(softmaxKernel, 1, sizeof(cl_mem), neurons);
		clSetKernelArg(softmaxKernel, 2, sizeof(double), &denom);
		globalWorkSize[0] = (size_t)softSize;
		CheckError(clEnqueueNDRangeKernel(queue, softmaxKernel, 1,
			nullptr,
			globalWorkSize,
			nullptr, 0, nullptr, nullptr));

		clFinish(queue);

		CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * softSize, 
			neur.data(), 0, nullptr, nullptr));
		calculatedClasses[startForThisRound] = getMaxElementIndex(neur);

		//cout << "confidences " << n_trainingData.size() << " " << n_confidences.size() << endl;
		//printVector(neur);
		n_confidences[startForThisRound] = neur;
		startForThisRound++;
		//cout << getMaxElementIndex(neur) << " | " << neur[getMaxElementIndex(neur)] << endl;;
	}

	delete image;

	if(calculatedClasses.size() == n_trainingDataTrueVals.size())
	{
		int numCorrect = 0;
		int errorCounts[softSize];
		for(int i=0; i < softSize; i++)
			errorCounts[i] = 0;
		for(int i=0; i< calculatedClasses.size(); i++)
		{
			if(calculatedClasses[i] == n_trainingDataTrueVals[i])
				numCorrect++;
			else
				errorCounts[(int)n_trainingDataTrueVals[i]]++;

		}
		cout << "Accuracy on training data run: " << numCorrect << " out of " << n_trainingDataTrueVals.size() << ". " << numCorrect/(double)calculatedClasses.size()*100 << "%" << endl;

		for(int i=0; i < softSize; i++)
		{
			printf("Errors when trueVal was %d: %d\n",i,errorCounts[i]);
		}
	}/*
	else
	{
		
		for(int i=0; i < calculatedClasses.size(); i++)
		{
			cout << calculatedClasses[i] << endl;
		}

	}*/

	clReleaseCommandQueue(queue);
	clReleaseMemObject(*neurons);
	clReleaseMemObject(*prevNeurons);
	for(int w = 0; w < weights.size(); w++)
	{
		clReleaseMemObject(weights[w]);
		clReleaseMemObject(biases[w]);
	}
	//weights.resize(0);
	//biases.resize(0);

	clReleaseKernel(convKernel);
	clReleaseKernel(zeroPadKernel);
	clReleaseKernel(maxPoolKernel);
	clReleaseKernel(reluKernel);
	clReleaseKernel(leakyReluKernel);
	clReleaseKernel(softmaxKernel);
	clReleaseProgram(CNForward);

	//clReleaseContext(context);

	//cout << "end newRun" << endl;
}

void Net::getConfidences(vector<vector<double> >& confidences) const
{
	confidences = n_confidences;
}

int Net::getNumCategories() const
{
	return n_numCategories;
}

void Net::run(bool useGPU) // run only goes forward and will be on the GPU if possible.
{
	cl_int error = CL_SUCCESS;

	vector<int> calculatedClasses(n_trainingData.size());
	//get num of platforms and we will use the first one
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);
	vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs(platformIdCount,platformIds.data(), nullptr);

	cl_uint gpudeviceIdCount = 0;
	clGetDeviceIDs(platformIds[0],CL_DEVICE_TYPE_GPU, 0, nullptr, &gpudeviceIdCount);

	//if we have gpus, use them. else run forward like normal.
	if(gpudeviceIdCount > 0 && useGPU)
	{
		cout << "GPU(s) found. Running program on GPU" << endl;
		vector<cl_device_id> gpudeviceIds(gpudeviceIdCount);
		clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_GPU, gpudeviceIdCount, gpudeviceIds.data(), nullptr);

		//use all the gpus?
		//make the context
		const cl_context_properties contextProperties[] = 
		{
			CL_CONTEXT_PLATFORM,
			reinterpret_cast<cl_context_properties>(platformIds[0]),
			0,0
		};
		cl_context context = clCreateContext(contextProperties, gpudeviceIdCount, gpudeviceIds.data(), 
		nullptr, nullptr, &error);
		CheckError(error);

		//build the program
		cl_program CNForward = CreateProgram(LoadKernel("../kernels/ConvNetForward_kernel.cl"), context);
		const char* options = "-cl-single-precision-constant";
		//cout << "Build Program " << clBuildProgram(CNForward, gpudeviceIdCount, gpudeviceIds.data(), options, nullptr, nullptr) << endl;
		CheckError(clBuildProgram(CNForward, gpudeviceIdCount, gpudeviceIds.data(), options, nullptr, nullptr));
		cl_kernel reluKernel, leakyReluKernel, convKernel, maxPoolKernel, softmaxKernel, zeroPadKernel;

		//Create the kernels

		reluKernel = clCreateKernel(CNForward, "relu", &error);
		CheckError(error);

		leakyReluKernel = clCreateKernel(CNForward, "leakyRelu", &error);
		CheckError(error);

		convKernel = clCreateKernel(CNForward, "convolve", &error);
		CheckError(error);
		zeroPadKernel = clCreateKernel(CNForward, "zeroPad", &error);
		CheckError(error);

		maxPoolKernel = clCreateKernel(CNForward, "maxPool", &error);
		CheckError(error);

		softmaxKernel = clCreateKernel(CNForward, "softmax", &error);
		CheckError(error);


		//cout << "Kernels created" << endl;
			
		//make the command queue
		int q = 1;
		cl_command_queue queue = clCreateCommandQueue(context, gpudeviceIds[q], 0, &error);
			CheckError(error);

		//make a vector of pointers to kernels that is parallel to n_layers. Use this to make memory
		//on gpu for the weights and biases. get maxSize of every set of neurons and make two of
		//those on gpu
		int numConvLayers = 0;

		int maxNeuronSize = n_layers[0]->getNumNeurons();

		vector<cl_kernel*> layers(n_layers.size());
		layers[0] = nullptr; //because input layer
		vector<cl_mem> weights;
		vector<int> weightSizes;
		vector<cl_mem> biases;
		vector<int> biasSizes;
		for(int i=1; i< n_layers.size(); i++)
		{
			int type = n_layers[i]->getType();
			if(type == Net::CONV_LAYER)
			{
				//cout << "conv " << numConvLayers << "i " << i << endl;
				//cout << "Conv" << endl;
				layers[i] = &convKernel;
				numConvLayers++;
				ConvLayer *conv = (ConvLayer*) n_layers[i];
				int numWeights = conv->getNumWeights();
				//cout << "get some weights" << endl;
				double* w = conv->getWeights();
				//cout << "weights got" << endl;
				weights.push_back(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					sizeof(double) * numWeights, w, &error));
				weightSizes.push_back(numWeights);
				CheckError(error);

				//get biases too
				int numBiases = conv->getNumBiases();
				double *b = conv->getBiases();
				biases.push_back(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					sizeof(double) * numBiases, b, &error));
				biasSizes.push_back(numBiases);
				CheckError(error);

				int maxSizeNeeded = conv->getMaxSizeNeeded();
				if(maxSizeNeeded > maxNeuronSize)
				{
					maxNeuronSize = maxSizeNeeded;
				}

				delete w;
				delete b;
				//cout << "End Conv" << endl;
			}
			else if (type == Net::MAX_POOL_LAYER)
				layers[i] = &maxPoolKernel;
			else if (type == Net::SOFTMAX_LAYER)
				layers[i] = &softmaxKernel;
			else if (type == Net::ACTIV_LAYER)
			{
				ActivLayer* act = (ActivLayer*)n_layers[i];
				if(act->getActivationType() == ActivLayer::RELU)
					layers[i] = &reluKernel;
				else if (act->getActivationType() == ActivLayer::LEAKY_RELU)
					layers[i] = &leakyReluKernel;
			}
			else
			{
				cout << "Unknown layer type for the GPU implemenation. Defaulting to CPU." << endl;
				gradCheck = false;
				for(int r=0; r < n_trainingData.size(); r++)
				{
					n_layers[0] = n_trainingData[r];

					//run forward pass
					forwardprop();

					//get the results and save them into n_results
				}
				return;
			}
			if(n_layers[i]->getNumNeurons() > maxNeuronSize)
			{
				maxNeuronSize = n_layers[i]->getNumNeurons();
			}
		}

		//cout << "GPU Layers initialized" << endl;
		
		/*
		cl_mem n = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * maxNeuronSize,
			nullptr, &error);
		CheckError(error);
		cl_mem pn = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * maxNeuronSize,
			nullptr, &error);
		CheckError(error);
		*/
		cl_mem n = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * maxNeuronSize,
				nullptr, &error);
		CheckError(error);
		cl_mem *neurons = &n;
		cl_mem p = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * maxNeuronSize,
				nullptr, &error);
		CheckError(error);
		cl_mem *prevNeurons = &p;

		//double* testHold = new double[maxNeuronSize];

		//cout << "Max neuron size: " << maxNeuronSize << endl;
		
		int imageSize = ((InputLayer*)n_layers[0])->getImageSize();
		double* image = new double[imageSize];

		/**************************************
		 *
		 *	Start going through the images
		 *
		 **************************************/
		int softSize = n_layers.back()->getNumNeurons();
		vector<double> neur(softSize);
		//cout << "Starting run on GPU(s)" << endl;
		for(int r=0; r < n_trainingData.size(); r++)
		{
			//set the new image(s)
			int startForThisRound = r;
			if(r < n_trainingData.size())
				n_trainingData[r]->getImage(image, imageSize);
			

			//printArray(image, imageSize);

			//cout << "copy image(s) into GPU" << endl;
			CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0,
					sizeof(double) * imageSize,
					image, 0, nullptr, nullptr));
			clFinish(queue);

			/*
			cout  << "trying convKernel" << endl;

			double tweights[] = {1,0,1,-1,0,0,-1,1,0,0,1,0,1,0,0,1,-1,-1,-1,0,0,1,1,-1,-1,1,0,0,1,-1,-1,-1,-1,-1,0,-1,1,1,-1,0,0,-1,0,1,-1,1,-1,1,1,1,-1,-1,0,1};
			double tbiases[] = {1,0};
			cl_mem tw = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * 54,
				tweights, &error);
			cl_mem tb = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * 2,
				tbiases, &error);
			CheckError(error);


			double testArray[] = {
				0,1,2, 1,2,1, 0,0,0, 2,2,2, 2,0,1,
				2,2,1, 1,0,0, 2,1,1, 2,2,0, 1,2,0,
				1,1,0, 1,0,1, 2,2,0, 0,1,0, 2,2,2,
				1,1,0, 0,2,1, 0,1,0, 1,2,2, 2,0,2,
				2,0,0, 2,0,2, 0,1,1, 2,1,0, 2,2,0};
			cl_mem ta = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * 5*5*3,
				testArray, &error);
			CheckError(error);
			int stride = 2;
			int filterSize = 3;
			int pwid = 5;
			int pdep = 3;

			cout << "setting args" << endl;


			clSetKernelArg(convKernel, 0, sizeof(cl_mem), &ta);
			clSetKernelArg(convKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(convKernel, 2, sizeof(cl_mem), &tw);
			clSetKernelArg(convKernel, 3, sizeof(cl_mem), &tb);
			int numFilters = 2;
			clSetKernelArg(convKernel, 4, sizeof(int), &numFilters);
			clSetKernelArg(convKernel, 5, sizeof(int), &filterSize);
			clSetKernelArg(convKernel, 6, sizeof(int), &stride);
			clSetKernelArg(convKernel, 7, sizeof(int), &pwid);
			clSetKernelArg(convKernel, 8, sizeof(int), &pdep);
			cout << "args set" << endl;
			const size_t globalWorkSiz[] = {8,0,0};
			CheckError(clEnqueueNDRangeKernel(queue, convKernel, 1,
				nullptr,
				globalWorkSiz,
				nullptr, 0, nullptr, nullptr));

			cout << "convKernel done" << endl;

			clEnqueueReadBuffer(queue, *neurons, CL_TRUE, 0, sizeof(double) * maxNeuronSize,
						testHold, 0, nullptr, nullptr);
					//cout << testHold[0] << endl;
			printArray(testHold, 18);
			
			return;
			*/

			//run kernels from index = 1 to index = end - 1. special stuff needs to be done on 
			//data before softmax is run
			int curConvLayer = 0;
			size_t globalWorkSize[] = {0,0,0};
			cl_mem *temp;
			
			for(int i=1; i< layers.size() - 1; i++)
			{	
				//cout << "start for" << endl;
				if(layers[i] == &convKernel)
				{	/*
					vector<int> hyper(7);
					hyper[0] = c_weights[0].size(); // filterSize
					hyper[1] = c_stride;
					hyper[2] = c_prevNeuronWidth;
					hyper[3] = c_prevNeuronDepth;
					hyper[4] = c_biases.size();  // num filters
					hyper[5] = c_padding;
					hyper[6] = c_prevNeuronHeight;
					*/

					//cout << "running convKernel " << curConvLayer << endl;
					vector<int> hyper = ((ConvLayer*)n_layers[i])->getKernelHyperParameters();

					if(hyper[5] != 0) //if padding != 0
					{
						//run the zeroPad Kernel
						clSetKernelArg(zeroPadKernel, 0, sizeof(cl_mem), prevNeurons);
						clSetKernelArg(zeroPadKernel, 1, sizeof(cl_mem), neurons);
						clSetKernelArg(zeroPadKernel, 2, sizeof(int), &(hyper[5])); // padding
						clSetKernelArg(zeroPadKernel, 3, sizeof(int), &(hyper[2])); // prevWidth
						clSetKernelArg(zeroPadKernel, 4, sizeof(int), &(hyper[6])); // prevHeight
						clSetKernelArg(zeroPadKernel, 5, sizeof(int), &(hyper[3])); // depth (before and after zero pad)

						// run it for the size of the new array
						//							  (prevWidth+ 2 * padding ) * (prevHeight+2 * padding ) * depth
						globalWorkSize[0] = (size_t) ((hyper[2] + 2 * hyper[5]) * (hyper[6] + 2 * hyper[5]) * hyper[3]);
						CheckError(clEnqueueNDRangeKernel(queue, zeroPadKernel, 1,
							nullptr,
							globalWorkSize,
							nullptr, 0, nullptr, nullptr));
						clFinish(queue);

						//swap the buffers so prevNeurons holds the zero padded data
						temp = neurons;
						neurons = prevNeurons;
						prevNeurons = temp;

					}
					
					clSetKernelArg(convKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(convKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(convKernel, 2, sizeof(cl_mem), &weights[curConvLayer]);
					clSetKernelArg(convKernel, 3, sizeof(cl_mem), &biases[curConvLayer]);
					
					int numFilters = hyper[4];
					int paddedWidth =  hyper[2] + 2 * hyper[5];
					clSetKernelArg(convKernel, 4, sizeof(int), &numFilters);
					clSetKernelArg(convKernel, 5, sizeof(int), &(hyper[0])); // filterSize
					clSetKernelArg(convKernel, 6, sizeof(int), &(hyper[1])); // stride
					clSetKernelArg(convKernel, 7, sizeof(int), &paddedWidth); // prevWidth
					clSetKernelArg(convKernel, 8, sizeof(int), &(hyper[3])); // prevDepth

					globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();

					CheckError(clEnqueueNDRangeKernel(queue, convKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
					curConvLayer++;
				}
				else if(layers[i] == &maxPoolKernel)
				{
					//cout << "running maxPoolKernel " << endl;
					vector<int> hyper = ((MaxPoolLayer*)n_layers[i])->getKernelHyperParameters();

					clSetKernelArg(maxPoolKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(maxPoolKernel, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(maxPoolKernel, 2, sizeof(int), &(hyper[0]));
					clSetKernelArg(maxPoolKernel, 3, sizeof(int), &(hyper[1]));
					clSetKernelArg(maxPoolKernel, 4, sizeof(int), &(hyper[2]));
					clSetKernelArg(maxPoolKernel, 5, sizeof(int), &(hyper[3]));
					
					globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();
					
					CheckError(clEnqueueNDRangeKernel(queue, maxPoolKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
				}
				else if(layers[i] == &reluKernel)
				{
					//cout << "running reluKernel " << endl;
					clSetKernelArg(reluKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(reluKernel, 1, sizeof(cl_mem), neurons);

					globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();
					
					CheckError(clEnqueueNDRangeKernel(queue, reluKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
				}
				else if(layers[i] == &leakyReluKernel)
				{
					//cout << "running leakyReluKernel " << endl;
					clSetKernelArg(leakyReluKernel, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(leakyReluKernel, 1, sizeof(cl_mem), neurons);

					globalWorkSize[0] = (size_t)n_layers[i]->getNumNeurons();

					CheckError(clEnqueueNDRangeKernel(queue, leakyReluKernel, 1,
						nullptr,
						globalWorkSize,
						nullptr, 0, nullptr, nullptr));
				}
				clFinish(queue);

				/*
				if(i == 3)
				{
					clEnqueueReadBuffer(queue, *neurons, CL_TRUE, 0, sizeof(double) * maxNeuronSize,
						testHold, 0, nullptr, nullptr);
					cout << testHold[0] << endl;

					printArray(testHold, globalWorkSize[0]);
					cout << "From layer " << i << endl;
					return;
				}
				*/
				//swap prev and cur neuron pointers
				temp = neurons;
				neurons = prevNeurons;
				prevNeurons = temp;
			}

			//cout << "starting softmaxKernel" <<  endl;
			//end of forward through normal layers. now softmax is all that is left
			// the current neurons are in prevNeurons due to the pointer swap
			
			CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * softSize, 
				neur.data(), 0, nullptr, nullptr));
			//clFinish(queue[q]);
			maxSubtraction(neur);
			double denom = vectorESum(neur);
			CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * softSize,
				neur.data(), 0, nullptr, nullptr));

			clFinish(queue);

			//set kernel args
			//cout << "running softmaxKernel " << r <<  endl;
			clSetKernelArg(softmaxKernel, 0, sizeof(cl_mem), prevNeurons);
			clSetKernelArg(softmaxKernel, 1, sizeof(cl_mem), neurons);
			clSetKernelArg(softmaxKernel, 2, sizeof(double), &denom);
			globalWorkSize[0] = (size_t)softSize;
			CheckError(clEnqueueNDRangeKernel(queue, softmaxKernel, 1,
				nullptr,
				globalWorkSize,
				nullptr, 0, nullptr, nullptr));


			clFinish(queue);


			CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * softSize, 
				neur.data(), 0, nullptr, nullptr));
			calculatedClasses[startForThisRound++] = getMaxElementIndex(neur);
			//cout << getMaxElementIndex(neur) << " | " << neur[getMaxElementIndex(neur)] << endl;;
		}

		delete image;

		if(calculatedClasses.size() == n_trainingDataTrueVals.size())
		{
			int numCorrect = 0;
			for(int i=0; i< calculatedClasses.size(); i++)
			{
				if(calculatedClasses[i] == n_trainingDataTrueVals[i])
					numCorrect++;
			}
			cout << "Accuracy on training data run: " << numCorrect << " out of " << n_trainingDataTrueVals.size() << ". " << numCorrect/(double)calculatedClasses.size()*100 << "%" << endl;
		}
		else
		{
			for(int i=0; i < calculatedClasses.size(); i++)
			{
				cout << calculatedClasses[i] << endl;
			}
		}

		clReleaseCommandQueue(queue);
		clReleaseMemObject(*neurons);
		clReleaseMemObject(*prevNeurons);
		for(int w = 0; w < weights.size(); w++)
		{
			clReleaseMemObject(weights[w]);
			clReleaseMemObject(biases[w]);
		}

		clReleaseKernel(convKernel);
		clReleaseKernel(zeroPadKernel);
		clReleaseKernel(maxPoolKernel);
		clReleaseKernel(reluKernel);
		clReleaseKernel(leakyReluKernel);
		clReleaseKernel(softmaxKernel);
		clReleaseProgram(CNForward);

		//clReleaseContext(context);


	}
	else // if no gpus
	{
		//CPU implemenation
		gradCheck = false;
		//vector<vector<vector<double> > >& lastLayerGradients = n_layers.back()->getdNeurons();
		//setAll3DVector(lastLayerGradients,1);
		int numCorrect;
		SoftmaxLayer* soft = (SoftmaxLayer*)n_layers.back();
		numCorrect = 0;
		//set the next training image as the InputLayer for the net
		for(int t=0; t< n_trainingData.size(); t++)
		{
			n_layers[0] = n_trainingData[t];

			//run forward pass
			forwardprop();

			//set trueVal only needed for backprop
			//soft->setTrueVal(n_trainingDataTrueVals[t]);

			int predictedClass = soft->getPredictedClass();
			if(predictedClass == n_trainingDataTrueVals[t])
				numCorrect++;

			//get prediction and see if we are right. add up the amount of rights and wrongs get get accuracy
			//and print for each epoch?

		}
		cout << "Accuracy: " << (double)numCorrect/n_trainingData.size()*100 << "%, " << numCorrect << " out of " << n_trainingData.size() << endl;

	}
}

int Net::numLayers() 
{
	return n_layers.size()-1; // subtract one because the input layer doesn't count as a real layer.
}

bool Net::addActivLayer()
{
	try
	{
		ActivLayer *activ = new ActivLayer(*n_layers.back(),n_activationType);
		n_layers.push_back(activ);
		if(n_activationType == ActivLayer::RELU)
		{
			n_hasRELULayer = true;
		}
		else if(n_activationType == ActivLayer::LEAKY_RELU)
		{
			n_hasLeakyRELULayer = true;
		}
		return true;
	}
	catch(...)
	{
		return false;
	}
}

bool Net::addActivLayer(int activationType)
{
	try
	{
		ActivLayer *activ = new ActivLayer(*n_layers.back(),activationType);
		n_layers.push_back(activ);
		if(activationType == ActivLayer::RELU)
		{
			n_hasRELULayer = true;
		}
		else if(activationType == ActivLayer::LEAKY_RELU)
		{
			n_hasLeakyRELULayer = true;
		}
		return true;
	}
	catch(...)
	{
		return false;
	}
}

bool Net::addConvLayer(int numFilters, int stride, int filterSize, int pad)
{
	try
	{
		ConvLayer *conv = new ConvLayer(*(n_layers.back()),numFilters,stride,filterSize,pad);
		n_layers.push_back(conv);
		n_hasConvLayer = true;
		return true;
	}
	catch(...)
	{
		return false;
	}
}

bool Net::addConvLayer(int numFilters, int stride, int filterSize, int pad, string weightsAndBiases)
{
	try
	{
		ConvLayer *conv = new ConvLayer(*(n_layers.back()),numFilters,stride,filterSize,pad, weightsAndBiases);
		n_layers.push_back(conv);
		n_hasConvLayer = true;
		return true;
	}
	catch(...)
	{
		cout << "in catch" << endl;
		return false;
	}
}
/*
bool Net::addFullyConnectedLayer(int outputSize)
{
	int num
	return Net::addConvLayer(outputSize,)
}
*/


bool Net::addMaxPoolLayer(int poolSize, int stride)
{
	try
	{
		MaxPoolLayer *pool = new MaxPoolLayer(*(n_layers.back()),poolSize, stride);
		n_layers.push_back(pool);
		n_hasMaxPoolLayer = true;
		return true;
	}
	catch(...)
	{
		return false;
	}
}

bool Net::addSoftmaxLayer()
{
	try
	{
		SoftmaxLayer *soft = new SoftmaxLayer(*(n_layers.back()));
		n_layers.push_back(soft);
		n_hasSoftmax = true;
		n_numCategories = soft->getNumNeurons();
		return true;
	}
	catch(...)
	{
		return false;
	}
}

//The true value will be the index of the correct class
void Net::addTrainingData(const vector<vector<vector<vector<double> > > >& trainingData, vector<double>& trueVals)
{
	const vector<vector<vector<vector<double> > > >& t = trainingData;	
	for(int n=0; n< t.size(); n++)
	{
		InputLayer *in = new InputLayer(t[n],&n_blankVector);
		n_trainingData.push_back(in);
		n_trainingDataTrueVals.push_back(trueVals[n]);
	}
}

void Net::addRealData(const vector<vector<vector<vector<double> > > >& realData)
{
	const vector<vector<vector<vector<double> > > >& r = realData;
	n_results.resize(n_results.size() + r.size());	
	for(int n=0; n< r.size(); n++)
	{
		InputLayer *in = new InputLayer(r[n],&n_blankVector);
		n_trainingData.push_back(in);
	}
}

void Net::setData(const vector<vector<vector<vector<double> > > >& data)
{
	n_results.clear(); n_results.resize(0);
	n_confidences.clear(); n_confidences.resize(0);
	n_trainingData.clear(); n_trainingData.resize(0);

	n_results.resize(data.size());
	n_confidences.resize(data.size());
	for(int n=0; n < data.size(); n++)
	{
		InputLayer *in = new InputLayer(data[n],&n_blankVector);
		n_trainingData.push_back(in);
	}
}

void Net::clear()
{
	n_layers.clear();
}

vector<vector<vector<double> > >* Net::getBlankVectorPointer()
{
	return &n_blankVector;
}

bool Net::setActivType(int activationType)
{
	if(0 <= activationType && activationType < ActivLayer::NUM_ACTIV_TYPES)
	{
		n_activationType = activationType;
		return true;
	}
	return false;
}

double Net::calcLoss(int indexOfTrueVal)
{
	// call softmax on the last layer of input data. The last layer should go down to a vector that is
	// 1x1xn or 1xnx1 or nx1x1
	vector<double> normedPredictions;
	softmax(n_layers.back()->getNeurons(), normedPredictions); // gets out predictions
	return -log(normedPredictions[indexOfTrueVal]);

}

int Net::getPredictedClass()
{
	vector<double> normedPredictions;
	softmax(n_layers.back()->getNeurons(), normedPredictions);
	int maxLoc = 0;
	for(int i=1; i<normedPredictions.size();i++)
	{
		if(normedPredictions[i] > normedPredictions[maxLoc])
		{
			maxLoc = i;
		}
	}
	//cout << maxLoc << endl;
	return maxLoc;
}

void Net::shuffleTrainingData(int times)
{
	if(times < 1)
		return;
	default_random_engine gen(time(0));
	uniform_int_distribution<int> distr(0,n_trainingData.size()-1);
	if(times == 1)
		cout << "Shuffling training images 1 time... ";
	else
		cout << "Shuffling training images " << times << " times... ";
	for(int t=0; t< times; t++)
	{
		for(int i=0; i< n_trainingData.size(); i++)
		{
			int swapIndex = distr(gen);
			InputLayer* temp  = n_trainingData[i];
			int tempTrue = n_trainingDataTrueVals[i];

			n_trainingData[i] 		  = n_trainingData[swapIndex];
			n_trainingDataTrueVals[i] = n_trainingDataTrueVals[swapIndex];

			n_trainingData[swapIndex] 		  = temp;
			n_trainingDataTrueVals[swapIndex] = tempTrue;
		}
	}
	cout << "Done" << endl;
}

bool Net::load(const char* filename)
{
	ifstream file;
	file.open(filename);
	string line;
	int loc;
	
	int lineNum = 0;

	if(!file.is_open())
	{
		cout << "File was unable to open" << endl;
		return false;
	}

	getline(file, line);
	if(line == "NET1.0")
	{
		int inputWidth, inputHeight, inputDepth, activationType;
		int netArgsFound = 0;
		getline(file, line); lineNum++;
		while(line != "END_NET")
		{
			if(line.find("activationType") != string::npos)
			{
				loc = line.find("=") + 1;
				activationType = stoi(line.substr(loc));
				netArgsFound++;
			}
			else if(line.find("inputWidth") != string::npos)
			{
				loc = line.find("=") + 1;
				inputWidth = stoi(line.substr(loc));
				netArgsFound++;
			}
			else if(line.find("inputHeight") != string::npos)
			{
				loc = line.find("=") + 1;
				inputHeight = stoi(line.substr(loc));
				netArgsFound++;
			}
			else if(line.find("inputDepth") != string::npos)
			{
				loc = line.find("=") + 1;
				inputDepth = stoi(line.substr(loc));
				netArgsFound++;
			}
			else if(line.find("RELU_CAP=") != string::npos)
			{
				loc = line.find("=") + 1;
				double relu_cap = stod(line.substr(loc));
				if(relu_cap != 5000.0)
					printf("RELU CAP seems to differ from default. Advise running with ConvNetCL.\n");
			}
			else if(line.find("LEAKY_RELU_CONST") != string::npos)
			{
				loc = line.find("=") + 1;
				double leaky = stod(line.substr(loc));
				if(leaky != .01)
					printf("LEAKY RELU constant seems to differ from default. Advise running with ConvNetCL.\n");
			}
			else if(line.find("MEAN") != string::npos)
			{
				loc = line.find("=") + 1;
				double mean = stod(line.substr(loc));
				if(mean != 0)
					printf("The training data seems to have been preprocessed collectively. Advise running with ConvNetCL.\n");
			}
			else if(line.find("STDDEV") != string::npos)
			{
				loc = line.find("=") + 1;
				double stddev = stod(line.substr(loc));
				if(stddev != 0)
					printf("The training data seems to have been preprocessed collectively. Advise running with ConvNetCL.\n");
			}
			else if(line.find("TRAINING_SIZE") != string::npos); //So it doesn't hit the else
			else
			{
				cout << "Improper file structure while getting Net args at line " << lineNum << ". Exiting load.";
				file.close();
				return false;
			}
			getline(file,line); lineNum++;
		}

		//check and make sure all 4 args were found
		if(netArgsFound != 4)
		{
			cout << "4 Net args needed. " << netArgsFound << " found. Exiting load.";
			file.close();
			return false;
		}
		//Lets init the Net.
		//cout << "Net params loaded" << endl;

		init(inputWidth,inputHeight,inputDepth);
		setActivType(activationType);


		//Now we get all the layers
		getline(file,line); lineNum++;
		while(line != "END_ALL")
		{
			if(line == "ACTIV_LAYER")
			{
				int numActivArgs = 0, layer_activationType;
				getline(file,line); lineNum++;
				while(line != "END_ACTIV_LAYER")
				{
					if(line.find("activationType") != string::npos)
					{
						loc = line.find("=") + 1;
						layer_activationType = stoi(line.substr(loc));
						numActivArgs++;
					}
					else
					{
						cout << "Improper file structure while getting ActivLayer args at line " << lineNum << ". Exiting load.";
						file.close();
						return false;
					}
					getline(file,line); lineNum++;
				}

				if(numActivArgs != 1)
				{
					cout << "1 ActivLayer arg needed. " << numActivArgs << " found. Line " << lineNum << ". Exiting load.";
					file.close();
					return false;
				}

				//make layer
				addActivLayer(layer_activationType);
				//cout << "ActivLayer added" << endl;
			}
			else if(line == "MAX_POOL_LAYER")
			{
				int numPoolArgs = 0, pool_stride, pool_size;
				getline(file,line); lineNum++;
				while(line != "END_MAX_POOL_LAYER")
				{
					if(line.find("stride") != string::npos)
					{
						loc = line.find("=") + 1;
						pool_stride = stoi(line.substr(loc));
						numPoolArgs++;
					}
					else if(line.find("poolSize") != string::npos)
					{
						loc = line.find("=") + 1;
						pool_size = stoi(line.substr(loc));
						numPoolArgs++;
					}
					else
					{
						cout << "Improper file structure while getting MaxPoolLayer args at line " << lineNum << ". Exiting load.";
						file.close();
						return false;
					}
					getline(file,line); lineNum++;
				}
				if(numPoolArgs != 2)
				{
					cout << "2 ActivLayer args needed. " << numPoolArgs << " found. Line " << lineNum << ". Exiting load.";
					file.close();
					return false;
				}

				addMaxPoolLayer(pool_size,pool_stride);
				//cout << "MaxPoolLayer added" << endl;
			}
			else if(line == "CONV_LAYER")
			{
				int conv_stride, conv_pad, conv_numFilters, conv_filterSize, convArgs = 0;
				string conv_weights;
				getline(file,line); lineNum++;
				while(line != "END_CONV_LAYER")
				{
					if(line.find("stride") != string::npos)
					{
						loc = line.find("=") + 1;
						conv_stride = stoi(line.substr(loc));
						convArgs++;
					}
					else if(line.find("padding") != string::npos)
					{
						loc = line.find("=") + 1;
						conv_pad = stoi(line.substr(loc));
						convArgs++;
					}
					else if(line.find("numFilters") != string::npos)
					{
						loc = line.find("=") + 1;
						conv_numFilters = stoi(line.substr(loc));
						convArgs++;
					}
					else if(line.find("filterSize") != string::npos)
					{
						loc = line.find("=") + 1;
						conv_filterSize = stoi(line.substr(loc));
						convArgs++;
					}
					else if(line.find("weights") != string::npos)
					{
						loc = line.find("=") + 1;
						conv_weights = line.substr(loc);
						convArgs++;
					}
					else
					{
						cout << "Improper file structure while getting ConvLayer args at line " << lineNum << ". Exiting load.";
						file.close();
						return false;
					}
					getline(file,line); lineNum++;
				}
				if(convArgs != 5)
				{
					cout << "5 ConvLayer args needed. " << convArgs << " found. Line " << lineNum << ". Exiting load.";
					file.close();
					return false;
				}
				//cout << "At addConvLayer" << endl;
				addConvLayer(conv_numFilters,conv_stride,conv_filterSize,conv_pad,conv_weights);
				//cout << "ConvLayer added" << endl;
			}
			else if(line == "SOFTMAX_LAYER")
			{
				addSoftmaxLayer();
				//cout << "SoftmaxLayer added" << endl;
			}
			else
			{
				cout << "Improper file structure while getting layers at line " << lineNum << ". Exiting load.";
				file.close();
				return false;
			}
			getline(file,line); lineNum++;
		}
		file.close();
		return true;
	}
	file.close();
	cout << "Unknown file format" << endl;
	return false;
}

bool Net::save(const char* filename)
{
	//get file stuff
	ofstream file;
	file.open(filename);

	if(!file.is_open())
		return false;

	char data[50];
	//need to save in such a way that it can load dynamically
	string out = "NET1.0\n";

	//put in Net hyperparameters. NOT TRAINING OR REAL DATA
	sprintf(data,"%d", n_activationType);
	out += "activationType=";
	out += data;
	out += '\n';

	vector<vector<vector<double> > > blankVector = n_blankInput.getNeurons();

	sprintf(data,"%lu",blankVector.size());
	out += "inputWidth=";
	out += data;
	out += '\n';

	sprintf(data,"%lu",blankVector[0].size());
	out += "inputHeight=";
	out += data;
	out += '\n';

	sprintf(data,"%lu",blankVector[0][0].size());
	out += "inputDepth=";
	out += data;
	out += '\n';

	out += "END_NET\n";

	for(int l=1; l < n_layers.size(); l++)
	{
		int type = n_layers[l]->getType();
		if(type == Net::MAX_POOL_LAYER)
		{
			MaxPoolLayer *pool = (MaxPoolLayer*)n_layers[l];
			out += "MAX_POOL_LAYER\n";

			out += pool->getHyperParameters();

			out += "END_MAX_POOL_LAYER\n";
		}
		else if(type == Net::ACTIV_LAYER)
		{
			ActivLayer *act = (ActivLayer*)n_layers[l];
			out += "ACTIV_LAYER\n";

			out += act->getHyperParameters();

			out += "END_ACTIV_LAYER\n";

		}
		else if(type == Net::CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)n_layers[l];
			out += "CONV_LAYER\n";

			out += conv->getHyperParameters();

			out += "END_CONV_LAYER\n";
		}
		else if(type == Net::SOFTMAX_LAYER)
		{
			out += "SOFTMAX_LAYER\n";
		}
	}
	out += "END_ALL";
	file << out;
	file.close();

	return true;
}

unsigned long Net::getMaxNeuronSize() const
{
	unsigned long maxNeuronSize = n_layers[0]->getNumNeurons();
	for(int i=0; i < n_layers.size(); i++)
	{
		if(n_layers[i]->getNumNeurons() > maxNeuronSize)
		{
			maxNeuronSize = n_layers[i]->getNumNeurons();
		}
	}

	return maxNeuronSize;
}

unsigned long Net::getMemForward() const
{
	//Needed mem is all the weights and biases, and 2 * maxNeuronSize
	unsigned long maxNeuronSize = n_layers[0]->getNumNeurons();
	unsigned long weightsAndBiasMem = 0;
	for(int i=0; i < n_layers.size(); i++)
	{
		if(n_layers[i]->getType() == Net::CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)n_layers[i];
			weightsAndBiasMem += conv->getMemWeightsAndBiases();
		}
		if(n_layers[i]->getNumNeurons() > maxNeuronSize)
		{
			maxNeuronSize = n_layers[i]->getNumNeurons();
		}
	}

	return weightsAndBiasMem + 2 * maxNeuronSize;
}

unsigned long Net::getMem() const
{
	//Needed mem is weightAndBiasMem + neuronMemForEachForwardLayer + 2*maxNeuronsSize(for backprop)
	unsigned long memMinusForwardNeuron = getMemForward();
	unsigned long forwardNeuronMem = 0;
	for(int i=0; i < n_layers.size(); i++)
	{
		forwardNeuronMem += n_layers[i]->getNumNeurons() * sizeof(double);
	}

	return memMinusForwardNeuron + forwardNeuronMem;
}

void Net::printTrainingDistribution() const
{
	int max = n_trainingDataTrueVals[0];
	for(int i=0; i< n_trainingDataTrueVals.size(); i++)
	{
		if(n_trainingDataTrueVals[i] > max)
			max = n_trainingDataTrueVals[i];
	}
	//cout << "max " << max << endl;

	int *amounts = new int[max+1];
	for(int i=0; i<= max; i++)
		amounts[i] = 0;
	for(int i=0; i < n_trainingDataTrueVals.size(); i++)
	{
		amounts[(int)n_trainingDataTrueVals[i]]++;
	}

	for(int i=0; i <= max; i++)
	{
		printf("True val: %d. Amount: %d.   %.4lf%%\n",i,amounts[i],amounts[i]/(double)n_trainingDataTrueVals.size()*100);
	}

	delete amounts;
}

/**********************
 * InputLayer
 **********************/

const int InputLayer::i_type = Net::INPUT_LAYER;

InputLayer::InputLayer()
{
	i_numNeurons = 0;
	i_resizeable = true;
}

InputLayer::~InputLayer(){}

InputLayer::InputLayer(const vector<vector<vector<double> > >& trainingImage, vector<vector<vector<double> > >* blankdNeurons)
{
	i_neurons = &trainingImage;
	//resize3DVector(i_dneurons,trainingImage.size(),trainingImage[0].size(),trainingImage[0][0].size());
	i_dneurons = blankdNeurons;
	i_resizeable = false;

	i_numNeurons = trainingImage.size() * trainingImage[0].size() * trainingImage[0][0].size();
}

int InputLayer::getType() const
{
	return i_type;
}

string InputLayer::getNeuronDims() const
{
	char out[100];
	sprintf(out, "%lu x %lu x %lu\n",(*i_neurons).size(),(*i_neurons)[0].size(),(*i_neurons)[0][0].size());
	return string(out);

}

double* InputLayer::getImage() const
{
	double *image = new double[i_numNeurons];
	int imIndex = 0;
	for(int i=0; i< i_neurons->size(); i++)
	{
		for(int j=0; j< (*i_neurons)[i].size(); j++)
		{
			for(int k=0; k< (*i_neurons)[i][j].size(); k++)
			{
				image[imIndex++] = (*i_neurons)[i][j][k];
			}
		}
	}
	return image;
}

void InputLayer::getImage(double* image, int size) const
{
	//cout << "here" << endl;
	int start = 0;
	if(size == i_numNeurons)
	{
		for(int i=0; i< i_neurons->size(); i++)
		{
			for(int j=0; j< (*i_neurons)[i].size(); j++)
			{
				for(int k=0; k< (*i_neurons)[i][j].size(); k++)
				{
					image[start++] = (*i_neurons)[i][j][k];
					//cout << (*i_neurons)[i][j][k] << endl;
				}
			}
		}
	}
	else
	{
		cout << size << " != " << i_numNeurons << endl;
	}
}

int InputLayer::getImageSize() const
{
	return i_numNeurons;
}

int InputLayer::getNumNeurons() const
{
	return i_numNeurons;
}

void InputLayer::forwardprop(const Layer& prevLayer){}
void InputLayer::backprop(Layer& prevLayer){}

const vector<vector<vector<double> > >& InputLayer::getNeurons() const
{
	//cout << "getNeurons: " << i_neurons->size() << endl;
	//cout << "neuron mem: " << i_neurons << endl;
	return *i_neurons;
}

vector<vector<vector<double> > >& InputLayer::getdNeurons()
{
	return *i_dneurons;
}

bool InputLayer::setImage(const vector<vector<vector<double> > >* trainingImage, vector<vector<vector<double> > >* blankdNeurons)
{
	if(i_resizeable)
	{
		//cout << "In setImage.\ntrainingImage.size() = " <<trainingImage->size() << endl;
		//const vector<vector<vector<double> > >& t = trainingImage;
		i_neurons = trainingImage;
		i_dneurons = blankdNeurons;
		//cout << "trainingImage mem " << trainingImage  << endl;
		//vectorClone(trainingImage,*i_neurons);
		//cout << "i_neurons.size() = " << i_neurons->size() << endl;
		i_resizeable = false;
		i_numNeurons = trainingImage->size() * (*trainingImage)[0].size() * (*trainingImage)[0][0].size();
		return true;
	}
	return false;
}

unsigned long InputLayer::getMem() const
{
	return getNumNeurons() * sizeof(double) * 2; //one for neurons, one for dneurons
}

/**********************
 * ConvLayer
 **********************/

const int ConvLayer::c_type = Net::CONV_LAYER;

ConvLayer::ConvLayer(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad)
{
	init(prevLayer,numFilters,stride,filterSize,pad);

	//need some way to initialize, save, and load weights and biases
	initRandomWeights();
}

ConvLayer::ConvLayer(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad, string weightsAndBiases)
{
	//cout << "Init ConvLayer with loaded weights" << endl;
	init(prevLayer,numFilters,stride,filterSize,pad);

	//need some way to initialize, save, and load weights and biases
	initWeights(weightsAndBiases);
}

void ConvLayer::init(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad)
{
	// need to set size of neurons, dneurons, weights, dweights. Also make sure that the sizes and such match
	// up with the amount of padding and stride. If padding is greater than 0 will need to make new 3d vector 
	// to hold padded array (or use if statements?) DO IN FORWARD PROP AND MAKE LOCAL VARIABLE.
	
	//cout << "prevLayer.getNeurons().size() = " << prevLayer.getNeurons().size() << endl;

	const vector<vector<vector<double> > > prevNeurons = prevLayer.getNeurons();
	int prevWidth = prevNeurons.size();
	int prevHeight = prevNeurons[0].size();
	int prevDepth = prevNeurons[0][0].size();

	c_prevNeuronWidth = prevWidth;
	c_prevNeuronHeight = prevHeight;
	c_prevNeuronDepth = prevDepth;

	//size of new layer (prevWidth - filterSize + 2*pad)/stride+1 x (prevHeight - filterSize + 2pad)/stride+1 
	// x numFilters
	double fnewWidth = (prevWidth - filterSize + 2*pad)/((double)stride) + 1;
	int newWidth = (prevWidth - filterSize + 2*pad)/stride + 1;
	double fnewHeight = (prevHeight - filterSize + 2*pad)/((double)stride) + 1;
	int newHeight = (prevHeight - filterSize + 2*pad)/stride + 1;
	int newDepth = numFilters;

	if(fabs(fnewHeight - newHeight) > .01 || fabs(fnewWidth - newWidth) > .01 )
	{
		cout << "Hyperparameters lead to bad size. Must have this equation be an int.\n\t(prevWidth - filterSize + 2*pad)/stride + 1;";
		throw "Incorrect hyperparameters";
	}

	c_stride = stride;
	c_padding = pad;
	c_paddedNeuronSize = (prevWidth + 2*pad) * (prevHeight + 2*pad) * prevDepth;

	resize3DVector(c_neurons,newWidth,newHeight,newDepth);
	resize3DVector(c_dneurons,newWidth,newHeight,newDepth);
	c_numNeurons = newWidth * newHeight * newDepth;

	int padVecSize = (prevWidth+2*pad)*(prevHeight+2*pad)*prevDepth;

	c_maxSizeNeeded = GETMAX(c_numNeurons, padVecSize);
	
	//Size of c_weights (and c_dweights) will be numFilters x filterSize x filterSize x numFilters
	c_weights.resize(numFilters);
	c_dweights.resize(numFilters);
	for(int i=0; i< numFilters; i++)
	{
		resize3DVector(c_weights[i], filterSize, filterSize, prevDepth);
		resize3DVector(c_dweights[i], filterSize, filterSize, prevDepth);
	}
	//cout << numFilters << "x" << filterSize << "x" << filterSize << "x" << prevDepth << endl;
	c_numWeights = numFilters * filterSize * filterSize * prevDepth;
	//cout << c_numWeights << endl;

	c_biases.resize(numFilters);
	c_dbiases.resize(numFilters);

	c_numBiases = numFilters;
}

ConvLayer::~ConvLayer(){}

string ConvLayer::getNeuronDims() const
{
	char out[100];
	sprintf(out, "%lu x %lu x %lu\n",c_neurons.size(),c_neurons[0].size(),c_neurons[0][0].size());
	return string(out);

}

int ConvLayer::getType() const
{
	return c_type;
}

int ConvLayer::getPaddedNeuronSize() const{
	return c_paddedNeuronSize;
}

int ConvLayer::getNumNeurons() const
{
	return c_numNeurons;
}

int ConvLayer::getNumWeights() const
{
	return c_numWeights;
}

int ConvLayer::getMaxSizeNeeded() const
{
	return c_maxSizeNeeded;
}

void ConvLayer::initRandomWeights()
{
	default_random_engine gen(time(0));
	int filterSize = c_weights[0].size();
	//double max = pow(filterSize * filterSize + 1, -.5); //* filterSize * filterSize
	//uniform_real_distribution<double> distr(-max,max);

	double numInputs = filterSize * filterSize + 1;
	double sqrtIn = pow(2/numInputs,.5);
	//normal_distribution<double> distr(0,1);
	normal_distribution<double> distr(0,sqrtIn);

	//printf("num weights: %lu sqrtIn %lf\n", c_weights.size()*c_weights[0].size()*c_weights[0][0].size()
	//	*c_weights[0][0][0].size(), sqrtIn);

	//uniform_real_distribution<double> distr(-.05,.05); // + - .005 using meanSub or +- .05 worked better
	for(int f = 0;f<c_weights.size(); f++)
	{
		for(int i=0; i< c_weights[0].size(); i++)
		{
			for(int j=0; j< c_weights[0][0].size(); j++)
			{
				for(int k=0; k< c_weights[0][0][0].size(); k++)
				{
					//double rnum = distr(gen);
					c_weights[f][i][j][k] = distr(gen);//rnum;
					//cout << rnum << endl;

					//c_weights[f][i][j][k] = distr(gen)*sqrtIn;
				}
			}
		}
	}

	for(int b=0; b< c_biases.size(); b++)
	{
		c_biases[b] = 0;//distr(gen);
	}


}

void ConvLayer::initWeights(string weights)
{
	double weight;
	int startIndex = 0, endIndex;
	for(int f = 0;f<c_weights.size(); f++)
	{
		for(int i=0; i< c_weights[0].size(); i++)
		{
			for(int j=0; j< c_weights[0][0].size(); j++)
			{
				for(int k=0; k< c_weights[0][0][0].size(); k++)
				{
					endIndex = weights.find(',',startIndex);
					weight = stod(weights.substr(startIndex,endIndex));
					c_weights[f][i][j][k] = weight;
					startIndex = endIndex + 1;
				}
			}
		}
	}
	// now do the biases
	startIndex = weights.find('_') + 1;
	for(int b=0; b < c_biases.size(); b++)
	{
		endIndex = weights.find(',',startIndex);
		c_biases[b] = stod(weights.substr(startIndex,endIndex));
		startIndex = endIndex + 1;
	}
}

vector<int> ConvLayer::getKernelHyperParameters() const
{
	vector<int> hyper(7);
	hyper[0] = c_weights[0].size(); // filterSize
	hyper[1] = c_stride;
	hyper[2] = c_prevNeuronWidth; // width of prevneurons
	hyper[3] = c_prevNeuronDepth;
	hyper[4] = c_biases.size();  // num filters
	hyper[5] = c_padding;
	hyper[6] = c_prevNeuronHeight;

	return hyper;
}

void ConvLayer::_putWeights(double* weights, int vectIndex) const
{
	int start = c_weights[0].size() * c_weights[0][0].size() * c_weights[0][0][0].size() * vectIndex;
	//printf("%d %d\n", start, vectIndex);
	for(int i=0; i< c_weights[vectIndex].size(); i++)
	{
		for(int j=0; j< c_weights[vectIndex][i].size(); j++)
		{
			for(int k=0; k< c_weights[vectIndex][i][j].size(); k++)
			{
				weights[start++] = c_weights[vectIndex][i][j][k];
				//weights[start++] = c_weights.at(vectIndex).at(i).at(j).at(k);

			}
		}
	}
}

double* ConvLayer::getWeights() const 
{
	//cout << "allocing " << c_numWeights << " weights" << endl;
	//cout << c_weights.size() << "x" << c_weights[0].size() << "x" << c_weights[0][0].size() << "x" << c_weights[0][0][0].size() << endl;
	//int numWeights = c_weights.size() * c_weights[0].size() * c_weights[0][0].size();
	double *weights = new double[c_numWeights];

	//cout << "allocing threads" << endl;
	thread *t = new thread[c_weights.size()];

	for(int f=0; f< c_weights.size(); f++)
	{
		t[f] = thread(&ConvLayer::_putWeights,this,weights,f);
	}

	//cout << "joining threads" << endl;
	for(int f = 0; f < c_weights.size(); f++)
	{
		t[f].join();
	}

	//delete t;

	//cout << "end conv" << endl;
	return weights;

}

void ConvLayer::setWeights(double* weights)
{
	for(int f=0; f < c_weights.size(); f++)
	{
		for(int i=0; i < c_weights[f].size(); i++)
		{
			for(int j=0; j < c_weights[f][i].size(); j++)
			{
				for(int k=0; k < c_weights[f][i][j].size(); k++)
				{
					c_weights[f][i][j][k] = *(weights++);
				}
			}
		}
	}
}

int ConvLayer::getNumBiases() const
{
	return c_numBiases;
}

double* ConvLayer::getBiases() const
{
	double *biases = new double[c_numBiases];

	for(int b = 0; b < c_numBiases; b++)
	{
		biases[b] = c_biases[b];
	}

	return biases;
}

void ConvLayer::setBiases(double* biases)
{
	for(int b=0; b <c_biases.size(); b++)
	{
		c_biases[b] = *(biases++);
	}
}

void ConvLayer::forwardprop(const Layer& prevLayer)
{
	vector<vector<vector<double> > > padSource;
	const vector<vector<vector<double> > > *source;
	if(c_padding != 0)
	{
		padZeros(prevLayer.getNeurons(),c_padding,padSource);
		source = &padSource;
		if(Net::walkthrough)
		{
			cout << "In ConvLayer padding forward" << endl;
			printVector(padSource);
			getchar();
		}
	}
	else
		source = &prevLayer.getNeurons();
	double sum;
	int oX, oY;
	int subsetWidth = c_weights[0].size(); // this is the same as filterSize
	int subsetHeight = c_weights[0][0].size();
	for(int f=0; f<c_weights.size(); f++) // which filter we're on
	{
		oX = 0;
		for(int i=0; i <= source->size()-subsetWidth; i+= c_stride) // row in source   		i < source.size()-1
		{
			oY = 0;
			for(int j=0; j <= (*source)[i].size()-subsetHeight;j+=c_stride) // col in source	j < source[i].size()-1
			{
				//now we go into the stride subset
				sum = 0;
				
				for(int s=0; s < subsetWidth; s++) // row in stride subset of source
				{
					for(int r=0; r < subsetHeight; r++) // col in stride subset of source
					{
						for(int k=0; k < (*source)[i][j].size(); k++) // depth in source
						{
							sum += (*source)[i+s][j+r][k] * c_weights[f][s][r][k];
							//sum += source.at(i+s).at(j+r).at(k) * weights.at(f).at(s).at(r).at(k);

							//can I set some of the prevLayer.getdNeuron local values at this point???
						}
					}
				}
				// add bias
				sum += c_biases[f];
				// add into c_neurons[i%stride, j%stride, f]
				c_neurons[oX][oY++][f] = sum;
			}
			oX++;
		}
	}
	/*
	printVector(c_neurons);
	cout << "conv" << endl;
	exit(0);
	*/

	if(Net::walkthrough)
	{
		cout << "In ConvLayer forwardprop" << endl;
		printVector(c_neurons);
		getchar();
	}
}



void ConvLayer::backprop(Layer& prevLayer)
{
	if(Net::walkthrough)
	{
		cout << "In ConvLayer backprop" << endl;
		printVector(c_dneurons);
		getchar();
	}

	vector<vector<vector<double> > >& p_dNeurons = prevLayer.getdNeurons();
	vector<vector<vector<double> > > padded_dNeurons;
	resize3DVector(padded_dNeurons,p_dNeurons.size() + 2*c_padding,p_dNeurons[0].size() + 2*c_padding, p_dNeurons[0][0].size());
	//setAll3DVector(p_dNeurons, 0);
	//setAll1DVector(c_dbiases, 0);
	setAll3DVector(padded_dNeurons,0);
	for(int b=0; b< c_dbiases.size(); b++)
	{
		c_dbiases[b] = 0;
	}
	setAll4DVector(c_dweights, 0);

	vector<vector<vector<double> > > padSource;
	const vector<vector<vector<double> > > *source;
	if(c_padding != 0)
	{
		padZeros(prevLayer.getNeurons(),c_padding,padSource);
		source = &padSource;

		if(Net::walkthrough)
		{
			cout << "In ConvLayer padding back source" << endl;
			printVector(padSource);
			getchar();
		}
	}
	else
		source = &prevLayer.getNeurons();

	//vector<vector<vector<double> > > source;
	//padZeros(prevLayer.getNeurons(),c_padding,source);

	int oX, oY; // outX & outY
	int subsetWidth = c_weights[0].size();
	int subsetHeight = c_weights[0][0].size();
	for(int f=0; f<c_weights.size(); f++) // which filter we're on
	{
		oX = 0;
		for(int i=0; i <= source->size()-subsetWidth; i+= c_stride) // row in source
		{
			oY = 0;
			for(int j=0; j <= (*source)[i].size()-subsetHeight;j+=c_stride) // col in source
			{
				//now we go into the stride subset				
				for(int s=0; s < subsetWidth; s++) // row in stride subset of source
				{
					for(int r=0; r < subsetHeight; r++) // col in stride subset of source
					{
						for(int k=0; k < (*source)[i][j].size(); k++) // depth in source
						{
							padded_dNeurons[i+s][j+r][k] += c_dneurons[oX][oY][f] * c_weights[f][s][r][k];
							c_dweights[f][s][r][k]  	 += c_dneurons[oX][oY][f] * (*source)[i+s][j+r][k];

							//padded_dNeurons.at(i+s).at(j+r).at(k) += c_dneurons.at(oX)[oY][f] * c_weights.at(f).at(s).at(r).at(k);
							//c_dweights.at(f).at(s).at(r).at(k) += c_dneurons.at(oX)[oY][f] * source.at(i+s).at(j+r).at(k);
						}
					}
				}
				// add bias
				c_dbiases[f] += c_dneurons[oX][oY++][f];
				//c_dbiases.at(f) += c_dneurons.at(oX).at(oY++).at(f);
			}
			oX++;
		}
	}


	//put the padded_dNeurons into the real p_dNeurons
	for(int i=c_padding; i < padded_dNeurons.size() - c_padding; i++)
	{
		for(int j=c_padding; j < padded_dNeurons[i].size()-c_padding; j++)
		{
			for(int k=0; k < padded_dNeurons[i][j].size(); k++)
			{
				p_dNeurons[i-c_padding][j-c_padding][k] = padded_dNeurons[i][j][k];
			}
		}
	}

	//end here if doing gradient check cause we don't want to update the weights

	//we need to substract the gradient * the stepSize from the weights and biases
	if(!Net::gradCheck)
	{
		//update the weights
		for(int f=0;f<c_weights.size();f++)
		{
			for(int i=0; i< c_weights[0].size(); i++)
			{
				for(int j=0; j< c_weights[0][0].size(); j++)
				{
					for(int k=0; k< c_weights[0][0][0].size(); k++)
					{
						//if(c_dweights[f][i][j][k] != 0)
							//cout << c_dweights[f][i][j][k] << endl;
						c_weights[f][i][j][k] -= Net::stepSize * c_dweights[f][i][j][k];
					}
				}
			}
		}

		//update the biases
		for(int i=0; i< c_biases.size(); i++)
		{
			c_biases[i] -= Net::stepSize * c_dbiases[i];
		}	

		if(Net::walkthrough)
		{
			cout << "Updated biases" << endl;
			printVector(c_biases);
			getchar();
			cout << endl << "Updated weights" << endl;
			printVector(c_weights);
			getchar();
		}
	}

	if(Net::walkthrough && c_padding != 0)
	{
		cout << "In ConvLayer padding backprop" << endl;
		printVector(padded_dNeurons);
		getchar();
	}
}

const vector<vector<vector<double> > >& ConvLayer::getNeurons() const
{
	return c_neurons;
}

vector<vector<vector<double> > >& ConvLayer::getdNeurons()
{
	return c_dneurons;
}

string ConvLayer::getHyperParameters() const
{
	char data[50];
	string out = "";

	sprintf(data,"%d",c_stride);
	out += "stride=";
	out += data;
	out += '\n';

	sprintf(data,"%d",c_padding);
	out += "padding=";
	out += data;
	out += '\n';

	sprintf(data,"%lu",c_weights.size());
	out += "numFilters=";
	out += data;
	out += '\n';

	sprintf(data,"%lu",c_weights[0].size());
	out += "filterSize=";
	out += data;
	out += '\n';

	out += "weights=";
	for(int f=0; f < c_weights.size(); f++)
	{
		for(int i=0; i< c_weights[0].size(); i++)
		{
			for(int j=0; j< c_weights[0][0].size(); j++)
			{
				for(int k=0; k< c_weights[0][0][0].size(); k++)
				{
					sprintf(data,"%lf,",c_weights[f][i][j][k]);
					out += data;
				}
			}
		}
	}
	//out += '\n';

	//out += "biases=";
	out += "_";
	for(int b=0; b< c_biases.size(); b++)
	{
		sprintf(data,"%lf,",c_biases[b]);
		out += data;
	}
	out += '\n';

	return out;
}

unsigned long ConvLayer::getMemWeightsAndBiases() const
{
	//int neuronsbytes = c_numNeurons * sizeof(double);
	unsigned long weightbytes = c_numWeights * sizeof(double);
	unsigned long biasbytes = c_numBiases * sizeof(double);

	return weightbytes + biasbytes;
}

unsigned long ConvLayer::getMem() const
{
	unsigned long weightbiasbytes = getMemWeightsAndBiases();
	unsigned long neuronbytes = c_numNeurons * sizeof(double);

	return weightbiasbytes + neuronbytes;
}

/**********************
 * MaxPoolLayer
 **********************/

const int MaxPoolLayer::m_type = Net::MAX_POOL_LAYER;

MaxPoolLayer::MaxPoolLayer(const Layer& prevLayer, int poolSize, int stride)
{
 	// need to set size of neurons and dneurons and make sure it goes evenly across new neurons
 	const vector<vector<vector<double> > > prevNeurons = prevLayer.getNeurons();
 	//cout << "prevLayer.getNeurons().size() = " << prevNeurons.size() << endl;
 	int pWidth = prevNeurons.size();
 	int pHeight = prevNeurons[0].size();
 	int pDepth = prevNeurons[0][0].size();

 	// new volume is (W1 - poolSize)/stride + 1 x (H1 - poolSize)/stride + 1 x D1
 	double fnewWidth = (pWidth - poolSize)/((double)stride) + 1;
 	int newWidth = (pWidth - poolSize)/stride + 1;
 	double fnewHeight = (pHeight - poolSize)/((double)stride) + 1;
 	int newHeight = (pHeight - poolSize)/stride + 1;
 	int newDepth = pDepth;

 	if(fabs(fnewHeight - newHeight) > .01 || fabs(fnewWidth - newWidth) > .01 )
	{
		cout << "Hyperparameters lead to bad size. Must have this equation be an int.\n\t(prevWidth - poolSize)/stride + 1;";
        cout << "MaxPoolLayer with poolsize "<<poolSize<< " and stride " << stride <<endl;
		throw "Incorrect hyperparameters";
	}

	m_stride = stride;
	m_poolSize = poolSize;
	m_prevWidth = pWidth;
	m_prevDepth = pDepth;

	resize3DVector(m_neurons,newWidth,newHeight,newDepth);
	resize3DVector(m_dneurons,newWidth,newHeight,newDepth);
	m_numNeurons = newWidth * newHeight * newDepth;
}

MaxPoolLayer::~MaxPoolLayer(){}

int MaxPoolLayer::getType() const
{
	return m_type;
}

int MaxPoolLayer::getNumNeurons() const
{
	return m_numNeurons;
}

string MaxPoolLayer::getNeuronDims() const
{
	char out[100];
	sprintf(out, "%lu x %lu x %lu\n",m_neurons.size(),m_neurons[0].size(),m_neurons[0][0].size());
	return string(out);
}

vector<int> MaxPoolLayer::getKernelHyperParameters() const
{
	vector<int> hyper(4);
	hyper[0] = m_prevWidth;
	hyper[1] = m_prevDepth;
	hyper[2] = m_poolSize;
	hyper[3] = m_stride;

	return hyper;
}

void MaxPoolLayer::forwardprop(const Layer& prevLayer)
{
	const vector<vector<vector<double> > >& source = prevLayer.getNeurons();
	int oX=0, oY=0;//, di, dj; //, dk;
	for(int k=0; k<source[0][0].size(); k++) // depth
	{
		oX = 0;
		for(int i=0;i<source.size()-1;i+=m_stride) // rows
		{
			oY = 0;
			for(int j=0;j<source[0].size()-1;j+=m_stride) // cols
			{
				double maxVal = source[i][j][k];
				for(int s=0;s<m_poolSize;s++)
				{
					for(int r=0; r< m_poolSize; r++)
					{
						if(source[i+s][j+r][k] >= maxVal)
						{
							maxVal = source[i+s][j+r][k];
						}
					}
				}
				m_neurons[oX][oY++][k] = maxVal;
			}
			oX++;
		}
	}

	/*
	cout << m_neurons[0][0][0] << endl;
	printVector(m_neurons);
	cout << "max layer. size " << getNumNeurons() << endl;
	exit(0);*/

	if(Net::walkthrough)
	{
		cout << "In MaxPoolLayer forwardprop" << endl;
		printVector(m_neurons);
		getchar();
	}
}

void MaxPoolLayer::backprop(Layer& prevLayer)
{
	if(Net::walkthrough)
	{
		cout << "In MaxPoolLayer backprop" << endl;
		printVector(m_dneurons);
		getchar();
	}

	//m_dneurons should have been set at this point by the next layer's backprop
	//set all prev dNeurons to 0
	vector<vector<vector<double> > >& p_dNeurons = prevLayer.getdNeurons();
 	setAll3DVector(p_dNeurons, 0);
 	const vector<vector<vector<double> > >& source = prevLayer.getNeurons();

 	//for each set in maxPool, add in the m_dneurons val to the max. All others will stay 0.
	int oX=0, oY=0, di, dj; //, dk;
	for(int k=0; k<source[0][0].size(); k++) // depth
	{
		oX = 0;
		for(int i=0;i<source.size()-1;i+=m_stride) // rows
		{
			oY = 0;
			for(int j=0;j<source[0].size()-1;j+=m_stride) // cols
			{
				double maxVal = source[i][j][k];
				for(int s=0;s<m_poolSize;s++)
				{
					for(int r=0; r< m_poolSize; r++)
					{
						if(source[i+s][j+r][k] >= maxVal)
						{
							maxVal = source[i+s][j+r][k];
							di = i+s;
							dj = j+r;
							//dk = k;
						}

					}
				}
				p_dNeurons[di][dj][k] += m_dneurons[oX][oY++][k];				
			}
			oX++;
		}
	}
}


const vector<vector<vector<double> > >& MaxPoolLayer::getNeurons() const
{
	return m_neurons;
}

vector<vector<vector<double> > >& MaxPoolLayer::getdNeurons()
{
	return m_dneurons;
}

string MaxPoolLayer::getHyperParameters() const
{
	string out = "";
	char data[50];

	sprintf(data,"%d",m_stride);
	out += "stride=";
	out += data;
	out += '\n';

	sprintf(data,"%d",m_poolSize);
	out += "poolSize=";
	out += data;
	out += '\n';

	return out;
}

unsigned long MaxPoolLayer::getMem() const
{
	return m_numNeurons * sizeof(double);
}

/**********************
 * ActivLayer
 **********************/

const int ActivLayer::a_type = Net::ACTIV_LAYER;

const double ActivLayer::LEAKY_RELU_CONST = .01;

const double ActivLayer::RELU_CAP = 5; // was 5000

ActivLayer::ActivLayer(const Layer& prevLayer, const int activationType)
{
	int notFound = true;
	//check if valid activation Type
	if(0 <= activationType && activationType < ActivLayer::NUM_ACTIV_TYPES)
	{ 
		notFound = false;
	}
	
	if(notFound)
	{
		cout << "The activationType was not valid. Please use a valid activationType. Exp: ActivLayer::RELU";
		throw "Invalid activationType";
	}

	//cout << "prevLayer.getNeurons().size() = " << prevLayer.getNeurons().size() << endl;

	a_activationType = activationType;
	vector<vector<vector<double> > > prevNeurons = prevLayer.getNeurons();
	int w = prevNeurons.size();
	int h = prevNeurons[0].size();
	int d = prevNeurons[0][0].size();
	resize3DVector(a_neurons,w,h,d);
	resize3DVector(a_dneurons,w,h,d);
	a_numNeurons = w * h * d;
}

ActivLayer::~ActivLayer(){}

int ActivLayer::getType() const
{
	return a_type;
}

int ActivLayer::getNumNeurons() const
{
	return a_numNeurons;
}

int ActivLayer::getActivationType() const
{
	return a_activationType;
}

string ActivLayer::getNeuronDims() const
{
	char out[100];
	sprintf(out, "%lu x %lu x %lu\n",a_neurons.size(),a_neurons[0].size(),a_neurons[0][0].size());
	return string(out);

}

void ActivLayer::forwardprop(const Layer& prevLayer)
{
	const vector<vector<vector<double> > >& prevNeurons = prevLayer.getNeurons();

	if(a_activationType == ActivLayer::RELU)
	{
		for(int i=0; i< prevNeurons.size(); i++)
		{
			for(int j=0; j< prevNeurons[0].size(); j++)
			{
				for(int k=0; k< prevNeurons[0][0].size(); k++)
				{
					a_neurons[i][j][k] = GETMAX(0,prevNeurons[i][j][k]);
					if(a_neurons[i][j][k] > ActivLayer::RELU_CAP)
						a_neurons[i][j][k] = ActivLayer::RELU_CAP;
				}
			}
		}
	}
	else if (a_activationType == ActivLayer::LEAKY_RELU)
	{
		for(int i=0; i< prevNeurons.size(); i++)
		{
			for(int j=0; j< prevNeurons[0].size(); j++)
			{
				for(int k=0; k< prevNeurons[0][0].size(); k++)
				{
					if(prevNeurons[i][j][k] < 0)
					{
						a_neurons[i][j][k] = LEAKY_RELU_CONST * prevNeurons[i][j][k];
						if(a_neurons[i][j][k] < -ActivLayer::RELU_CAP)
							a_neurons[i][j][k] = -ActivLayer::RELU_CAP;
					}
					else if(prevNeurons[i][j][k] > ActivLayer::RELU_CAP)
					{
						a_neurons[i][j][k] = ActivLayer::RELU_CAP;
					}
					else
						a_neurons[i][j][k] = prevNeurons[i][j][k];
				}
			}
		}
	}

	/*
	cout << a_neurons[0][0][0]<<endl;
	printVector(a_neurons);
	cout << "activLayer" << endl;
	exit(0);
	*/

	if(Net::walkthrough)
	{
		cout << "In ActivLayer forwardprop" << endl;
		printVector(a_neurons);
		getchar();
	}
}

void ActivLayer::backprop(Layer& prevLayer)
{
	if(Net::walkthrough)
	{
		cout << "In ActivLayer backprop" << endl;
		printVector(a_dneurons);
		getchar();
	}

	const vector<vector<vector<double> > >& prevNeurons = prevLayer.getNeurons();
	vector<vector<vector<double> > >& p_dNeurons = prevLayer.getdNeurons();
	if(a_activationType == ActivLayer::RELU)
	{
		for(int i=0; i< prevNeurons.size(); i++)
		{
			for(int j=0; j< prevNeurons[0].size(); j++)
			{
				for(int k=0; k< prevNeurons[0][0].size(); k++)
				{
					if(prevNeurons[i][j][k] >= 0 && prevNeurons[i][j][k] <= ActivLayer::RELU_CAP)
					{
						p_dNeurons[i][j][k] = a_dneurons[i][j][k];
					}
					else
					{
						p_dNeurons[i][j][k] = 0;
					}
				}
			}
		}
	}
	else if(a_activationType == ActivLayer::LEAKY_RELU)
	{
		for(int i=0; i< prevNeurons.size(); i++)
		{
			for(int j=0; j< prevNeurons[0].size(); j++)
			{
				for(int k=0; k< prevNeurons[0][0].size(); k++)
				{
					if(prevNeurons[i][j][k] >= 0 && prevNeurons[i][j][k] <= ActivLayer::RELU_CAP)
					{
						p_dNeurons[i][j][k] = a_dneurons[i][j][k];
					}
					else if(prevNeurons[i][j][k] >= -ActivLayer::RELU_CAP)
					{
						p_dNeurons[i][j][k] = LEAKY_RELU_CONST * a_dneurons[i][j][k];
					}
					else
					{
						p_dNeurons[i][j][k] = 0;
					}
				}
			}
		}
	}
}

const vector<vector<vector<double> > >& ActivLayer::getNeurons() const
{
	return a_neurons;
}

vector<vector<vector<double> > >& ActivLayer::getdNeurons()
{
	return a_dneurons;
}

string ActivLayer::getHyperParameters() const
{
	string out = "";
	char data[50];

	sprintf(data,"%d",a_activationType);
	out += "activationType=";
	out += data;
	out += '\n';

	return out;
}

unsigned long ActivLayer::getMem() const
{
	return a_numNeurons * sizeof(double);
}

/**********************
 * SoftmaxLayer
 **********************/

 const int SoftmaxLayer::s_type = Net::SOFTMAX_LAYER;

 SoftmaxLayer::SoftmaxLayer(const Layer& prevLayer)
 {
 	const vector<vector<vector<double> > >& prevNeurons = prevLayer.getNeurons();
 	int numDirs = 0;
	int i,j,k, *dir;
	int width = prevNeurons.size();
	int height = prevNeurons[0].size();
	int depth = prevNeurons[0][0].size();
	if(width  > 1) {numDirs++; dir = &i; s_neurons.resize(width);}
	if(height > 1) {numDirs++; dir = &j; s_neurons.resize(height);}
	if(depth  > 1) {numDirs++; dir = &k; s_neurons.resize(depth);}
	if(numDirs != 1 || width < 1 || height < 1 || depth < 1)
	{
		throw "Incorrect dimensions";
	}
	s_dneurons.resize(s_neurons.size());

	s_3neurons.resize(1);
	s_3neurons[0].resize(1);
	s_3neurons[0][0] = s_neurons;

	s_3dneurons.resize(1);
	s_3dneurons[0].resize(1);
	s_3dneurons[0][0] = s_dneurons;
 }

 SoftmaxLayer::~SoftmaxLayer(){}

 int SoftmaxLayer::getNumNeurons() const
 {
 	return s_neurons.size();
 }

string SoftmaxLayer::getNeuronDims() const
{
	char out[100];
	sprintf(out, "%lu x %lu x %lu\n",s_3neurons.size(),s_3neurons[0].size(),s_3neurons[0][0].size());
	return string(out);
}


 void SoftmaxLayer::forwardprop(const Layer& prevLayer)
 {
 	const vector<vector<vector<double> > >& prevNeurons = prevLayer.getNeurons();
	int i,j,k, *dir;
	int width = prevNeurons.size();
	int height = prevNeurons[0].size();
	int depth = prevNeurons[0][0].size();
	if(width  > 1) {dir = &i;}
	else if(height > 1) {dir = &j;}
	else {dir = &k;}

	for(i=0;i<width;i++)
	{
		for(j=0;j<height;j++)
		{
			for(k=0;k<depth;k++)
			{
				s_neurons[*dir] = prevNeurons[i][j][k];
			}
		}
	}

	maxSubtraction(s_neurons);
	double denom = vectorESum(s_neurons);
	for(int n=0; n < s_neurons.size(); n++)
	{
		s_neurons[n] = exp(s_neurons[n])/denom;
	}

	if(Net::walkthrough)
	{
		cout << "In SoftmaxLayer forwardprop" << endl;
		printVector(s_neurons);
		getchar();
	}

 }

 void SoftmaxLayer::setTrueVal(int trueVal)
 {
 	if(trueVal < 0 || s_neurons.size() <= trueVal)
 		throw "Invalid trueVal";
 	s_trueVal = trueVal;
 }

 void SoftmaxLayer::gradientCheck(Layer& prevLayer)
 {
 	vector<vector<vector<double> > >& prevdNeurons = prevLayer.getdNeurons();
 	const vector<vector<vector<double> > >& prevNeurons = prevLayer.getNeurons();
 	cout << "\n\n";
 	int i,j,k, *dir;
	int width = prevNeurons.size();
	int height = prevNeurons[0].size();
	int depth = prevNeurons[0][0].size();
	if(width  > 1) {dir = &i;}
	else if(height > 1) {dir = &j;}
	else {dir = &k;}
 	for(i = 0; i< width; i++)
 	{
 		for(j=0; j< height; j++)
 		{
 			for(k=0; k < depth; k++)
 			{
 				cout << "Softmax prevNeurons["<<i<<"]["<<j<<"]["<<k<<"]:\n";
 				cout << "Analytical Gradient:\t" << prevdNeurons[i][j][k]<< endl;
 				cout << "old s_neurons: ";
 				printVector(s_neurons);

 				vector<double> oldError = getError();
 				s_neurons[*dir] += Net::GRADCHECK_H;

 				//from forwardprop
 				maxSubtraction(s_neurons);
				double denom = vectorESum(s_neurons);
				for(int n=0; n < s_neurons.size(); n++)
				{
					s_neurons[n] = exp(s_neurons[n])/denom;
				}

				if(Net::walkthrough)
				{
					cout << "In Gradient SoftmaxLayer forwardprop" << endl;
					printVector(s_neurons);
					getchar();
				}
				//end from forwardprop

				cout << "new s_neurons: ";
				printVector(s_neurons);

				vector<double> newError = getError();
				double numericalGradient = 0;
				for(int e=0; e<newError.size(); e++)
				{
					cout << (newError[e] - oldError[e])/Net::GRADCHECK_H << endl;
					numericalGradient += (newError[e] - oldError[e])/Net::GRADCHECK_H;
				}
				cout << "Numberical Gradient:\t" << numericalGradient << endl;

 				s_neurons[*dir] -= Net::GRADCHECK_H;

 				cout <<"\n\n";
 			}
 		}
 	}
 }

 void SoftmaxLayer::backprop(Layer& prevLayer)
 {
 	for(int i=0; i< s_neurons.size(); i++)
 	{
 		if(i == s_trueVal)
 		{
 			//s_dneurons[i] = 1 - s_neurons[i]; // wrong
 			s_dneurons[i] = s_neurons[i] - 1; // correct
 		}
 		else
 		{
 			//s_dneurons[i] = -s_neurons[i]; //wrong
 			s_dneurons[i] = s_neurons[i]; // correct
 		}
 	}

 	if(Net::walkthrough || Net::showErrors)
	{
		cout << "In SoftmaxLayer backprop" << endl;
		//printVector(s_neurons);
		printVector(s_dneurons);
		if(Net::walkthrough)
			getchar();
	}

 	vector<vector<vector<double> > >& prevdNeurons = prevLayer.getdNeurons();
 	int n = 0;
 	for(int i=0; i< prevdNeurons.size(); i++)
 	{
 		for(int j=0; j< prevdNeurons[i].size(); j++)
 		{
 			for(int k=0; k< prevdNeurons[i][j].size(); k++)
 			{
 				prevdNeurons[i][j][k] = s_dneurons[n++];
 			}
 		}
 	}
 }

 int SoftmaxLayer::getPredictedClass()
 {
 	int maxLoc = 0;
	for(int i=1; i<s_neurons.size();i++)
	{
		if(s_neurons[i] > s_neurons[maxLoc])
		{
			maxLoc = i;
		}
	}
	//cout << maxLoc << endl;
	return maxLoc;
 }

 vector<double> SoftmaxLayer::getError() 
 {
 	for(int i=0; i< s_neurons.size(); i++)
 	{
 		if(i == s_trueVal)
 		{
 			s_dneurons[i] = 1 - s_neurons[i];
 		}
 		else
 		{
 			s_dneurons[i] = -s_neurons[i];
 		}
 	}
 	if(Net::showErrors)
 		printVector(s_dneurons);
 	return s_dneurons;
 }

 void SoftmaxLayer::setError(vector<double> error)
 {
 	if(error.size() != s_dneurons.size())
 	{
 		throw "Incorrect error vector size";
 	}
 	for(int i=0;i<error.size(); i++)
 	{
 		s_dneurons[i] = error[i];
 	}
 }

 int SoftmaxLayer::getType() const 
 {
 	return s_type;
 }

 const vector<vector<vector<double> > >& SoftmaxLayer::getNeurons() const
 {
 	return s_3neurons;
 } 

 vector<vector<vector<double> > >& SoftmaxLayer::getdNeurons()
 {
 	return s_3dneurons;
 }

 unsigned long SoftmaxLayer::getMem() const
 {
 	return s_neurons.size() * sizeof(double);
 }

/***********************************
 * Functions
 ***********************************/
void softmax(const vector<vector<vector<double> > >& vect, vector<double>& out)
{
	//cout << "orig vector";
	//printVector(vect);
	int numDirs = 0;
	int i,j,k, *dir;
	int width = vect.size();
	int height = vect[0].size();
	int depth = vect[0][0].size();
	if(width  > 1) {numDirs++; dir = &i; out.resize(width);}
	if(height > 1) {numDirs++; dir = &j; out.resize(height);}
	if(depth  > 1) {numDirs++; dir = &k; out.resize(depth);}
	if(numDirs != 1 || width < 1 || height < 1 || depth < 1)
	{
		throw "Incorrect dimensions";
	}
	//if(dir == &i) cout << "i" << endl;
	//if(dir == &j) cout << "j" << endl;
	//if(dir == &k) cout << "k" << endl;
	for(i=0;i<width;i++)
	{
		for(j=0;j<height;j++)
		{
			for(k=0;k<depth;k++)
			{
				out[*dir] = vect[i][j][k];
			}
		}
	}

	//now its in a 1D array

	//subtract the max for numerical stability
	maxSubtraction(out);
	//cout << "Pre exp" << endl;
	//printVector(out);
	double denom = vectorESum(out);
	//cout << "denom " << denom << endl;
	for(int n=0; n< out.size(); n++)
	{
		out[n] = exp(out[n])/denom;
	}
	//cout << "Final array ";
	//printVector(out);
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

void setAll4DVector(vector<vector<vector<vector<double> > > > &vect, double val)
{
	for(int n=0; n< vect.size(); n++)
	{
		setAll3DVector(vect[n],val);
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

void printVector(const vector<vector<vector<vector<double> > > > &vect)
{
	for(int n=0; n< vect.size(); n++)
	{
		printVector(vect[n]);
		cout << endl;
	}
}

void printVector(const vector<vector<vector<double> > > &vect)
{
	for(int i=0;i < vect.size();i++) // rows
	{
		cout << "|";
		for(int j=0;j < vect[i].size();j++)
		{
			for(int k=0; k < vect[i][j].size(); k++)
			{
				cout << setw(4) << vect[i][j][k];
				if(k != vect[i][j].size()-1) cout << ",";
			}
			cout << "|";
		}
		cout << endl;
	}
}

void printVector(const vector<double>& vect)
{
	cout << "|";
	for(int i=0; i< vect.size(); i++)
	{
		cout << vect[i];
		if(i != vect.size()-1) cout << ",";
	}
	cout << "|" << endl;
}

/*********************
 *
 * padZeros pads the outer edges of the array with the specified number of 0s.
 * It does this on every depth level. Depth is not padded.
 *
 *********************/
void padZeros(const vector<vector<vector<double> > > &source, int numZeros, vector<vector<vector<double> > > &dest)
{
	int width2 = source.size() + 2*numZeros;
	int height2 = source[0].size() + 2*numZeros;
	int depth2 = source[0][0].size();
	//cout << "pad zeros:" << endl;
	//printf("w %lu h %lu d %lu numZeros %d\n",source.size(),source[0].size(), source[0][0].size(), numZeros);
	//resize dest vector
	resize3DVector(dest,width2,height2,depth2);
	setAll3DVector(dest,0);
	for(int i=numZeros; i< dest.size()-numZeros; i++) // rows
	{
		for(int j=numZeros; j< dest[0].size()-numZeros; j++) // cols
		{
			for(int k=0; k< dest[0][0].size(); k++) // depth
			{
				dest[i][j][k] = source[i-numZeros][j-numZeros][k];
			}
		}
	}
}


double vectorESum(const vector<double> &source)
{
	double sum = 0;
	for(int i=0; i < source.size(); i++)
	{
		sum += exp(source[i]);
	}
	return sum;
}

void vectorClone(const vector<vector<vector<double> > > &source, vector<vector<vector<double> > > &dest)
{
	resize3DVector(dest,source.size(),source[0].size(),source[0][0].size());
	for(int i=0; i< dest.size(); i++)
	{
		for(int j=0; j< dest[0].size(); j++)
		{
			for(int k=0; k< dest[0][0].size(); k++)
			{
				dest[i][j][k] = source[i][j][k];
			}
		}
	}
	
}

void compressImage(vector<vector<vector<vector<double> > > >& vect, double newMin, double newMax)
{
	for(int n=0; n< vect.size(); n++)
	{
		compressImage(vect[n],newMin, newMax);
	}
}

void compressImage(vector<vector<vector<double> > >& vect, double newMin, double newMax)
{
	for(int i=0; i< vect.size(); i++)
	{
		for(int j=0; j< vect[i].size(); j++)
		{
			for(int k=0; k< vect[i][j].size(); k++)
			{
				vect[i][j][k] = vect[i][j][k]/255 * (newMax-newMin) + newMin;
			}
		}
	}
}

void maxSubtraction(vector<double>& vect)
{
	double max = vect[0];
	for(int i=1;i<vect.size();i++)
	{
		if(vect[i] > max)
			max = vect[i];
	}
	for(int i=0; i< vect.size(); i++)
	{
		vect[i] -= max;
	}
}

void vectorSubtraction(vector<vector<vector<double> > >& vect, double toSub)
{
	for(int i=0; i < vect.size(); i++)
	{
		for(int j=0; j< vect[i].size(); j++)
		{
			for(int k=0; k< vect[i][j].size(); k++)
			{
				vect[i][j][k] -= toSub;
			}
		}
	}
}

void vectorDivision(vector<vector<vector<double> > >& vect, double toDiv)
{
	for(int i=0; i < vect.size(); i++)
	{
		for(int j=0; j< vect[i].size(); j++)
		{
			for(int k=0; k< vect[i][j].size(); k++)
			{
				vect[i][j][k] /= toDiv;
			}
		}
	}
}

void vectorDivision(vector<vector<vector<vector<double> > > >& vect, double toDiv)
{
	for(int a = 0; a < vect.size(); a++)
		for(int i=0; i < vect[a].size(); i++)
		{
			for(int j=0; j< vect[a][i].size(); j++)
			{
				for(int k=0; k< vect[a][i][j].size(); k++)
				{
					vect[a][i][j][k] /= toDiv;
				}
			}
		}
}

void preprocessByFeature(vector<vector<vector<vector<double> > > >& vect)
{
	vector<vector<vector<double> > > meanVector;
	resize3DVector(meanVector,vect[0].size(),vect[0][0].size(),vect[0][0][0].size());
	setAll3DVector(meanVector,0);

	//get mean by feature
	for(int a=0; a < vect.size(); a++)
	{
		for(int i=0; i < vect[a].size(); i++)
		{
			for(int j=0; j< vect[a][i].size(); j++)
			{
				for(int k=0; k< vect[a][i][j].size(); k++)
				{
					meanVector[i][j][k] += vect[a][i][j][k];
				}
			}
		}
	}
	vectorDivision(vect, vect.size());

	//subtract the means
	for(int a=0; a < vect.size(); a++)
		for(int i=0; i < meanVector.size(); i++)
			for(int j=0; j < meanVector[i].size(); j++)
				for(int k=0; k < meanVector[i][j].size(); k++)
					vect[a][i][j][k] -= meanVector[i][j][k];



}

void preprocessCollective(vector<vector<vector<vector<double> > > >& vect)
{
	double m = mean(vect);
	double sdev = stddev(vect,m);

	for(int a=0; a < vect.size(); a++)
	{
		for(int i=0; i< vect[a].size();i++)
		{
			for(int j=0; j< vect[a][i].size(); j++)
			{
				for(int k=0; k< vect[a][i][j].size(); k++)
				{
					vect[a][i][j][k] = (vect[a][i][j][k] - m)/sdev;
				}
			}
		}
	}
}

void preprocess(vector<vector<vector<vector<double> > > >& vect)
{
	for(int v=0; v< vect.size(); v++)
	{
		preprocess(vect[v]); //thread this
	}
}

void preprocess(vector<vector<vector<double> > > & vect)
{
//    for(int i=0; i< vect.size();i++)
//        for(int j=0; j< vect[i].size(); j++)
//            for(int k=0; k< vect[i][j].size(); k++)
    
    

	//preprocess using (val - mean)/stdDeviation for all elements
	double m = mean(vect);
	double stddv = stddev(vect,m);

	//cout << "mean"<<m<<endl;
	//cout<<"stddev"<<stddv<<endl;
	for(int i=0; i< vect.size();i++)
	{
		for(int j=0; j< vect[i].size(); j++)
		{
			for(int k=0; k< vect[i][j].size(); k++)
			{
				vect[i][j][k] = (vect[i][j][k] - m)/stddv;
                //if(v == 1) {cout << vect[i][j][k] << ", ";}
			}
		}
	}
	// if(v==1)
	// {
	//     cout << endl;
	//     exit(0);
	// }
}

double mean(const vector<vector<vector<double> > > & vect)
{
	double mean = 0;
	for(int i=0; i< vect.size(); i++)
	{
		for(int j=0; j< vect[0].size(); j++)
		{
			for(int k=0; k< vect[0][0].size(); k++)
			{
				mean += vect[i][j][k];
			}
		}
	}
	mean /= vect.size() * vect[0].size() * vect[0][0].size();
	return mean;
}

double mean(const vector<vector<vector<vector<double> > > > & vect)
{
	double mean = 0;
	for(int a=0; a < vect.size(); a++)
		for(int i=0; i< vect[a].size(); i++)
		{
			for(int j=0; j< vect[a][i].size(); j++)
			{
				for(int k=0; k< vect[a][i][j].size(); k++)
				{
					mean += vect[a][i][j][k];
				}
			}
		}
	mean /= vect.size() * vect[0].size() * vect[0][0].size() * vect[0][0][0].size();
	return mean;
}

double stddev(const vector<vector<vector<vector<double> > > > & vect) 
{
	double m = mean(vect);
	return stddev(vect,m);
}

double stddev(const vector<vector<vector<vector<double> > > > & vect, double mean) 
{
	double sqdiffs = 0;
	double temp;
	for(int a=0; a < vect.size(); a++)
		for(int i=0; i< vect[a].size();i++)
		{
			for(int j=0; j< vect[a][i].size(); j++)
			{
				for(int k=0; k< vect[a][i][j].size(); k++)
				{
					temp = (vect[a][i][j][k] - mean);
					sqdiffs += temp * temp;
				}
			}
		}
	double sqdiffMean = sqdiffs / (vect.size() * vect[0].size() * vect[0][0].size() * vect[0][0][0].size());
	return sqrt(sqdiffMean);
}

double stddev(const vector<vector<vector<double> > > & vect) 
{
	double m = mean(vect);
	return stddev(vect,m);
}

double stddev(const vector<vector<vector<double> > > & vect, double mean) 
{
	double sqdiffs = 0;
	double temp;
	for(int i=0; i< vect.size();i++)
	{
		for(int j=0; j< vect[i].size(); j++)
		{
			for(int k=0; k< vect[i][j].size(); k++)
			{
				temp = (vect[i][j][k] - mean);
				sqdiffs += temp * temp;
			}
		}
	}
	double sqdiffMean = sqdiffs / (vect.size() * vect[0].size() * vect[0][0].size());
	return sqrt(sqdiffMean);
}

void meanSubtraction(vector<vector<vector<vector<double> > > >& vect)
{
	for(int n=0; n< vect.size(); n++)
	{
		meanSubtraction(vect[n]);
	}
}

void meanSubtraction(vector<vector<vector<double> > >& vect)
{
	double mean = 0;
	for(int i=0; i< vect.size(); i++)
	{
		for(int j=0; j< vect[0].size(); j++)
		{
			for(int k=0; k< vect[0][0].size(); k++)
			{
				mean += vect[i][j][k];
			}
		}
	}
	mean /= vect.size() * vect[0].size() * vect[0][0].size();

	for(int i=0; i< vect.size(); i++)
	{
		for(int j=0; j< vect[0].size(); j++)
		{
			for(int k=0; k< vect[0][0].size(); k++)
			{
				vect[i][j][k] -= mean;;
			}
		}
	}
}

void meanSubtraction(vector<double>& vect)
{
	double mean = 0;
	for(int i=0;i<vect.size();i++)
	{
		mean += vect[i];
	}
	mean /= vect.size();
	for(int i=0; i< vect.size(); i++)
	{
		vect[i] -= mean;
	}
}

void printArray(double* array, int size)
{
	for(int i=0; i< size; i++)
	{
		cout << array[i] << ", ";
	}
	cout << endl << endl;
}

int getMaxElementIndex(const vector<double>& vect)
{
	if(vect.size() < 1)
		return -1;
	double max = vect[0];
	double maxIndex = 0;
	for(int i=1; i< vect.size(); i++)
	{
		if(vect[i] > max)
		{
			max = vect[i];
			maxIndex = i;
		}
	}

	return maxIndex;
}

double getMaxElement(const vector<double>& vect)
{
	if(vect.size() < 1)
		return -1;
	double max = vect[0];
	for(int i=1; i< vect.size(); i++)
	{
		if(vect[i] > max)
		{
			max = vect[i];
		}
	}

	return max;
}

std::string LoadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	//cout << result << endl;
	return result;
}

void CheckError (cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::exit (1);
	}
}

cl_program CreateProgram (const std::string& source, cl_context& context)
{
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}
