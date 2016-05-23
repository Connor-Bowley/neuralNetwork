
#include "ConvNetCL.h"
#include <iostream>
#include <random>
#include <fstream>

using namespace std;

typedef std::vector<std::vector<std::vector<double> > > imVector;

#define GETMAX(x,y) (x > y) ? x: y


/*****************************************
 * Constructors and Destructors and init
 *****************************************/

Net::Net(const char* filename)
{
	load(filename);
}

Net::Net(int inputWidth, int inputHeight, int inputDepth)
{
	init(inputWidth, inputHeight, inputDepth);
}

Net::~Net()
{
	Layer *point;
	for(int i=0; i< __layers.size(); i++)
	{
		point = __layers.back();
		if(point->layerType == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)point;
			delete conv->weights;
			delete conv->biases;
		}
		__layers.pop_back();
		delete point;
	}

	if(__isFinalized)
	{
		clReleaseCommandQueue(queue);
		clReleaseMemObject(*neurons);
		clReleaseMemObject(*prevNeurons);
		for(int w = 0; w < clWeights.size(); w++)
		{
			clReleaseMemObject(clWeights[w]);
			clReleaseMemObject(clBiases[w]);
		}
		//running
		clReleaseKernel(convKernelF);
		clReleaseKernel(zeroPadKernelF);
		clReleaseKernel(maxPoolKernelF);
		clReleaseKernel(reluKernelF);
		clReleaseKernel(leakyReluKernelF);
		clReleaseKernel(softmaxKernelF);
		clReleaseProgram(CNForward);

		//training
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
	}

	clReleaseContext(__context);
}

void Net::init(int inputWidth, int inputHeight, int inputDepth)
{
	pushBackLayerSize(inputWidth,inputHeight,inputDepth);
	__layers.resize(1);
	initOpenCL();
}

void Net::initOpenCL()
{
	cl_int error;
	//platforms
	clGetPlatformIDs(0, nullptr, &__platformIdCount);
	__platformIds.resize(__platformIdCount);
	clGetPlatformIDs(__platformIdCount, __platformIds.data(), nullptr);

	//devices
	clGetDeviceIDs(__platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &__deviceIdCount);
	__deviceIds.resize(__deviceIdCount);
	clGetDeviceIDs(__platformIds[0], CL_DEVICE_TYPE_ALL, __deviceIdCount, __deviceIds.data(), nullptr);

	//context
	const cl_context_properties contextProperties[] = 
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties>(__platformIds[0]),
		0,0
	};
	__context = clCreateContext(contextProperties, __deviceIdCount, __deviceIds.data(),
		nullptr, nullptr, &error);
	CheckError(error);

}

/*****************************************
 * Functions dealing with layers
 *****************************************/

bool Net::addActivLayer()
{
	return addActivLayer(__defaultActivType);
}

bool Net::addActivLayer(int activType)
{
	if(activType >= MAX_ACTIV || activType < 0)
		return false;

	int prevWidth  = __neuronDims.back()[0];
	int prevHeight = __neuronDims.back()[1];
	int prevDepth  = __neuronDims.back()[2];

	ActivLayer* activ = new ActivLayer();
	activ->layerType = ACTIV_LAYER;
	activ->activationType = activType;

	__layers.push_back(activ);
	pushBackLayerSize(prevWidth,prevHeight,prevDepth);
	__isFinalized = false;
	return true;
}

bool Net::addConvLayer(int numFilters, int stride, int filterSize, int pad)
{
	return addConvLayer(numFilters, stride, filterSize, pad, string("random"));
}

bool Net::addConvLayer(int numFilters, int stride, int filterSize, int pad, string weightsAndBiases)
{
	int prevWidth  = __neuronDims.back()[0];
	int prevHeight = __neuronDims.back()[1];
	int prevDepth  = __neuronDims.back()[2];
	//check to make sure stride fits
	int widthNumer  = prevWidth - filterSize + 2 * pad;
	int heightNumer = prevHeight- filterSize + 2 * pad;
	if(widthNumer % stride != 0 || heightNumer % stride != 0) //incorrect hyperparameters
		return false;

	int newWidth = widthNumer/stride + 1;
	int newHeight = heightNumer/stride + 1;
	int newDepth = numFilters;


	ConvLayer* conv = new ConvLayer();
	conv->layerType = CONV_LAYER;
	conv->stride = stride;
	conv->filterSize = filterSize;
	//padding stuff
	conv->padding = pad;
	conv->paddedNeuronWidth = prevWidth + 2 * pad;
	conv->paddedNeuronHeight = prevHeight + 2 * pad;
	conv->paddedNeuronSize = conv->paddedNeuronWidth * conv->paddedNeuronHeight * prevDepth;

	int newNeuronSize = newWidth * newHeight * newDepth;
	conv->maxSizeNeeded = GETMAX(newNeuronSize, conv->paddedNeuronSize);

	//weights and biases
	conv->numBiases = numFilters;
	conv->numWeights = filterSize * filterSize * prevDepth * numFilters;
	conv->weights = new double[conv->numWeights];
	conv->biases = new double[conv->numBiases];
	if(weightsAndBiases.find("random") != string::npos)
		initRandomWeights(conv);
	else
		initWeights(conv, weightsAndBiases);

	__layers.push_back(conv);
	pushBackLayerSize(newWidth, newHeight, newDepth);

	if(__autoActivLayer)
		addActivLayer();
	__isFinalized = false;
	return true;
}

bool Net::addFullyConnectedLayer(int outputSize)
{
	return addConvLayer(outputSize, 1, __neuronDims.back()[0], 0);
}

bool Net::addMaxPoolLayer(int poolSize, int stride)
{
	int prevWidth  = __neuronDims.back()[0];
	int prevHeight = __neuronDims.back()[1];
	int prevDepth  = __neuronDims.back()[2];

	int widthNumer  = prevWidth - poolSize;
	int heightNumer = prevHeight- poolSize;
	if(widthNumer % stride != 0 || heightNumer % stride != 0) //incorrect hyperparameters
		return false;

	int newWidth = widthNumer/stride + 1;
	int newHeight = heightNumer/stride + 1;
	int newDepth = prevDepth;

	MaxPoolLayer* pool = new MaxPoolLayer();
	pool->layerType = MAX_POOL_LAYER;
	pool->stride = stride;
	pool->poolSize = poolSize;

	__layers.push_back(pool);
	pushBackLayerSize(newWidth, newHeight, newDepth);
	__isFinalized = false;
	return true;
}


void Net::pushBackLayerSize(int width, int height, int depth)
{
	__neuronSizes.push_back(width * height * depth);
	__neuronDims.resize(__neuronDims.size() + 1);
	__neuronDims.back().resize(3);
	__neuronDims.back()[0] = width;
	__neuronDims.back()[1] = height;
	__neuronDims.back()[2] = depth;
}

void Net::initRandomWeights(ConvLayer* conv)
{
	default_random_engine gen(time(0));

	//use the number of inputs to get the random start weights
	double numInputs = conv->filterSize * conv->filterSize + 1;
	double sqrtIn = pow(2/numInputs,.5);
	normal_distribution<double> distr(0,sqrtIn);

	conv->weights = new double[conv->numWeights];
	conv->biases  = new double[conv->numBiases];

	for(int f = 0;f < conv->numWeights; f++)
		conv->weights[f] = distr(gen);

	for(int b=0; b < conv->numBiases; b++)
		conv->biases[b] = 0;
}

void Net::initWeights(ConvLayer* conv, string& weights)
{
	int startIndex = 0, endIndex;
	for(int f = 0; f < conv->numWeights; f++)
	{
		endIndex = weights.find(',',startIndex);
		conv->weights[f] = stod(weights.substr(startIndex,endIndex));
		startIndex = endIndex + 1;	
	}
	// now do the biases
	startIndex = weights.find('_') + 1;
	for(int b=0; b < conv->numBiases; b++)
	{
		endIndex = weights.find(',',startIndex);
		conv->biases[b] = stod(weights.substr(startIndex,endIndex));
		startIndex = endIndex + 1;
	}
}

bool Net::setActivType(int activationType)
{
	if(activationType >= MAX_ACTIV || activationType < 0)
		return false;
	__defaultActivType = activationType;
	return true;
}

void Net::setAutoActivLayer(bool isAuto)
{
	__autoActivLayer = isAuto;
}


bool Net::finalize()
{
	__errorLog = "";
	bool returnVal = true;

	//check to see if there is enough output neurons on last layer for each class to be represented
	if(__neuronSizes.back() < __numClasses)
	{
		__errorLog += "OUTPUT_SIZE_ERROR: There are not enough output neurons to represent all classes\n";
		returnVal = false;
	}

	//need at least one hidden layer
	if(__layers.size() < 2)
	{
		__errorLog += "NET_TOO_SMALL_ERROR: There must be at least one hidden layer.\n";
		returnVal = false;
	}

	//if we are going to return false, do it now. else set up OpenCL
	if(!returnVal)
		return returnVal;

	//Do a bunch of OpenCL stuff
	int q = 0;
	if(__device != -1)
		q = __device;
	else if(__useGPU)
	{
		for(int i = 1; i < __deviceIdCount; i++)
		{
			cl_device_type type;
			CheckError(clGetDeviceInfo(__deviceIds[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, nullptr));
			if(type == CL_DEVICE_TYPE_GPU)
				q = i;
		}
	}

	cout << "Finalizing CNN using device " << q << endl;

	cl_int error;

	if(!__stuffBuilt)
	{
		//build the program
		//running
		CNForward = CreateProgram(LoadKernel("../kernels/ConvNetForward_kernel.cl"), __context);
		const cl_device_id* deviceToBuild = &(__deviceIds[q]);
		CheckError(clBuildProgram(CNForward, 1, deviceToBuild, nullptr, nullptr, nullptr));
		//training
		CNTraining = CreateProgram(LoadKernel("../kernels/ConvNetTraining_kernel.cl"), __context);
		CheckError(clBuildProgram(CNTraining, 1, deviceToBuild, nullptr, nullptr, nullptr));
		//Create the kernels; check for errors
		//running
		reluKernelF = clCreateKernel(CNForward, "relu", &error); CheckError(error);
		leakyReluKernelF = clCreateKernel(CNForward, "leakyRelu", &error); CheckError(error);
		convKernelF = clCreateKernel(CNForward, "convolve", &error); CheckError(error);
		convKernelFC = clCreateKernel(CNForward, "convolveConstant", &error); CheckError(error);
		zeroPadKernelF = clCreateKernel(CNForward, "zeroPad", &error); CheckError(error);
		maxPoolKernelF = clCreateKernel(CNForward, "maxPool", &error); CheckError(error);
		softmaxKernelF = clCreateKernel(CNForward, "softmax_allCL", &error); CheckError(error);
		//training
		reluKernel = clCreateKernel(CNTraining, "relu", &error); CheckError(error);
		reluBackKernel = clCreateKernel(CNTraining, "relu_back", &error); CheckError(error);
		leakyReluKernel = clCreateKernel(CNTraining, "leakyRelu", &error); CheckError(error);
		leakyReluBackKernel = clCreateKernel(CNTraining, "leakyRelu_back", &error); CheckError(error);
		convKernel = clCreateKernel(CNTraining, "convolve", &error); CheckError(error);
		convBackNeuronsKernel = clCreateKernel(CNTraining, "convolve_back_neurons", &error); CheckError(error);
		convBackBiasesKernel = clCreateKernel(CNTraining, "convolve_back_biases", &error); CheckError(error);
		convBackWeightsKernel = clCreateKernel(CNTraining, "convolve_back_weights", &error); CheckError(error);
		convBackWeightsMomentKernel = clCreateKernel(CNTraining, "convolve_back_weights_moment", &error); CheckError(error);
		zeroPadKernel = clCreateKernel(CNTraining, "zeroPad", &error); CheckError(error);
		zeroPadBackKernel = clCreateKernel(CNTraining, "zeroPad_back", &error); CheckError(error);
		maxPoolKernel = clCreateKernel(CNTraining, "maxPool", &error); CheckError(error);
		maxPoolBackKernel = clCreateKernel(CNTraining, "maxPool_back", &error); CheckError(error);
		softmaxKernel = clCreateKernel(CNTraining, "softmax", &error); CheckError(error);
		softmaxBackKernel = clCreateKernel(CNTraining, "softmax_back", &error); CheckError(error);
		copyArrayKernel = clCreateKernel(CNTraining, "copyArray", &error); CheckError(error);
		maxSubtractionKernel = clCreateKernel(CNTraining, "maxSubtraction", &error); CheckError(error);
		vectorESumKernel = clCreateKernel(CNTraining, "vectorESum", &error); CheckError(error);
		//make the queue
		queue = clCreateCommandQueue(__context, __deviceIds[q], 0, &error); 
		CheckError(error);
		__stuffBuilt = true;
	}

	int numConvLayers = 0;
	__maxNeuronSize = __neuronSizes[0];

	//if it has been previous finalized there might be stuff in the weights and biases, so remove it
	for(int i = 0; i < clWeights.size(); i++)
	{
		clReleaseMemObject(clWeights[i]);
		clReleaseMemObject(clBiases[i]);
	}
	clWeights.resize(0);
	clBiases.resize(0);

	for(int i=1; i < __layers.size(); i++)
	{
		int type = __layers[i]->layerType;
		if(type == CONV_LAYER)
		{
			numConvLayers++;
			ConvLayer *conv = (ConvLayer*) __layers[i];
			if(__constantMem)
			{
				clWeights.push_back(clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					sizeof(double) * conv->numWeights, conv->weights, &error));
				CheckError(error);
				clBiases.push_back(clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					sizeof(double) * conv->numBiases, conv->biases, &error));
				CheckError(error);
			}
			else
			{
				clWeights.push_back(clCreateBuffer(__context, CL_MEM_COPY_HOST_PTR,
					sizeof(double) * conv->numWeights, conv->weights, &error));
				CheckError(error);
				clBiases.push_back(clCreateBuffer(__context, CL_MEM_COPY_HOST_PTR,
					sizeof(double) * conv->numBiases, conv->biases, &error));
				CheckError(error);
			}

			//none of the other layers have the ability to increase the size from the layer before
			if(conv->maxSizeNeeded > __maxNeuronSize)
			{
				__maxNeuronSize = conv->maxSizeNeeded;
			}
		}
		else if (type == MAX_POOL_LAYER); //these are here so they don't catch on the else statement
		else if (type == ACTIV_LAYER);
		else
		{
			cout << "Unknown layer type for the GPU implemenation. Defaulting to CPU." << endl;
			cout << "Type: " << type << endl;
		}
	}
	n = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize,
			nullptr, &error);
	CheckError(error);
	neurons = &n;
	p = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double) * __maxNeuronSize,
			nullptr, &error);
	CheckError(error);
	prevNeurons = &p;

	denom = clCreateBuffer(__context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &error);


	__isFinalized = true;
	return true;
}


/*****************************************
 * Running and Training
 *****************************************/

 void Net::run(bool useGPU)
 {
 	// if(useGPU != __useGPU)
 	// {
 	// 	__useGPU = useGPU;
 	// 	__isFinalized = false;
 	// }

 	if(!__isFinalized)
 		finalize();

 	if(!__dataPreprocessed)
 		preprocessData();

 	//set some softmax related args that won't change
 	clSetKernelArg(maxSubtractionKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	clSetKernelArg(vectorESumKernel, 1, sizeof(int), &(__neuronSizes.back()));
 	for(int i=0; i < __confidences.size(); i++)
 		__confidences[i].resize(__neuronSizes.back());

 	vector<double> test(__maxNeuronSize);

 	for(int r = 0; r < __data.size(); r++)
 	{
 		// for(int c=0; c< __data[r].size(); c++)
 		// 	cout << "|" << __data[r][c];
 		// cout << endl;
 		// exit(0);
 		// cout << "image " << r << " of " << __data.size() << endl;
 		//put in the next image
 		CheckError(clEnqueueWriteBuffer(queue, (*prevNeurons), CL_TRUE, 0,
				sizeof(double) * __neuronSizes[0],
				__data[r].data(), 0, nullptr, nullptr));
		clFinish(queue);

		int curConvLayer = 0;
		size_t globalWorkSize[] = {0,0,0};
		cl_mem *temp;
		//go through the layers
		for(int i = 1; i < __layers.size(); i++) //start at 1 because 0 is input
		{
			//printf("Layer %d, type %d\n", i, __layers[i]->layerType);
			if(__layers[i]->layerType == CONV_LAYER)
			{
				ConvLayer* conv = (ConvLayer*)__layers[i];
				if(conv->padding != 0) //if padding != 0
				{
					clSetKernelArg(zeroPadKernelF, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(zeroPadKernelF, 1, sizeof(cl_mem), neurons);
					clSetKernelArg(zeroPadKernelF, 2, sizeof(int), &(conv->padding)); // padding
					clSetKernelArg(zeroPadKernelF, 3, sizeof(int), &(__neuronDims[i-1][0])); // prevWidth
					clSetKernelArg(zeroPadKernelF, 4, sizeof(int), &(__neuronDims[i-1][1])); // prevHeight
					clSetKernelArg(zeroPadKernelF, 5, sizeof(int), &(__neuronDims[i-1][2])); // depth (before and after zero pad)

					// run it for the size of the new array
					globalWorkSize[0] = (size_t) conv->paddedNeuronSize;
					CheckError(clEnqueueNDRangeKernel(queue, zeroPadKernelF, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
					clFinish(queue);

					//swap the buffers so prevNeurons holds the zero padded data
					temp = neurons;
					neurons = prevNeurons;
					prevNeurons = temp;
				}

				clSetKernelArg(convKernelF, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(convKernelF, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(convKernelF, 2, sizeof(cl_mem), &clWeights[curConvLayer]);
				clSetKernelArg(convKernelF, 3, sizeof(cl_mem), &clBiases[curConvLayer]);
				clSetKernelArg(convKernelF, 4, sizeof(int), &(conv->numBiases)); //numFilters
				clSetKernelArg(convKernelF, 5, sizeof(int), &(conv->filterSize));
				clSetKernelArg(convKernelF, 6, sizeof(int), &(conv->stride));
				clSetKernelArg(convKernelF, 7, sizeof(int), &(conv->paddedNeuronWidth)); // prevWidth
				clSetKernelArg(convKernelF, 8, sizeof(int), &(__neuronDims[i-1][2])); // prevDepth

				globalWorkSize[0] = (size_t)__neuronSizes[i];
				CheckError(clEnqueueNDRangeKernel(queue, convKernelF, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				curConvLayer++;
			}
			else if(__layers[i]->layerType == MAX_POOL_LAYER)
			{
				MaxPoolLayer* pool = (MaxPoolLayer*)__layers[i];
				clSetKernelArg(maxPoolKernelF, 0, sizeof(cl_mem), prevNeurons);
				clSetKernelArg(maxPoolKernelF, 1, sizeof(cl_mem), neurons);
				clSetKernelArg(maxPoolKernelF, 2, sizeof(int), &(__neuronDims[i-1][0])); //prevwidth
				clSetKernelArg(maxPoolKernelF, 3, sizeof(int), &(__neuronDims[i-1][2])); //prevdepth
				clSetKernelArg(maxPoolKernelF, 4, sizeof(int), &(pool->poolSize)); //poolsize
				clSetKernelArg(maxPoolKernelF, 5, sizeof(int), &(pool->stride)); //stride
				//printf("Size: %d, w: %d, h: %d, d: %d, str: %d, pool: %d\n", __neuronSizes[i], __neuronDims[i-1][0], __neuronDims[i-1][1], __neuronDims[i-1][2], pool->stride, pool->poolSize);
				globalWorkSize[0] = (size_t)__neuronSizes[i];
				CheckError(clEnqueueNDRangeKernel(queue, maxPoolKernelF, 1,
					nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			}
			else if(__layers[i]->layerType == ACTIV_LAYER)
			{
				int type = ((ActivLayer*)__layers[i])->activationType;
				if(type == RELU)
				{
					clSetKernelArg(reluKernelF, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(reluKernelF, 1, sizeof(cl_mem), neurons);

					globalWorkSize[0] = (size_t)__neuronSizes[i];
					CheckError(clEnqueueNDRangeKernel(queue, reluKernelF, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				}
				else if(type == LEAKY_RELU)
				{
					clSetKernelArg(leakyReluKernelF, 0, sizeof(cl_mem), prevNeurons);
					clSetKernelArg(leakyReluKernelF, 1, sizeof(cl_mem), neurons);

					globalWorkSize[0] = (size_t)__neuronSizes[i];
					CheckError(clEnqueueNDRangeKernel(queue, leakyReluKernelF, 1,
						nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
				}
			}

			clFinish(queue);

		// cout << "Layer " << i << endl;
		// 	CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes[i],
		// 		test.data(), 0, nullptr, nullptr));
		// 	for(int j=0; j< __neuronSizes[i]; j++)
		// 	{
		// 		cout << test[j] << ", ";
		// 	}
		// 	cout << endl << endl;
		// 	getchar();

			temp = neurons;
			neurons = prevNeurons;
			prevNeurons = temp;
		}

		//cout << "Softmax" << endl;
		//softmax. 
		//cout << "soft size: " << __neuronSizes.back() << endl;
		globalWorkSize[0] = 1;
		//maxSubtraction. arg 1 set above
		clSetKernelArg(maxSubtractionKernel, 0, sizeof(cl_mem), prevNeurons);
		CheckError(clEnqueueNDRangeKernel(queue, maxSubtractionKernel, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

	// CheckError(clEnqueueReadBuffer(queue, (*prevNeurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
	// 	__confidences[r].data(), 0, nullptr, nullptr));
	// for(int c=0; c < __confidences[r].size(); c++)
	// 	cout << __confidences[r][c] << " ";
	// cout << endl;

		//vectorESum. arg 1 set above
		clSetKernelArg(vectorESumKernel, 0, sizeof(cl_mem), prevNeurons);
		clSetKernelArg(vectorESumKernel, 2, sizeof(cl_mem), &denom);
		CheckError(clEnqueueNDRangeKernel(queue, vectorESumKernel, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

	// CheckError(clEnqueueReadBuffer(queue, (denom), CL_TRUE, 0, sizeof(double),
	// 	__confidences[r].data(), 0, nullptr, nullptr));
	// cout << __confidences[r][0] << endl;
		
		//softmax
		globalWorkSize[0] = (size_t)__neuronSizes.back();
		clSetKernelArg(softmaxKernelF, 0, sizeof(cl_mem), prevNeurons);
		clSetKernelArg(softmaxKernelF, 1, sizeof(cl_mem), neurons);
		clSetKernelArg(softmaxKernelF, 2, sizeof(cl_mem), &denom);
		CheckError(clEnqueueNDRangeKernel(queue, softmaxKernelF, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

		CheckError(clEnqueueReadBuffer(queue, (*neurons), CL_TRUE, 0, sizeof(double) * __neuronSizes.back(),
		 	__confidences[r].data(), 0, nullptr, nullptr));

	// for(int c=0; c < __confidences[r].size(); c++)
	// 	cout << __confidences[r][c] << " ";
	// cout << endl;
	// getchar();


 	}
 }

 void Net::getCalculatedClasses(vector<int>& dest)
 {
 	dest.resize(__confidences.size());
 	for(int i=0; i < dest.size(); i++)
 		dest[i] = getMaxElementIndex(__confidences[i]);
 }

 void Net::getConfidences(vector<vector<double> >& confidences)
 {
 	confidences = __confidences;
 }

int Net::getMaxElementIndex(const vector<double>& vect)
{
	if(vect.size() < 1)
		return -1;
	double max = vect[0];
	double maxIndex = 0;
	for(int i=1; i< vect.size(); i++)
		if(vect[i] > max)
		{
			max = vect[i];
			maxIndex = i;
		}
	return maxIndex;
}

/*****************************************
 * Functions dealing with data
 *****************************************/

//running
void Net::addData(const vector<imVector>& data)
{
	for(int d = 0; d < data.size(); d++)
	{
		__data.resize(__data.size() + 1);
		__data.back().resize(__neuronSizes[0]);
		int dat = 0;
		for(int i=0; i < data[d].size(); i++)
			for(int j=0; j < data[d][i].size(); j++)
				for(int k=0; k < data[d][i][j].size(); k++)
					__data.back()[dat++] = data[d][i][j][k];
	}
	__confidences.resize(__data.size());
	__dataPreprocessed = false;
}

void Net::clearData()
{
	__data.resize(0);
}

void Net::setData(const vector<imVector>& data)
{
	clearData();
	addData(data);
}

//training
bool Net::addTrainingData(const vector<imVector>& trainingData, const vector<double>& trueVals)
{
	if(trainingData.size() != trueVals.size())
		return false;

	for(int t = 0; t < trainingData.size(); t++)
	{
		int trueIndex = getTrueValIndex(trueVals[t]);

		__trainingData[trueIndex].resize(__trainingData[trueIndex].size() + 1);
		__trainingData[trueIndex].back().resize(__neuronSizes[0]);
		int dat = 0;
		for(int i=0; i < trainingData[t].size(); i++)
			for(int j=0; j < trainingData[t][i].size(); j++)
				for(int k=0; k < trainingData[t][i][j].size(); k++)
					__trainingData[trueIndex].back()[dat++] = trainingData[t][i][j][k];
	}

	__numClasses = __trueVals.size();
	return true;
}

void Net::clearTrainingData()
{
	__trainingData.resize(0);
	__trueVals.resize(0);
}

bool Net::setTrainingData(const vector<imVector>& trainingData, const vector<double>& trueVals)
{
	clearTrainingData();
	return addTrainingData(trainingData,trueVals);
}

int Net::getTrueValIndex(double trueVal)
{
	for(int i=0; i < __trueVals.size(); i++)
		if(__trueVals[i] == trueVal)
			return i;

	int newSize = __trueVals.size() + 1;
	__trueVals.resize(newSize);
	__trueVals.back() = trueVal;
	__trainingData.resize(newSize);
	return __trueVals.size();
}

int Net::getNumClasses() const
{
	return __neuronSizes.back();
}

void Net::preprocessData() // thread this 
{
	//preprocess using (val - mean)/stdDeviation for all elements
	for(int i = 0; i < __data.size(); i++)
	{
		//get mean
		double mean = 0;
		for(int pix = 0; pix < __data[i].size(); pix++)
			mean += __data[i][pix];
		mean /= __data[i].size();

		//cout << "mean = " << mean << endl;

		//get stddev
		double stddev = 0; 
		double temp;
		for(int pix = 0; pix < __data[i].size(); pix++)
		{
			temp = __data[i][pix] - mean;
			stddev += temp * temp;
		}
		stddev = sqrt(stddev / __data[i].size());

		//cout << "stddev = " << stddev << endl;
		//exit(0);

		//adjust the values
		for(int pix=0; pix < __data[i].size(); pix++)
			__data[i][pix] = (__data[i][pix] - mean)/stddev;
	}

	__dataPreprocessed = true;
}

/*****************************************
 * Load and Save
 *****************************************/

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

	setAutoActivLayer(false);

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
				addConvLayer(conv_numFilters,conv_stride,conv_filterSize,conv_pad,conv_weights);
			}
			else if(line == "SOFTMAX_LAYER")
			{ /*addSoftmaxLayer();*/}
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

	//put in Net hyperparameters only. NOT TRAINING OR REAL DATA
	sprintf(data,"%d", __defaultActivType);
	out += "activationType="; out += data; out += '\n';

	sprintf(data,"%d",__neuronDims[0][0]);
	out += "inputWidth="; out += data; out += '\n';

	sprintf(data,"%d",__neuronDims[0][1]);
	out += "inputHeight="; out += data; out += '\n';

	sprintf(data,"%d",__neuronDims[0][2]); 
	out += "inputDepth="; out += data; out += '\n';

	out += "END_NET\n";

	for(int l=1; l < __layers.size(); l++)
	{
		int type = __layers[l]->layerType;
		if(type == MAX_POOL_LAYER)
		{
			MaxPoolLayer *pool = (MaxPoolLayer*)__layers[l];
			out += "MAX_POOL_LAYER\n";

			sprintf(data,"%d",pool->stride);
			out += "stride="; out += data; out += '\n';

			sprintf(data,"%d",pool->poolSize);
			out += "poolSize="; out += data; out += '\n';

			out += "END_MAX_POOL_LAYER\n";
		}
		else if(type == ACTIV_LAYER)
		{
			ActivLayer *act = (ActivLayer*)__layers[l];
			out += "ACTIV_LAYER\n";

			sprintf(data,"%d",act->activationType);
			out += "activationType="; out += data; out += '\n';

			out += "END_ACTIV_LAYER\n";

		}
		else if(type == CONV_LAYER)
		{
			ConvLayer* conv = (ConvLayer*)__layers[l];
			out += "CONV_LAYER\n";

			sprintf(data,"%d",conv->stride);
			out += "stride="; out += data; out += '\n';

			sprintf(data,"%d",conv->padding);
			out += "padding="; out += data; out += '\n';

			sprintf(data,"%d",conv->numBiases); // same as numFilters
			out += "numFilters="; out += data; out += '\n';

			sprintf(data,"%d",conv->filterSize);
			out += "filterSize="; out += data; out += '\n';

			out += "weights=";
			for(int f=0; f < conv->numWeights; f++)
			{
				sprintf(data,"%lf,",conv->weights[f]);
				out += data;
			}
			out += "_"; // biases
			for(int b=0; b < conv->numBiases; b++)
			{
				sprintf(data,"%lf,",conv->biases[b]);
				out += data;
			}
			out += '\n';

			out += "END_CONV_LAYER\n";
		}
	}
	out += "SOFTMAX_LAYER";
	out += "END_ALL";
	file << out;
	file.close();

	return true;
}

/*****************************************
 * OpenCL Functions
 *****************************************/

int Net::getDevice() const
{
	return __device;
}

bool Net::setDevice(unsigned int device)
{
	if(device >= __deviceIdCount)
		return false;
	cl_uint canDouble;
	CheckError(clGetDeviceInfo(__deviceIds[device], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &canDouble, nullptr));
	if(canDouble == 0)
		return false;
	__device = device;
	return true;
}

void Net::setGPU(bool useGPU)
{
	__useGPU = useGPU;
}

void Net::setConstantMem(bool useConstantMem)
{
	__constantMem = useConstantMem;
}

void Net::CheckError (cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::exit (1);
	}
}

std::string Net::LoadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	//cout << result << endl;
	return result;
}

cl_program Net::CreateProgram (const std::string& source, cl_context& context)
{
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}