#include <algorithm>
#include "ConvNetCL_light.h"
#include <iostream>
#include <iomanip>
#include <cfloat>
#include <random>
#include <fstream>
#include <random>
#include <unordered_map>
#include <sstream>

// 	includes brought in from ConvNetCL.h
//
// #include <vector>	
// #include <string> 
// #include <time.h>
// 
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
//
// #ifdef __APPLE__
//  	#include "OpenCL/opencl.h"
// #else
//  	#include "CL/cl.h"
// #endif
//typedef std::vector<std::vector<std::vector<double> > > imVector; // typedef pulled from the .h file

using namespace std;

/*****************************************
 * Constructors and Destructors and inits
 *****************************************/

Net::Net()
{

}

Net::Net(const char* filename)
{
	load(filename);
}

Net::~Net()
{
	Layer *point;
	for(int i=0; i< __layers.size(); i++)
	{
		point = __layers.back();

		delete point;
	}
}


void Net::copyLayers(const Net& other)
{
	//nicely delete all current layers
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
	__layers.resize(0);

	__layers.push_back(new Layer); //input layer
	int wid = other.__neuronDims[0][0];
	int hei = other.__neuronDims[0][1];
	int dep = other.__neuronDims[0][2];
	pushBackLayerSize(wid,hei,dep);

	for(int l = 1; l < other.__layers.size(); l++)
	{
		if(other.__layers[l]->layerType == CONV_LAYER)
		{
			const ConvLayer* conv = (ConvLayer*) other.__layers[l];

			ConvLayer* myconv = new ConvLayer();
			*myconv = *conv;
			myconv->layerType = CONV_LAYER;

			int widthNumer  = conv->paddedNeuronWidth - conv->filterSize;//prevWidth - filterSize + 2 * pad;
			int heightNumer = conv->paddedNeuronHeight - conv->filterSize;//prevHeight- filterSize + 2 * pad;
			int newWidth = widthNumer/conv->stride + 1;
			int newHeight = heightNumer/conv->stride + 1;
			int newDepth = conv->numBiases;// = numFilters;
			pushBackLayerSize(newWidth,newHeight,newDepth);
			__layers.push_back(myconv);
		}
		else if(other.__layers[l]->layerType == MAX_POOL_LAYER)
		{
			const MaxPoolLayer* pool = (MaxPoolLayer*)other.__layers[l];
			addMaxPoolLayer(pool->poolSize, pool->stride);
			// MaxPoolLayer* mypool = (MaxPoolLayer*)__layers.back();

			// printf("pool x stride. Old: %d x %d. New: %d x %d.\n",pool->poolSize,pool->stride, mypool->poolSize, mypool->stride);
		}
		else if(other.__layers[l]->layerType == ACTIV_LAYER)
		{
			const ActivLayer* act = (ActivLayer*)other.__layers[l];
			addActivLayer(act->activationType);
		}

		// if(other.__layers[l]->layerType == CONV_LAYER)
		// {
		// 	ConvLayer* myconv = (ConvLayer*)__layers.back();
		// 	ConvLayer* otherConv = (ConvLayer*)other.__layers[l];
		// 	// printf("ConvLayers equal: %d\n", myconv->equals(*otherConv));
		// }
	}
}

void Net::init(int inputWidth, int inputHeight, int inputDepth)
{
	if(__inited)
		return;
	pushBackLayerSize(inputWidth,inputHeight,inputDepth);
    Layer* abstract = new Layer;
	__layers.resize(1);
    __layers.back() = abstract;
    __inited = true;
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
	{
		printf("ActivLayer failing to add with type number %d\n", activType);
		return false;
	}

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

bool Net::addConvLayer(int numFilters, int stride, int filterSize, int pad, const string& weightsAndBiases)
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
	// conv->weights = new double[conv->numWeights];
	// conv->biases = new double[conv->numBiases];
	// if(weightsAndBiases.find("random") != string::npos)
	// 	initRandomWeights(conv, prevDepth);
	// else
	// 	initWeights(conv, weightsAndBiases);

	__layers.push_back(conv);
	pushBackLayerSize(newWidth, newHeight, newDepth);

	//check to see if it has max weights
	if(conv->numWeights > __maxWeightSize)
		__maxWeightSize = conv->numWeights;

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

void Net::initRandomWeights(ConvLayer* conv, int prevDepth)
{
    //cout << "making random weights" << endl;
	default_random_engine gen(time(0));

	//use the number of inputs to get the random start weights
	double numInputs = conv->filterSize * conv->filterSize * prevDepth + 1;
	double sqrtIn = pow(2/numInputs,.5);
	normal_distribution<double> distr(0,sqrtIn);

	//printf("num weights: %d sqrtIn %lf\n", conv->numWeights, sqrtIn);

	conv->weights = new double[conv->numWeights];
	conv->biases  = new double[conv->numBiases];

	for(int f = 0;f < conv->numWeights; f++)
		conv->weights[f] = distr(gen);

	for(int b=0; b < conv->numBiases; b++)
		conv->biases[b] = 0;
}

void Net::initWeights(ConvLayer* conv, const string& weights)
{
	// cout << "start init weights" << endl;
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
	// cout << "done init weights" << endl;
}

bool Net::setActivType(int activationType)
{
	if(activationType >= MAX_ACTIV || activationType < 0)
		return false;
	__defaultActivType = activationType;
	return true;
}

void Net::printLayerDims() const 
{
	printf("Input         %d x %d x %d\n", __neuronDims[0][0], __neuronDims[0][1], __neuronDims[0][2]);
	for(int i=1; i < __neuronDims.size(); i++)
		if(__layers[i]->layerType == CONV_LAYER)
			printf("Convolution   %d x %d x %d\n", __neuronDims[i][0], __neuronDims[i][1], __neuronDims[i][2]);
		else if(__layers[i]->layerType == MAX_POOL_LAYER)
			printf("Max Pool      %d x %d x %d\n", __neuronDims[i][0], __neuronDims[i][1], __neuronDims[i][2]);
		else if(__layers[i]->layerType == ACTIV_LAYER)
		{
			ActivLayer* act = (ActivLayer*)__layers[i];
			if(act->activationType == LEAKY_RELU)
			printf("Leaky RELU    %d x %d x %d\n", __neuronDims[i][0], __neuronDims[i][1], __neuronDims[i][2]);
			else if(act->activationType == RELU)
			printf("RELU          %d x %d x %d\n", __neuronDims[i][0], __neuronDims[i][1], __neuronDims[i][2]);
		}
		else
			printf("Unknown Layer: %d\n",__layers[i]->layerType);
}

int Net::getInputHeight() const
{
	return __neuronDims[0][0];
}

int Net::getInputWidth() const
{
	return __neuronDims[0][1];
}

void Net::setAutoActivLayer(bool isAuto)
{
	__autoActivLayer = isAuto;
}

void Net::getClassNames(vector<ClassInfo>& infos) const
{
	infos = __classes;
}



string Net::getClassForTrueVal(int trueVal) const
{
	for(int i = 0; i < __classes.size(); i++)
		if(__classes[i].trueVal == trueVal)
			return __classes[i].name;
	return "";
}


string Net::secondsToString(time_t seconds)
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


string Net::tolower(string str)
{
	transform(str.begin(), str.end(), str.begin(), ::tolower);
	return str;
}

bool Net::stringToDims(string str, int* dims)
{
	//should look like 32x32x3 or 32 x 32 x 3
	str = tolower(str);
	str.erase(remove(str.begin(), str.end(), ' '), str.end());
	stringstream tokens(str);
	string item;
	int i = 0;
	while(getline(tokens, item, 'x'))
	{
		if(i == 3)
		{
			printf("You can only have 3 dimensions for a layer size.\n");
			return false;
		}
		if(!item.empty())
		{
			dims[i++] = stoi(item);
		}

	}
	return true;
}

/*****************************************
 * Load and Save
 *****************************************/

bool Net::load(const char* filename)
{
	if(__inited)
		return false;
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
		setAutoActivLayer(false);
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
				__RELU_CAP = stod(line.substr(loc));
			}
			else if(line.find("LEAKY_RELU_CONST") != string::npos)
			{
				loc = line.find("=") + 1;
				__LEAKY_RELU_CONST = stod(line.substr(loc));
			}
			else if(line.find("MEAN") != string::npos)
			{
				loc = line.find("=") + 1;
				__mean = stod(line.substr(loc));
				if(__mean != 0)
					__preprocessIndividual = false;
				else
					__preprocessIndividual = true;
			}
			else if(line.find("STDDEV") != string::npos)
			{
				loc = line.find("=") + 1;
				__stddev = stod(line.substr(loc));
			}
			else if(line.find("TRAINING_SIZE") != string::npos)
			{
				loc = line.find("=") + 1;
				__trainingSize = stoul(line.substr(loc));
			}
			else if(line.find("CLASSES") != string::npos)
			{
				while(true)
				{
					ClassInfo info;
					getline(file,line); lineNum++; //this should get the trueVal
					if(line.find("END_CLASSES") != string::npos)
						break;
					info.trueVal = stoi(line);
					getline(file,line); lineNum++;
					info.name = line;

					__classes.push_back(info);
				}
			}
			else
			{
				cout << "Improper file structure while getting Net args at line " << lineNum << ". Exiting load.";
				file.close();
				return false;
			}
			getline(file,line); lineNum++;
		}

		// printf("End net\n");

		//check and make sure all 4 args were found
		if(netArgsFound < 4)
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
			// printf("%s\n", line.c_str());
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
					// printf("%s\n", line.c_str());
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

		// printf("End load\n");
		file.close();
		return true;
	}
	else if(tolower(line) == "net_config")
	{
		int lineNum = 1;
		bool haveInput = false;
		char buf[500];
		//while(file >> line)
		while(getline(file, line))
		{
			// if(!file.good())
			// 	break;

			lineNum++;

			//see if it starts with pound. If so it is a comment.
			int j = 0;
			for(j = 0; j < line.length(); j++)
			{
				if(line[j] != ' ' && line[j] != '\t' && line[j] != '\n' && line[j] != '\r')
					break;
			}
			// printf("j: %d length: %lu\n", j, line.length());
			if(j >= line.length() || line[j] == '#')
				continue;

			//tokenize the line and put the results in the items vector
			vector<string> items;
			stringstream tokens(line);
			string item;
			int dims[] = {-1,-1,-1};
			while(getline(tokens, item, ' '))
			{
				if(!item.empty())
				{
					items.push_back(item);
				}
			}
			// cout << "line: " << line << endl;
			// cout << "items size: " << items.size() << endl;

			items[0] = tolower(items[0]);

			if(items[0] == "global_activ")
			{
				if(tolower(items[1]) == "leaky_relu")
					setActivType(LEAKY_RELU);
				else if(tolower(items[1]) == "relu")
					setActivType(RELU);
				else
				{
					printf("Line %d: Unimplemented activation type \"%s\".\n", lineNum, items[1].c_str());
					return false;
				}
			}
			else if(items[0] == "auto_activ")
			{
				if(tolower(items[1]) == "false")
					setAutoActivLayer(false);
				else if(tolower(items[1]) == "true")
					setAutoActivLayer(true);
				else
				{
					printf("Line %d: auto_activ must be set to either true or false.\n",lineNum);
					return false;
				}
			}
			else if(items[0] == "input")
			{
				if(haveInput)
				{
					printf("Line %d. Cannot have input twice.\n", lineNum);
					return false;
				}
				if(!stringToDims(items[1],dims))
					return false;
				init(dims[0],dims[1],dims[2]);
				haveInput = true;
			}
			else if(!haveInput)
			{
				printf("Line %d: You need to have the input layer before any other layers.\n", lineNum);
				return false;
			}
			else if(items[0] == "conv")
			{
				int filSize = -1, pad = -1, stride = -1, numFil = -1;
				int dimIndex;
				for(int i = 1; i < items.size(); i++)
				{
					items[i] = tolower(items[i]);
					// cout << "lowered items[i]: " << items[i] << endl;
					if(items[i].find("numfil=") != string::npos)
						numFil = stoi(items[i].substr(items[i].find('=')+1));
					else if(items[i].find("filsize=") != string::npos)
						filSize = stoi(items[i].substr(items[i].find('=')+1));
					else if(items[i].find("pad=") != string::npos)
						pad = stoi(items[i].substr(items[i].find('=')+1));
					else if(items[i].find("stride=") != string::npos)
						stride = stoi(items[i].substr(items[i].find('=')+1));
					else if(items[i].find("x") != string::npos)
					{
						bool goodDims = stringToDims(items[i], dims);
						if(!goodDims) return false;
						dimIndex = i;
					}
					else
					{
						printf("Line %d: Unknown arg for Convolutional Layer \"%s\".\n",lineNum, items[i].c_str());
						return false;
					}
				}
				string errors = "";
				if(filSize <= 0)
				{
					sprintf(buf,"Line %d: filSize must exist and be positive\n",lineNum);
					errors += buf;
				}
				if(pad < 0)
				{
					sprintf(buf,"Line %d: pad must exist and be non-negative\n",lineNum);
					errors += buf;
				}
				if(stride <= 0)
				{
					sprintf(buf,"Line %d: stride must exist and be positive\n",lineNum);
					errors += buf;
				}
				if(numFil <= 0)
				{
					sprintf(buf,"Line %d: numFil must exist and be positive\n",lineNum);
					errors += buf;
				}
				if(errors != "")
				{
					printf("%s\n", errors.c_str());
					return false;
				}
				bool success = addConvLayer(numFil,stride,filSize,pad);
				if(!success)
				{
					printf("Line %d: Conv Layer failed to load successfully. Make sure the stride fits previous layer size.\n", lineNum);
					return false;
				}
				if(dims[0] != -1 && dims[1] != -1 && dims[2] != -1)
				{
					if(dims[0] != __neuronDims.back()[0] || dims[1] != __neuronDims.back()[1] || dims[2] != __neuronDims.back()[2])
					{
						printf("Line %d: The computed dimensions for conv layer do not match calculated.\n\t Given: %s, Calculated: %dx%dx%d\n", lineNum, items[dimIndex].c_str(),__neuronDims.back()[0],__neuronDims.back()[1],__neuronDims.back()[2]);
						return false;
					}
				}
			}
			else if(items[0] == "activ")
			{
				string type = tolower(items[1]);
				int activType;
				if(type == "relu")
					activType = RELU;
				else if(type == "leaky_relu")
					activType = LEAKY_RELU;
				else
				{
					printf("Line %d: Unknown or unimplemented activation type \"%s\". Try \"relu\" or \"leaky_relu\".\n", lineNum,items[1].c_str());
					return false;
				}
				bool success = addActivLayer(activType);
				if(!success)
				{
					printf("Line %d: Error adding activation layer with type %s\n", lineNum, items[1].c_str());
				}
			}
			else if(items[0] == "maxpool")
			{
				int stride = -1, pool = -1, dimIndex;
				for(int i = 1; i < items.size(); i++)
				{
					items[i] = tolower(items[i]);
					if(items[i].find("stride=") != string::npos)
						stride = stoi(items[i].substr(items[i].find('=')+1));
					else if(items[i].find("pool=") != string::npos)
						pool  = stoi(items[i].substr(items[i].find('=')+1));
					else if(items[i].find('x') != string::npos)
					{
						bool goodDims = stringToDims(items[i], dims);
						if(!goodDims) return false;
						dimIndex = i;						
					}
					else
					{
						printf("Line %d: Unknown arg for MaxPool Layer \"%s\".\n",lineNum, items[i].c_str());
						return false;
					}
				}

				string errors = "";
				if(stride <= 0)
				{
					sprintf(buf,"Line %d: stride must exist and be positive\n",lineNum);
					errors += buf;
				}
				if(pool <= 0)
				{
					sprintf(buf,"Line %d: pool must exist and be positive\n",lineNum);
					errors += buf;
				}
				if(errors != "")
				{
					printf("%s\n", errors.c_str());
					return false;
				}
				bool success = addMaxPoolLayer(pool, stride);
				if(!success)
				{
					printf("Line %d: MaxPool Layer failed to load correctly. Make sure stride fits previous layer size.\n", lineNum);
					return false;
				}
				if(dims[0] != -1 && dims[1] != -1 && dims[2] != -1)
				{
					if(dims[0] != __neuronDims.back()[0] || dims[1] != __neuronDims.back()[1] || dims[2] != __neuronDims.back()[2])
					{
						printf("Line %d: The computed dimensions for maxpool layer do not match calculated.\n\t Given: %s, Calculated: %dx%dx%d\n", lineNum, items[dimIndex].c_str(),__neuronDims.back()[0],__neuronDims.back()[1],__neuronDims.back()[2]);
						return false;
					}
				}
			}
			else if(items[0] == "fc")
			{
				// printf("fully connected\n");
				int outputSize = -1, dimIndex;
				if(items.size() > 3)
				{
					printf("Line %d: Too many args for Fully Connected Layer\n", lineNum);
					return false;
				}
				for(int i = 1; i < items.size(); i++)
				{
					items[i] = tolower(items[i]);
					if(items[i].find('x') != string::npos)
					{
						bool goodDims = stringToDims(items[i],dims);
						if(!goodDims) return false;
						dimIndex = i;
					}
					else
						outputSize = stoi(items[i]);
				}
				if(outputSize <= 0)
				{
					printf("Line %d: pool must exist and be positive\n",lineNum);
					return false;
				}
				bool success = addFullyConnectedLayer(outputSize);
				if(!success)
				{
					printf("Line %d: Error adding Fully Connected Layer.\n", lineNum);
					return false;
				}
				if(dims[0] != -1 && dims[1] != -1 && dims[2] != -1)
				{
					if(dims[0] != __neuronDims.back()[0] || dims[1] != __neuronDims.back()[1] || dims[2] != __neuronDims.back()[2])
					{
						printf("Line %d: The computed dimensions for fc layer do not match calculated.\n\t Given: %s, Calculated: %dx%dx%d\n", lineNum, items[dimIndex].c_str(),__neuronDims.back()[0],__neuronDims.back()[1],__neuronDims.back()[2]);
						return false;
					}
				}
			}
			else
			{
				printf("Line %d: Unknown arg \"%s\"\n", lineNum, line.c_str());
				return false;
			}
		}
		file.close();
		return true;
	}
	file.close();
	cout << "Unknown file format" << endl;
	return false;
}

Net::ConvLayer& Net::ConvLayer::operator=(const Net::ConvLayer& other)
{
	if(this != &other)
	{
		// printf("Copying convLayer\n");
		//see if we need to clean up our weights and biases
		if(weights != nullptr)
		{
			printf("Deleting old weights\n");
			delete weights;
		}
		if(biases != nullptr)
		{
			printf("Deleting old biases\n");
			delete biases;
		}

		//copy over everything
		numWeights = other.numWeights;
		numBiases = other.numBiases;
		numNeurons = other.numNeurons;
		padding = other.padding;
		stride = other.stride;
		filterSize = other.filterSize;
		paddedNeuronWidth = other.paddedNeuronWidth;
		paddedNeuronHeight = other.paddedNeuronHeight;
		paddedNeuronSize = other.paddedNeuronSize;
		maxSizeNeeded = other.maxSizeNeeded;

		weights = new double[numWeights];
		biases = new double[numBiases];
		// for(int i = 0; i < numWeights; i++)
		// 	weights[i] = other.weights[i];
		// for(int i = 0; i < numBiases; i++)
		// 	biases[i] = other.biases[i];
	}
	return *this;
}

double Net::estfpops() const
{
	double sum = 0;
	sum += estfpops_preprocessDataCollective(); // this does for 1 image
	for(int i = 0; i < __layers.size(); i++)
	{
		if(__layers[i]->layerType == CONV_LAYER)
			sum += estfpops_conv((ConvLayer*)__layers[i],i);
		else if(__layers[i]->layerType == ACTIV_LAYER)
			sum += estfpops_activ((ActivLayer*)__layers[i],i);
		else if(__layers[i]->layerType == MAX_POOL_LAYER)
			sum += estfpops_maxPool((MaxPoolLayer*)__layers[i],i);
	}
	sum += estfpops_softmax(); // uses __neuronSizes.back()
	return sum;
}

//this does for 1 image
double Net::estfpops_preprocessDataCollective() const
{
	double sum;
	double perPix = 4*fpop_array+fpop_assign+fpop_add+fpop_mult+fpop_compare+fpop_pluseq;
	sum = perPix * __neuronSizes[0];
	return sum;
}

double Net::estfpops_softmax() const
{
	double sum = 0;
	int s = __neuronSizes.back(); //number of classes

	//max subtraction. note always has 1 wu
	sum += (1+3*s)*fpop_compare + (3+2*s)*fpop_assign + (1+s)*fpop_array
		+ 3*fpop_pluseq;
	//end max subtraction

	//VecESum. always 1 wu
	sum += (1+s)*fpop_compare + 3*fpop_assign + s*fpop_array 
		+ 2*s*fpop_pluseq + s*fpop_exp;
	//end VecESum

	//softmax
	double soft = fpop_mult + fpop_assign + 2*fpop_array + fpop_exp;
	sum += soft * s;
	//end softmax
	return sum;
}

double Net::estfpops_conv(ConvLayer* conv, int layerNum) const
{
	double sum = 0;
	int f = conv->filterSize;
	int L = f * __neuronDims[layerNum-1][2]; // filtersize * prevDepth
	int fL = f * L;
	//per wu convolve
	double work = 0;
	work += (f + fL)*fpop_compare + 5*fpop_add
		+ (15 + fL)*fpop_mult + (12+f)*fpop_assign
		+ (3 + fL)*fpop_array + (2*f + 3*fL)*fpop_pluseq;
	sum += work * __neuronSizes[layerNum];
	//end per wu convolve

	//per wu zeropad
	double zero = 4*fpop_compare + 3*fpop_andor + 10*fpop_add
		+ 10*fpop_mult + 9*fpop_assign + 3*fpop_array;
	sum += zero * conv->paddedNeuronSize;
	//end per wu zeropad
	return sum;
}

double Net::estfpops_activ(ActivLayer* act, int layerNum) const
{
	double sum = 0;
	if(act->activationType == LEAKY_RELU || act->activationType == RELU)
	{
		//one neuron
		double leak = 0;
		leak += 3 * fpop_assign + 4 * fpop_compare + 2 * fpop_array;
		leak += 1 * fpop_mult + 1 * fpop_andor;

		//times all neurons
		leak *= __neuronSizes[layerNum];
		sum += leak;
	}
	return sum;
}

double Net::estfpops_maxPool(MaxPoolLayer* pool, int layerNum) const
{
	double sum;
	int p = pool->poolSize;

	//per workunit
	double work = 0;
	work += (p+2*p*p)*fpop_compare;
	work += 5*fpop_add + 10*fpop_mult;
	work += (9 + p + p*p)*fpop_assign;
	work += (2+2*p*p)*fpop_array;
	work += (2*p+2*p*p)*fpop_pluseq;
	//end per workunit
	sum = work * __neuronSizes[layerNum];
	return sum;
}

bool Net::ConvLayer::equals(const Net::ConvLayer& other)
{
	bool ret = true;
	ret = ret && (numWeights == other.numWeights);
	ret = ret && (numBiases == other.numBiases);
	ret = ret && (numNeurons == other.numNeurons);
	ret = ret && (padding == other.padding);
	ret = ret && (stride == other.stride);
	ret = ret && (filterSize == other.filterSize);
	ret = ret && (paddedNeuronWidth == other.paddedNeuronWidth);
	ret = ret && (paddedNeuronHeight == other.paddedNeuronHeight);
	ret = ret && (paddedNeuronSize == other.paddedNeuronSize);
	ret = ret && (maxSizeNeeded == other.maxSizeNeeded);
	if(!ret)
		return !ret;
	for(int i = 0; i < numWeights; i++)
		ret = ret && (weights[i] == other.weights[i]);
	for(int i = 0; i < numBiases; i++)
		ret = ret && (biases[i] == other.biases[i]);

	return ret;
}
