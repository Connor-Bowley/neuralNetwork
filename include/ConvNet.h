//
//  ConvNet.h
//  
//
//  Created by Connor Bowley on 3/15/16.
//
//

#ifndef ____ConvNet__
#define ____ConvNet__

//#include <stdio.h>
#include <vector>
#include <string>

#ifdef __APPLE__
 	#include "OpenCL/opencl.h"
#else
 	#include "CL/cl.h"
#endif


// Classes

class Layer{
public:
	virtual ~Layer(){};
	virtual int getType() const = 0;
	virtual int getNumNeurons() const = 0;
	virtual std::string getNeuronDims() const = 0;
	virtual unsigned long getMem() const = 0;
	virtual void forwardprop(const Layer& prevLayer) = 0;
	virtual void backprop(Layer& prevLayer) = 0;
	virtual const std::vector<std::vector<std::vector<double> > >& getNeurons() const = 0;
	virtual std::vector<std::vector<std::vector<double> > >& getdNeurons() = 0;
};

class InputLayer : public Layer{
public:
	InputLayer();
	~InputLayer();
	int getType() const;
	int getNumNeurons() const;
	std::string getNeuronDims() const;
	unsigned long getMem() const;
	double* getImage() const;
	void getImage(double* dest, int size) const;
	int getImageSize() const;
	InputLayer(const std::vector<std::vector<std::vector<double> > >& trainingImage, std::vector<std::vector<std::vector<double> > >* blankdNeurons);
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
	bool setImage(const std::vector<std::vector<std::vector<double> > >* trainingImage,std::vector<std::vector<std::vector<double> > >* blankdNeurons);
private:
	static const int i_type;
	bool i_resizeable;
	int i_numNeurons;
	const std::vector<std::vector<std::vector<double> > >  *i_neurons;
	std::vector<std::vector<std::vector<double> > > *i_dneurons;
};

class ConvLayer : public Layer{
public:
	ConvLayer(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad);
	ConvLayer(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad, std::string weightsAndBiases);
	~ConvLayer();
	int getType() const;
	int getNumNeurons() const;
	unsigned long getMemWeightsAndBiases() const;
	unsigned long getMem() const;
	int getPaddedNeuronSize() const;
	int getMaxSizeNeeded() const;
	double* getWeights() const;
	int getNumWeights() const;
	double* getBiases() const;
	int getNumBiases() const;
	std::string getNeuronDims() const;
	void setBiases(double* biases);
	void setWeights(double* weights);
	std::vector<int> getKernelHyperParameters() const;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	std::string getHyperParameters() const;
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
private:
	void _putWeights(double* weights, int vectIndex) const;
	void init(const Layer& prevLayer, int numFilters, int stride, int filterSize, int pad);
	void initRandomWeights();
	void initWeights(std::string weights);
	static const int c_type;
	std::vector<std::vector<std::vector<double> > > c_neurons;
	std::vector<std::vector<std::vector<double> > > c_dneurons;
	std::vector<std::vector<std::vector<std::vector<double> > > > c_weights;
	std::vector<std::vector<std::vector<std::vector<double> > > > c_dweights;
	int c_numWeights;
	int c_numBiases;
	int c_numNeurons;
	std::vector<double> c_biases;
	std::vector<double> c_dbiases;
	int c_padding;
	int c_stride;
	int c_prevNeuronWidth;
	int c_prevNeuronHeight;
	int c_prevNeuronDepth;
	int c_maxSizeNeeded;
	int c_paddedNeuronSize;
};

class MaxPoolLayer : public Layer{
public:
	MaxPoolLayer(const Layer& prevLayer, int poolSize, int stride);
	~MaxPoolLayer();
	int getType() const;
	int getNumNeurons() const;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	unsigned long getMem() const;
	std::string getNeuronDims() const;
	std::vector<int> getKernelHyperParameters() const;
	std::string getHyperParameters() const;
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
private:
	static const int m_type;
	std::vector<std::vector<std::vector<double> > > m_neurons;
	std::vector<std::vector<std::vector<double> > > m_dneurons;
	int m_stride;
	int m_poolSize;
	int m_numNeurons;
	int m_prevWidth;
	int m_prevDepth;
};

class ActivLayer : public Layer{
public:
	ActivLayer(const Layer& prevLayer, const int activationType);
	~ActivLayer();
	//ACTIV_TYPES (exp. RELU) must start at 0 and go up by one. The check to see if a type is valid is
	// if(0 <= type && type < ACTIV_NUM_TYPES) then Valid
	static const int RELU = 0;
	static const int LEAKY_RELU = 1;
	static const int NUM_ACTIV_TYPES = 2;
	int getType() const;
	unsigned long getMem() const;
	int getNumNeurons() const;
	std::string getNeuronDims() const;
	int getActivationType() const;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	std::string getHyperParameters() const;
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
private:
	static const int a_type;
	static const double RELU_CAP;
	static const double LEAKY_RELU_CONST;
	int a_activationType;
	std::vector<std::vector<std::vector<double> > > a_neurons;
	std::vector<std::vector<std::vector<double> > > a_dneurons;
	int a_numNeurons;
};

class SoftmaxLayer : public Layer{
public:
	SoftmaxLayer(const Layer& prevLayer);
	~SoftmaxLayer();
	int getPredictedClass();
	std::vector<double> getError();
	void setError(std::vector<double> error);
	int getType() const;
	std::string getNeuronDims() const;
	unsigned long getMem() const;
	int getNumNeurons() const;
	void forwardprop(const Layer& prevLayer);
	void backprop(Layer& prevLayer);
	void setTrueVal(int trueVal);
	void gradientCheck(Layer& prevLayer);
	const std::vector<std::vector<std::vector<double> > >& getNeurons() const;
	std::vector<std::vector<std::vector<double> > >& getdNeurons();
private:
	static const int s_type;
	std::vector<double> s_neurons;
	std::vector<std::vector<std::vector<double> > > s_3neurons;
	std::vector<double> s_dneurons;
	std::vector<std::vector<std::vector<double> > > s_3dneurons;
	int s_trueVal;
};

class Net{
public:
	Net(const char* filename);
	Net(int inputWidth, int inputHeight, int inputDepth);
	
	~Net();
	void forwardprop();
	void backprop();
	void printLayerDims();
	static double stepSize;
	static const int CONV_LAYER = 0;
	static const int MAX_POOL_LAYER = 1;
	static const int ACTIV_LAYER = 2;
	static const int INPUT_LAYER = 3;
	static const int SOFTMAX_LAYER = 4;
	static const bool walkthrough = false;
	static const bool showErrors = false;
	static const bool useMomentum = true;
	static bool gradCheck;
	static const double GRADCHECK_H;

	void debug();

	int numLayers();
	bool addActivLayer();
	bool addActivLayer(int activationType);
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad);
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad, std::string weightsAndBiases);
	bool addMaxPoolLayer(int poolSize, int stride);
	bool addSoftmaxLayer();
	void addTrainingData(const std::vector<std::vector<std::vector<std::vector<double> > > >& trainingData, std::vector<double>& trueVals);
	void clear();
	void OpenCLTrain(int epochs, bool useGPU = true);
	void newRun(std::vector<int>& calculatedClasses, bool useGPU = true);
	void getConfidences(std::vector<std::vector<double> >& confidences) const;
	bool setActivType(int activationType);
	void train(int epochs);
	void splitTrain(int epochs, bool useGPU = true);
	void runTrainingData();
	void miniBatchTrain(int epochs, int batchSize);
	void addRealData(const std::vector<std::vector<std::vector<std::vector<double> > > >& realData);
	void setData(const std::vector<std::vector<std::vector<std::vector<double> > > >& data);
	void run(bool useGPU=true);
	std::vector<std::vector<std::vector<double> > >* getBlankVectorPointer();
	int getPredictedClass();
	int getNumCategories() const;
	void gradientCheck();
	bool isActive() const;
	double calcLoss(int indexOfTrueVal);
	void shuffleTrainingData(int times=1);
	unsigned long getMem() const;
	bool setDevice(unsigned int device);
	unsigned long getMemForward() const;
	void printTrainingDistribution() const;

	bool save(const char* filename);
private:
	static int n_activationType;
	std::vector<InputLayer*> n_trainingData;
	std::vector<double> n_results;
	std::vector<std::vector<double> > n_confidences;
	std::vector<double> n_trainingDataTrueVals;
	std::vector<std::vector<std::vector<double> > > n_blankVector;
	InputLayer n_blankInput;
	std::vector<Layer*> n_layers;
	//bool n_training;
	bool n_hasConvLayer, n_hasMaxPoolLayer, n_hasRELULayer, n_hasLeakyRELULayer, n_hasSoftmax;
	int n_numCategories;

	bool load(const char* filename);
	void init(int, int, int);
	void initOpenCL();
	unsigned long getMaxNeuronSize() const;

	//opencl stuff
	cl_uint platformIdCount;
	cl_uint deviceIdCount;
	std::vector<cl_platform_id> platformIds;
	std::vector<cl_device_id> deviceIds;
	cl_context context;
	cl_uint __device;
};



//OpenCL helper functions
std::string LoadKernel (const char* name);

void CheckError (cl_int error);

cl_program CreateProgram (const std::string& source, cl_context& context);

// Other Functions
void padZeros(const std::vector<std::vector<std::vector<double> > > &source, int numZeros, std::vector<std::vector<std::vector<double> > > &dest);

void printVector(const std::vector<std::vector<std::vector<std::vector<double> > > > &vect);

void printVector(const std::vector<std::vector<std::vector<double> > > &vect);

void printVector(const std::vector<double>& vect);

void printArray(double* array, int size);

void resize3DVector(std::vector<std::vector<std::vector<double> > > &vect, int width, int height, int depth);

void setAll3DVector(std::vector<std::vector<std::vector<double> > > &vect, double val);

void setAll4DVector(std::vector<std::vector<std::vector<std::vector<double> > > > &vect, double val);

void softmax(const std::vector<std::vector<std::vector<double> > > &vect, std::vector<double>& normedPredictionsContainer);

void maxSubtraction(std::vector<double>& vect);

void meanSubtraction(std::vector<double>& vect);

void vectorSubtraction(std::vector<std::vector<std::vector<double> > >& vect, double toSub);

void vectorDivision(std::vector<std::vector<std::vector<double> > >& vect, double toDiv);

void vectorDivision(std::vector<std::vector<std::vector<std::vector<double> > > >& vect, double toDiv);

double mean(const std::vector<std::vector<std::vector<double> > >& vect);

double mean(const std::vector<std::vector<std::vector<std::vector<double> > > >& vect);

double stddev(const std::vector<std::vector<std::vector<double> > >& vect);

double stddev(const std::vector<std::vector<std::vector<double> > >& vect, double mean);

double stddev(const std::vector<std::vector<std::vector<std::vector<double> > > >& vect);

double stddev(const std::vector<std::vector<std::vector<std::vector<double> > > >& vect, double mean);

void preprocess(std::vector<std::vector<std::vector<double> > >& vect);

void preprocess(std::vector<std::vector<std::vector<std::vector<double> > > >& vect);

void preprocessCollective(std::vector<std::vector<std::vector<std::vector<double> > > >& vect);

void preprocessByFeature(std::vector<std::vector<std::vector<std::vector<double> > > >& vect);

void meanSubtraction(std::vector<std::vector<std::vector<double> > >& vect);

void meanSubtraction(std::vector<std::vector<std::vector<std::vector<double> > > >& vect);

void compressImage(std::vector<std::vector<std::vector<std::vector<double> > > >& vect, double newMin, double newMax);

void compressImage(std::vector<std::vector<std::vector<double> > >& vect, double newMin, double newMax);

double vectorESum(const std::vector<double> &source);

int getMaxElementIndex(const std::vector<double> &vect);

double getMaxElement(const std::vector<double> &vect); 

void vectorClone(const std::vector<std::vector<std::vector<double> > > &source, std::vector<std::vector<std::vector<double> > > &dest);

#endif /* defined(____ConvNet__) */
