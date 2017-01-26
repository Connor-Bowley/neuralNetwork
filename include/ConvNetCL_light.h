//
//  ConvNetCL_light.h
//  
//  not to be used WITH ConvNetCL but in place of ConvNetCL for info purposes only
//  Created by Connor Bowley on 3/15/16.
//
//

#ifndef ____ConvNetCL_light__
#define ____ConvNetCL_light__

#include <string>
#include <vector>
#include <time.h>

//defines for layers
#define ABSTRACT_LAYER -1
#define CONV_LAYER 0
#define MAX_POOL_LAYER 1
#define ACTIV_LAYER 2

//defines for ActivTypes
#define RELU 0
#define LEAKY_RELU 1
#define MAX_ACTIV 2

typedef std::vector<std::vector<std::vector<double> > > imVector;

class Net{
public:     // structs
	struct ClassInfo{
		std::string name = "";
		int trueVal = -1;
	};
private: 	// structs
	struct Layer{
		int layerType = ABSTRACT_LAYER;
	};

	struct ConvLayer : Layer{
		// int layerType = CONV_LAYER;
		double* weights = nullptr;
		double* biases = nullptr;
		int numWeights;
		int numBiases;
		int numNeurons;
		int padding;
		int stride;
		int filterSize;
		int paddedNeuronWidth;
		int paddedNeuronHeight;
		int paddedNeuronSize;
		int maxSizeNeeded;

		ConvLayer& operator=(const ConvLayer& other);
		bool equals(const ConvLayer& other);
	};

	struct MaxPoolLayer : Layer{
		int stride;
		int poolSize;
	};

	struct ActivLayer : Layer{
		int activationType;
	};

	struct WeightHolder{
		double trainAccuracy = 0;
		double testAccuracy = 0;
		std::vector<double*> weights; //weights[convLayer][weight]
		std::vector<double*> biases;

		~WeightHolder();
		void clearWeights();
	};

private: 	// members
	bool __inited = false;
	//hyperparameters
	double __learningRate = 1e-4;
	double __RELU_CAP = 5000.0;
	double __LEAKY_RELU_CONST = 0.01;
	double __l2Lambda = 0.05;
	double __MOMENT_CONST = 0.9;
	double __MAX_NORM_CAP = 6.0;
	
	//members dealing with layers
	std::vector<Layer*> __layers;  //[0] is input layer
	std::vector<int> __neuronSizes; //[0] is input layer
	std::vector<std::vector<int> > __neuronDims;  //[0] is input layer
	bool __autoActivLayer = true;
	int __maxNeuronSize;
	int __defaultActivType = 0;
	int __maxWeightSize = 0;

	bool __isFinalized = false;
	std::string __errorLog;

	//data and related members
	int __numClasses = 0;
	std::vector<ClassInfo> __classes;
		//training
		bool __trainingDataPreprocessed = false;
		bool __testDataPreprocessed = false;
		bool __preprocessIndividual = false;
		double __mean = 0;
		double __stddev = 0;
		unsigned long __trainingSize = 0;
		bool __isTraining = false;
		std::vector<std::vector<std::vector<double>* > > __trainingData; // class<list of<pointers-to-flattenedImages> >
		std::vector<double> __trueVals; // parallel vector of true values for __trainingData
		std::vector<std::vector<double> > __testData;
		std::vector<double> __testTrueVals;
		bool __useMomentum = true;
		int __smallestClassSize;
		std::string __saveName;
		bool __saveNet = false;
		//running
		bool __dataPreprocessed = false;
		std::vector<std::vector<double> > __data; // list of<flattened images>
		std::vector<std::vector<double> > *__dataPointer;
		std::vector<std::vector<double> > __confidences; // image<list of confidences for each class<confidence> > 

	int fpop_mult = 8;
	int fpop_add = 1;
	int fpop_compare = 0;
	int fpop_andor = 0;
	int fpop_array = 0;
	int fpop_assign = 0;
	int fpop_pluseq = fpop_add + fpop_assign;
	int fpop_timeseq = fpop_mult + fpop_assign;
	int fpop_exp = 20;

public: 	// functions
	//Constructors and Destructors
	Net();
	Net(const char* filename);
	~Net();

	//Equals
	Net& operator=(const Net& other);
	
	//functions dealing with layers and sizes
	void printLayerDims() const;
	int getInputWidth() const;
	int getInputHeight() const;

	int getNumClasses() const;
	void getClassNames(std::vector<ClassInfo>& infos) const;
	std::string getClassForTrueVal(int trueVal) const;

	bool load(const char* filename);
	double estfpops() const;

private:	// functions

	double estfpops_preprocessDataCollective() const;
	double estfpops_conv(ConvLayer* conv, int layerNum) const;
	double estfpops_activ(ActivLayer* act, int layerNum) const;
	double estfpops_maxPool(MaxPoolLayer* pool, int layerNum) const;
	double estfpops_softmax() const;

	bool addActivLayer();
	bool addActivLayer(int activationType);
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad);
	bool addMaxPoolLayer(int poolSize, int stride);
	bool addFullyConnectedLayer(int outputSize);
	bool setActivType(int activationType);
	void setAutoActivLayer(bool isAuto);
	void init(int inputWidth, int inputHeight, int inputDepth);
	//functions for operator=
	void copyLayers(const Net& other);

	//inits

	//functions dealing with layers
	bool addConvLayer(int numFilters, int stride, int filterSize, int pad, const std::string& weightsAndBias);
	void pushBackLayerSize(int width, int height, int depth);

	//weights and biases
	void initRandomWeights(ConvLayer* conv, int prevDepth);
	void initWeights(ConvLayer* conv, const std::string& weights);

	//functions dealing with data
	int getTrueValIndex(double trueVal);

	//training

	//other
	std::string tolower(std::string str);
	bool stringToDims(std::string str, int* dims);
	std::string secondsToString(time_t seconds);


	//OpenCL functions

};

#endif /* defined(____ConvNetCL__) */