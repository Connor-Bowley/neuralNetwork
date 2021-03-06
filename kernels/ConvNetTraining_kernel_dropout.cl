// backprops that need the neuron vals.
//		Conv - keep the padded vals

// backprops that need info. 
// 		Activ  	1 means write the dneuron val into it.
//				0 means write 0 into it (it fell outside the range)
//	   		    .01 means write leakyReluConst * dneuron val into it

//		maxPool write the index of the maxval for the new neuron into a parallel array for the new neurons.
//			in backprop search that array for your index and if they match, += the dneurons value into it


#define RELU_CAP 5.0 		 //max value that can pass through relu or leakyRelu
#define LEAKY_RELU_CONST .01 //multiplication constant for negative values in leakyRelu
#define l2Lambda 0.1 //.05		 //multiplication constant for L2 Regularization
#define MOMENT_CONST .99 	 //multiplication constant for momentum
#define MAX_NORM_CAP 3.5 	 //max absolute value a weight can have

/*************************************************
*
*	ActivLayer kernels
*
*************************************************/

__kernel void relu(__global double* prevNeurons, __global double* neurons, __global double* dneuronInfo)
{
	const int i = get_global_id(0);
	if(prevNeurons[i] >= 0 && prevNeurons[i] <= RELU_CAP)
	{
		neurons[i] = prevNeurons[i];
		dneuronInfo[i] = 1;
	}
	else
	{
		dneuronInfo[i] = 0;
		if(prevNeurons < 0)
			neurons[i] = 0;
		else
			neurons[i] = RELU_CAP;
	}
}

//prevNeurons is the set of neurons in the prev layer, the set you are writing into
__kernel void relu_back(__global double* prevdNeurons, __global double* dneurons, __global double* dneuronInfo)
{
	const int i = get_global_id(0);
	prevdNeurons[i] = dneuronInfo[i] * dneurons[i];	
}


//numthreads should be size of neurons and prevNeurons (should be same)
__kernel void leakyRelu(__global double* prevNeurons, __global double* neurons, __global double* dneuronInfo)
{
	const int i = get_global_id(0);
	//double newVal = prevNeurons[i] > 0 ? prevNeurons[i] : prevNeurons[i] * .01; 
	double newVal;
	double dneur = LEAKY_RELU_CONST;
	if(prevNeurons[i] >= 0) 
	{
		newVal = prevNeurons[i];
		dneur = 1;
	}
	else 
		newVal = prevNeurons[i] * LEAKY_RELU_CONST;

	if(-RELU_CAP <= newVal && newVal <= RELU_CAP)
		neurons[i] = newVal;
	else
	{
		dneur = 0;
		if(newVal < -RELU_CAP)
			neurons[i] = -RELU_CAP;
		else
			neurons[i] = RELU_CAP;
	}
	dneuronInfo[i] = dneur;
}

__kernel void leakyRelu_back(__global double* prevdNeurons, __global double* dneurons, __global double* dneuronInfo)
{
	const int i = get_global_id(0);
	prevdNeurons[i] = dneuronInfo[i] * dneurons[i];
}

/*************************************************
*
*	MaxPool kernels
*
*************************************************/

//num threads should be the size of the neurons after the maxPool
__kernel void maxPool(__global double* prevNeurons, __global double* neurons,
	int prevwidth, int prevdepth, int poolsize, int stride, __global int* maxIndexes)
{
	int width = prevwidth;
	int depth = prevdepth;
	
	//getting the start index of a flattened 3d array for maxPool
	int x = get_global_id(0);
	int i = x;
	int strxdep = stride * depth;
	int i_div_dep = i / depth;
	int numBlocksPerRow = (width - poolsize)/stride + 1; 
	//int ourHeight = i/numBlocksPerRow/depth;
	//int ourRowStartIndex = ourHeight * width * stride * depth + i%depth;
	//int ourRowShift = ((i/depth)%numBlocksPerRow) * stride * depth;
	//int ourStartIndex = ourRowStartIndex + ourRowShift;
	//i = ourRowStartIndex + ourRowShift;

	i = (i_div_dep/numBlocksPerRow * width * strxdep + i%depth) + (((i_div_dep)%numBlocksPerRow) * strxdep);
	
	int amountToNextLayer = (width - poolsize) * depth;
	int maxIndex = i;
	double maxVal = prevNeurons[i];
	for(int row = 0; row < poolsize; row++)
	{
		for(int col = 0; col < poolsize; col++)
		{
			if(prevNeurons[i] >= maxVal)
			{
				maxVal = prevNeurons[i];
				maxIndex = i;
			}
			i += depth;
		}
		i += amountToNextLayer;
	}
	//printf("forward maxIndex %d",);
	neurons[x] = maxVal;
	maxIndexes[x] = maxIndex;
}

//run for each neuron in prevdNeurons
__kernel void maxPool_back(__global double* prevdNeurons, __global double* dneurons, __global int* maxIndexes, int numIndexes)
{
	const int i = get_global_id(0);
	double result = 0;
	for(int j=0; j< numIndexes; j++)
	{
		if(maxIndexes[j] == i)
		//if(*(maxIndexes) == i)
		{
			//printf("maxIndex %d, j %d, dneurons[j] %f\n",*maxIndexes, j, dneurons[j]);
			result += dneurons[j];
		}
		//maxIndexes++;
	}

	prevdNeurons[i] = result;
}

/*************************************************
*
*	Convolution kernels (and Zero Padding)
*
*************************************************/

//can keep the padded vals by using the calling code
__kernel void convolve(__global double* prevNeurons, __global double* neurons,
	__global double* weights, __global double* biases, int numFilters, int filterSize, int stride,
	 int prevwidth, int prevdepth, __global char* dropout, double p)
{
	//int myHeight = myBlock/numBlocksPerRow;
	//int myRowStartIndex = (myBlock/numBlocksPerRow) * width * strxdep;
	//int myRowShift = (myBlock%numBlocksPerRow) * strxdep;

	int i = get_global_id(0);
	/*
	if(dropout[i] == 0)
	{
		neurons[i] = 0;
		return;
	}*/

	int width = prevwidth;
	int depth = prevdepth;

	int numBlocksPerRow = (width - filterSize)/stride + 1;
	
	int myFilter = i%numFilters;
	int filterLayerSize = filterSize * depth;
	int j = myFilter * filterSize * filterLayerSize; // myFilterStartIndex
	int myBlock = (i/numFilters) % (numBlocksPerRow*numBlocksPerRow);//numBlocksCanFitInSource;
	
	int strxdep = stride * depth;
	int myStartIndex = ((myBlock/numBlocksPerRow) * width * strxdep) + ((myBlock%numBlocksPerRow) * strxdep);
	int h = myStartIndex;

	int amountToNextLayer = (width - filterSize) * depth;

	double result = 0;
	__global double* curWeight = &(weights[j]);
	for(int a = 0; a < filterSize; a++) //for each layer in the filter
	{
		for(int b = 0; b < filterLayerSize; b++)
		{
			//result += weights[j++] * prevNeurons[h++];
			result += *(curWeight++) * prevNeurons[h++];
		}
		h += amountToNextLayer;
	}
	//printf("numFil: %d id: %d myBlock: %d\n",numFilters,get_global_id(0), myBlock);
	//printf("In convolve. Global id = %d\n\tmyFilter = %d\n\tresult = %f\n",i,myFilter,result);
	neurons[i] = (result + biases[myFilter]) / p;
}

__kernel void convolve_back_neurons(__global double* prevdNeurons, __global double* dneurons,
	__global double* weights, int numFilters, int filterSize, int stride, int prevwidth, int depth, 
	__global char* dropout, double p)
{
	//calculated const variables
	const int x = get_global_id(0);
	/*
	if(dropout[x] == 0) // check to see if this neuron drops out
	{
		prevdNeurons[x] = 0;
		return;
	}*/
	const int numWeightsPerFilter = filterSize * filterSize * depth;
	const int filterLayerSize = filterSize * depth;
	const int numBlocksPerRow = (prevwidth - filterSize)/stride + 1;

	//variable declarations
	int start = 0; 
	int origStart;
	int d = 0;
	double result = 0;
	int endlayer;
	int placeInFilter;

	//calculations used a lot
	int filxdep = filterSize * depth;
	int widthxdepth = prevwidth * depth;
	int toNextRow = filxdep + widthxdepth * (stride-1);

	for(int a=0; a < numBlocksPerRow; a++) //change to numBlocksPerCol to all for non-square images
	{
		for(int b = 0; b < numBlocksPerRow; b++)
		{
			origStart = start;
			if(x < origStart)
			{
				prevdNeurons[x] = result;
				return;
			}
			for(int miniHeight = 0; miniHeight < filterSize; miniHeight++)
			{
				endlayer = start + filxdep;
				if(start <= x && x < endlayer)
				{
					placeInFilter = (x - start) + miniHeight*filxdep;
					for(int f=0; f < numFilters; f++)
					{
						result += weights[placeInFilter] * dneurons[d+f] * dropout[d+f] / p; //[d++];
						//printf("x %d weights %f dneurons %f result %f\n",x,weights[placeInFilter],dneurons[d+f],result);
						placeInFilter += numWeightsPerFilter; // gets us to same element in next filter
					}
					//if we found it in this minilayer, it wont be in any of the others
					break;
				}
				//else
				//d += numFilters; // only need to do if we don't go through them all in adding to result
				start += widthxdepth;
			}
			d += numFilters;
			start = origStart + depth;
		}
		start = origStart + toNextRow;
	}
	//printf("x %d final result %f\n",x,result);
	prevdNeurons[x] = result;
	//printf("x %d result %f\n",x,result);
	
}

__kernel void convolve_back_weights(__global double* weights, __global double* prevNeurons, __global double* dneurons,
	int depth, int stride, int prevwidth, int filterSize, int numFilters, double stepSize, __global char* dropout)
{
	
	int x = get_global_id(0);
	int numWeightsPerFilter = filterSize * filterSize * depth;
	int d = x/numWeightsPerFilter; //myFilter


	int p = x % numWeightsPerFilter; // my place in the filter

	int numBlocksPerRow = (prevwidth - filterSize)/stride + 1;
	double myDerivative = 0;

	int depxstr = depth * stride;
	int toNextBlockDown = filterSize*depth + prevwidth*depth*(stride-1);

	for(int a=0; a < numBlocksPerRow; a++)
	{
		for(int b = 0; b < numBlocksPerRow; b++) //change to b < numBlocksPerCol to allow for non-square images. would need prevheight
		{
			myDerivative += prevNeurons[p] * dneurons[d] * dropout[d] *.5;
			p += depxstr;
			d += numFilters;
		}
		p += toNextBlockDown;
	}

	//L2 Reg
	myDerivative += l2Lambda * weights[x];

	//max-norm
	double myWeight = weights[x];
	myWeight -= stepSize * myDerivative;
	if(myWeight > MAX_NORM_CAP)
		weights[x] = MAX_NORM_CAP;
	else if(myWeight < -MAX_NORM_CAP)
		weights[x] = -MAX_NORM_CAP;
	else
		weights[x] = myWeight;

	//w/out max-norm
	//weights[x] -= stepSize * myDerivative;
	
}
__kernel void convolve_back_weights_moment(__global double* weights, __global double* prevNeurons, __global double* dneurons,
	int depth, int stride, int prevwidth, int filterSize, int numFilters, double stepSize, __global double* velocity,
	__global char* dropout)
{
	
	int x = get_global_id(0);
	int numWeightsPerFilter = filterSize * filterSize * depth;
	int d = x/numWeightsPerFilter; //myFilter

	int p = x % numWeightsPerFilter; // my place in the filter

	int numBlocksPerRow = (prevwidth - filterSize)/stride + 1;
	double myDerivative = 0;

	int depxstr = depth * stride;
	int toNextBlockDown = filterSize*depth + prevwidth*depth*(stride-1);

	for(int a=0; a < numBlocksPerRow; a++)
	{
		for(int b = 0; b < numBlocksPerRow; b++) //change to b < numBlocksPerCol to allow for non-square images. would need prevheight
		{
			myDerivative += prevNeurons[p] * dneurons[d] * dropout[d] *.5;
			p += depxstr;
			d += numFilters;
		}
		p += toNextBlockDown;
	}

	//L2 Reg?
	myDerivative += l2Lambda * weights[x];// * weights[x];

	double myVel = velocity[x];
	myVel = MOMENT_CONST * myVel - stepSize * myDerivative;

	//max-norm
	double myWeight = weights[x];
	myWeight += myVel;
	if(myWeight > MAX_NORM_CAP)
		weights[x] = MAX_NORM_CAP;
	else if(myWeight < -MAX_NORM_CAP)
		weights[x] = -MAX_NORM_CAP;
	else
		weights[x] = myWeight;

	//w/out max-norm
	//weights[x] += myVel;
	//velocity[x] = myVel;
	
}

//should have numBiases work units
__kernel void convolve_back_biases(__global double* biases, __global double* dneurons, int dneuronSize, 
	int dneuronDepth, double stepSize, __global char* dropout)
{
	
	int i = get_global_id(0);
	int j = i;//%dneuronDepth //which filter we're in. dont need % because there is only one bias per filter
	double myDerivative = 0;
	int dneurFaceSize = dneuronSize/dneuronDepth;
	for(int a = 0; a< dneurFaceSize; a++) // calc myDerivative
	{
		myDerivative += dneurons[j] * dropout[j] *.5;
		j+=dneuronDepth;
	}
	//printf("stepSize %lf myDerivative %lf, result %.9lf\n", stepSize, myDerivative, stepSize * myDerivative);
	biases[i] -= stepSize * myDerivative; // update bias

}

__kernel void zeroPad(__global double *prevNeurons, __global double *neurons, int pad, int prevwidth,
	int prevheight, int depth)
{
	int x = get_global_id(0);

	//turn x into i, j, k
	const int nw = prevwidth + 2*pad;
	const int nh = prevheight + 2*pad;

	int ourDepth = x%depth;
	int ourCol = ((x-ourDepth)/depth) % nw;
	int ourRow = ((x-ourDepth)/depth) / nw;

	if(ourRow < pad || ourRow >= nh-pad || ourCol < pad || ourCol >= nw-pad)
		neurons[x] = 0;
	else
	{
		int i = ourRow - pad;
		int j = ourCol - pad;
		int k = ourDepth;
		int oldIndex = (i * prevwidth * depth) + (j * depth) + k;

		neurons[x] = prevNeurons[oldIndex];
	}
}

// run on each of the padded dneurons. NOT prevdNeurons!
__kernel void zeroPad_back(__global double* prevdNeurons, __global double* dneurons, int pad, int prevwidth,
	int prevheight, int depth)
{
	int x = get_global_id(0);

	//turn x into i, j, k
	int nw = prevwidth + 2*pad;
	int nh = prevheight + 2*pad;

	int ourDepth = x%depth;
	int ourCol = ((x-ourDepth)/depth) % nw;
	int ourRow = ((x-ourDepth)/depth) / nw;

	if(!(ourRow < pad || ourRow >= nh-pad || ourCol < pad || ourCol >= nw-pad))
	{
		int i = ourRow - pad;
		int j = ourCol - pad;
		int k = ourDepth;
		int oldIndex = (i * prevwidth * depth) + (j * depth) + k;

		prevdNeurons[oldIndex] = dneurons[x];
	}
}

/*************************************************
*
*	Softmax kernels
*
*************************************************/

__kernel void softmax(__global double *prevNeurons, __global double *neurons, double denominator)
{
	int i = get_global_id(0);
	neurons[i] = exp(prevNeurons[i])/denominator;
}

//pushes the derivatives into prevdNeurons
__kernel void softmax_back(__global double* dNeurons, __global double* neurons, int trueVal)
{
	int i = get_global_id(0);

	if(i == trueVal)
	{
		dNeurons[i] = neurons[i] - 1;
		//printf("neurons[i] on trueVal: %f\n", neurons[i]);
		//dNeurons[i] = neurons[i];
	}
	else
	{
		dNeurons[i] = neurons[i];
		//dNeurons[i] = neurons[i] - 1;
	}
}


/*************************************************
*
*	Helper kernels
*
*************************************************/

__kernel void copyArray(__global double* source, __global double* dest)
{
	int x = get_global_id(0);
	dest[x] = source[x];
}

