/************************************
*
*	ConvNetDriver
*
*	Created by Connor Bowley on 5/2/16
*
*	This program is for running trained CNNs that were trained using ConvNet.
*
*	Usage:
*		./ConvNetDriver cnnConfig.txt binaryImagesFile outputName.txt append(defaults to false)
*
*************************************/

#include "ConvNet.h"

#include <cctype>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>


using namespace std;
using namespace convnet;

void saveClassesToFile(const char* filename, const vector<int>& calcedClasses, bool append)
{
	ofstream out;
	char buffer[50];
	int size;
	if(append)
		out.open(filename,ios::app);
	else
		out.open(filename,ios::trunc);

	for(int i=0; i < calcedClasses.size(); i++)
	{
		sprintf(buffer,"%d\n",calcedClasses[i]);
		size = 0;
		while(buffer[size] != '\0')
			size++;
		out.write(buffer,size);
	}
	out.close();
}

int main(int argc, char** argv)
{
	if(argc != 4 && argc != 5)
	{
		cout << "Usage: ./ConvNetDriver cnnConfig.txt binaryImagesFile outputName.txt append(true or false, defaults to false)" << endl;
		return 0;
	}

	bool append = false;
	if(argc == 5)
	{
		int i = 0;
		while(argv[4][i] != '\0')
		{
			argv[4][i] = tolower(argv[4][i]);
			i++;
		}
		string arg(argv[4]);
		if(arg.find("true") != string::npos)
			append = true;
	}

	time_t starttime, endtime;

	//set up net
	Net net(argv[1]);

	if(net.numLayers() < 0)
	{
		cout << "Net must have at least one layer. Exiting. Make sure the cnnConfig is correct." << endl;
		return 0;
	}

	//get images and preprocess them
	vector<imVector> images(0);
	short sizeByte, xSize, ySize, zSize;

	starttime = time(NULL);
	if (!convertBinaryToVector(argv[2], images)) {
		cout << "Exiting." << endl;
		return 0;
	}
	endtime = time(NULL);
	cout << "Time for bringing in binary: " << secondsToString(endtime - starttime) << endl;
	

	preprocess(images);

	net.addRealData(images);

	vector<int> calcedClasses(0);

	//cout << "Starting OpenCL Run" << endl;
	starttime = time(NULL);
	net.newRun(calcedClasses,false);
	endtime = time(NULL);
	cout << "Time for OpenCL code: " << secondsToString(endtime - starttime) << endl;

	saveClassesToFile(argv[3], calcedClasses, append);
}