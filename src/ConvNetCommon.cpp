#include "ConvNetCommon.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>

namespace convnet {

bool getNextImage(std::ifstream& in, imVector& dest, short x, short y, short z, short sizeByte, double *trueVal = nullptr);

std::string secondsToString(std::time_t seconds)
{
	std::time_t secs = seconds%60;
	std::time_t mins = (seconds%3600)/60;
	std::time_t hours = seconds/3600;
    std::ostringstream s; 

	if(hours > 0)
		s << hours << " hours, ";
    if(mins > 0)
		s << mins << " mins, ";
	s << secs << " secs";

	std::string out = s.str();
	return out;
}

std::string secondsToString(float seconds)
{
	float secs = ((int)seconds % 60) + (seconds - (int)seconds);
	std::time_t mins = ((int)seconds % 3600) / 60;
	std::time_t hours = (int)seconds/3600;
	std::ostringstream s; 

	if(hours > 0)
		s << hours << " hours, ";
	if(mins > 0)
		s << mins << " mins, ";

	s << std::fixed << std::setprecision(2) << secs;
	std::string out = s.str();
	return out;
}

std::time_t getTime(std::string tim)
{
	std::time_t t = 0;
	t += std::stoi(tim.substr(tim.rfind(':')+1)); // seconds
	t += 60 * std::stoi(tim.substr(tim.find(':')+1, 2)); //minutes
	t += 3600 * std::stoi(tim.substr(0,2)); //hours

	return t;
}

std::string getTime(std::time_t tim)
{
	int seconds = tim % 60;
	tim /= 60; //now tim is in minutes
	int minutes = tim % 60;
	int hours = tim / 60;
	std::ostringstream s;

	if(hours < 10)
		s << "0";
	s << hours << ":";

	if(minutes < 10)
		s << "0";
	s << minutes << ":";

	if(seconds < 10)
		s << "0";
	s << seconds;
	
	std::string out = s.str();
	return out;
}

std::string tolower(std::string str)
{
	std::transform(str.begin(), str.end(), str.begin(), std::tolower);
	return str;
}

template<typename T>
T readVariable(std::ifstream& in)
{
	T var;
	in.read((char *)(&var), sizeof(var));
	return var;
}

char readChar(std::ifstream& in) { return readVariable<char>(in); }
unsigned char readUChar(std::ifstream& in) { return readVariable<unsigned char>(in); }
short readShort(std::ifstream& in) { return readVariable<short>(in); }
unsigned short readUShort(std::ifstream& in) { return readVariable<unsigned short>(in); }
int readInt(std::ifstream& in) { return readVariable<int>(in); }
unsigned int readUInt(std::ifstream& in) { return readVariable<unsigned int>(in); }
float readFloat(std::ifstream& in) { return readVariable<float>(in); }
double readDouble(std::ifstream& in) { return readVariable<double>(in); }

void resize3DVector(imVector& vect, int width, int height, int depth)
{
	vect.resize(width);
	for (auto&& i : vect) {
		i.resize(height);

		for (auto&& j : i) {
			j.resize(depth);
		}
	}
}

void setAll3DVector(imVector& vect, double val)
{
	for (auto&& i : vect) {
		for(auto&& j : i) {
			for(auto&& k : j) {
				k = val;
			}
		}
	}
}

double vectorSum(std::vector<double> const& vect)
{
	double sum = 0.0;
	for (auto&& i : vect)
		sum += i;
	return sum;
}

double vectorSumSq(std::vector<double> const& vect)
{
	double sum = 0.0;
	for (auto&& i : vect)
		sum += i * i;
	return sum;
}

bool allElementsEquals(std::vector<double> const& vect)
{
	for (auto&& i : vect) {
		if(vect[0] != i)
			return false;
	}

	return true;
}

void squareElements3DVector(imVector& vect)
{
	for (auto&& i : vect) {
		for(auto&& j : i) {
			for(auto&& k : j) {
				k *= k;
			}
		}
	}
}

template<typename T, typename A, typename B, typename C>
void convert1DArrayTo3DVector(const T *array, int width, int height, int depth, std::vector<std::vector<std::vector<T, C>, B>, A> &dest)
{
	resize3DVector<T>(vect, width, height, depth)
	
	// TODO: speed this up with foreach / iterator
	for(int i=0; i < width; ++i) {
		for(int j=0; j < height; ++j) {
			for(int k=0; k < depth; ++k) {
				dest[i][j][k] = *array++;
			}
		}
	}
}

bool getNextImage(std::ifstream& in, imVector& dest, short x, short y, short z, short sizeByte, double *trueVal)
{
	resize3DVector(dest, x, y, z);

	for(int i=0; i < x; ++i) {
		for(int j=0; j < y; ++j) {
			for(int k=0; k < z; ++k) {
				if(sizeByte == 1)
					dest[i][j][k] = (double)readUChar(in);
				else if(sizeByte == -1)
					dest[i][j][k] = (double)readChar(in);
				else if(sizeByte == 2)
					dest[i][j][k] = (double)readUShort(in);
				else if(sizeByte == -2)
					dest[i][j][k] = (double)readShort(in);
				else if(sizeByte == 4)
					dest[i][j][k] = (double)readUInt(in);
				else if(sizeByte == -4)
					dest[i][j][k] = (double)readInt(in);
				else if(sizeByte == 5)
					dest[i][j][k] = (double)readFloat(in);
				else if(sizeByte == 6)
					dest[i][j][k] = readDouble(in);
				else
				{
					std::cout << "Unknown sizeByte: " << sizeByte << "." << std::endl;
					return false;
				}
			}
		}
	}

	if (trueVal)
		*trueVal = (double)readUShort(in);

	return true;
}

bool convertBinaryToVectorTest(	const char *filename, std::vector<imVector>& dest, std::vector<double> *trueVals, 
								short sizeByte, short xSize, short ySize, short zSize)
{
	short testSizeByte, testXSize, testYSize, testZSize;
	if (!convertBinaryToVector(filename, dest, trueVals, nullptr, nullptr, testSizeByte, testXSize, testYSize, testZSize)) {
		return false;
	}

	if (sizeByte != testSizeByte || xSize != testXSize || ySize != testYSize || zSize != testZSize) {
		std::cout << "Dimensions of the test data does not match the training data." << std::endl;
		return false;
	}

	return true;
}

bool convertBinaryToVector(	const char *filename, std::vector<imVector>& dest, std::vector<double> *trueVals, std::vector<std::string> *names,
							std::vector<int> *trues, short& sizeByte, short& xSize, short& ySize, short& zSize)
{
	std::ifstream in;
	in.open(filename, std::ios::binary);
	if(!in.is_open()) {
		std::cout << "Failed to open " << filename << "." << std::endl;
		return false;
	}

	// determine the size of the file
	in.seekg(0, in.end);
	long end = in.tellg();
	in.seekg(0, in.beg);

	// read in the size, x, y, z
	sizeByte = readShort(in);
	xSize = readShort(in);
	ySize = readShort(in);
	zSize = readShort(in);

	if(xSize == 0 || ySize == 0 || zSize == 0) {
		std::cout << "None of the dimensions can be zero." << std::endl;
		return false;
	}

	if (names && trues) {
		char oldOrNew = readChar(in);

		if(oldOrNew == '\0') { // new
			int numClasses = readInt(in);

			for(int i = 0; i < numClasses; ++i)
			{
				unsigned int tru = readUInt(in);
				char ch;
				std::string name = "";

				while((ch = readChar(in)) != '\0') {
					name += ch;
				}

				(*names).push_back(name);
				(*trues).push_back(tru);
			}
		}
		else { // old
			in.seekg(-1, in.cur);
		}
	}

	// trueVals isn't always needed
	double* trueVal = nullptr;
	if (trueVals)
		trueVal = new double;

	while((long)in.tellg() != end) {
		if((long)in.tellg() > end) {
			std::cout << "The counter went past the max. There might be something wrong with the file format." << std::endl;
			break;
		}

		dest.resize(dest.size() + 1);
		if (!getNextImage(in, dest.back(), xSize, ySize, zSize, sizeByte, trueVal)) {
			delete trueVal;
			return false;
		}

		// add our trueVal, if needed
		if (trueVals)
			(*trueVals).push_back(*trueVal);
	}

	delete trueVal;
	std::cout << "Num images = " << dest.size() << std::endl;
	return true;
}

bool convertBinaryToVector( const char *filename, std::vector<imVector>& dest, std::vector<double> *trueVals, std::vector<std::string> *names, 
                            std::vector<int> *trues) {
    short sizeByte, xSize, ySize, zSize;
    return convertBinaryToVector(filename, dest, trueVals, nullptr, nullptr, sizeByte, xSize, ySize, zSize);
}

bool convertBinaryToVector(const char *filename, std::vector<imVector>& dest, std::vector<double> *trueVals) {
	return convertBinaryToVector(filename, dest, trueVals, nullptr, nullptr);
}

bool convertBinaryToVector(const char *filename, std::vector<imVector>& dest) {
	return convertBinaryToVector(filename, dest, nullptr, nullptr, nullptr);
}

};