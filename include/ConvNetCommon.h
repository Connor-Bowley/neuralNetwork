/** 
 * \file ConvNetCommon.h
 * \brief Common functions, defines, typedefs, etc used throughout ConvNet
 *
 * All code used in multiple files should ultimately be moved here. This will
 * then compile down into a library to be linked with the other parts of theta
 * project, or used on its own.
 */

#ifndef ____ConvNetCommon__
#define ____ConvNetCommon__

#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/**
 * \macro GETMAX
 * \param x Variable 1
 * \parma y Variable 2
 * \return y if greater than x; otherwise x
 * \brief Gets the maximum value between two comparable values
 */
#define GETMAX(x,y) (((x) > (y)) ? (x) : (y))

namespace convnet {

// \typedef imVector
// \brief A triple-vector of doubles used for training
typedef std::vector<std::vector<std::vector<double>>> imVector;

/** 
 * \func secondsToString
 * \param seconds The time in seconds to convert
 * \return A string representation in hours, minutes, seconds
 * \brief Converts a number of seconds into a human-readable form
 */
std::string secondsToString(std::time_t seconds);
std::string secondsToString(float seconds);

/**
 * \func getTime
 * \param tim The string representation of the time in HH:mm:ss format
 * \return The numeric representation of the time in seconds
 * \brief Converts a given HH:mm:ss time into a number
 */
std::time_t getTime(std::string tim);

/**
 * \func getTime
 * \param tim The numeric representation of the time
 * \return The string representation of the time in HH:mm:ss format
 * \brief Converts a given number of time in seconds into an HH:mm:ss string
 */
std::string getTime(std::time_t tim);

/**
 * \func tolower
 * \param str The string to convert to lowercase
 * \return A copy of the str in lower case
 * \brief Converts an entire string to lowercase
 */
std::string tolower(std::string str);

/**
 * \func readVariable
 * \param in The input stream from which to read
 * \return The value of the variable read in
 * \brief Reads in a variable of an type that can be sizeof()'d
 */
template<typename T>
T readVariable(std::ifstream& in);

char readChar(std::ifstream& in);
unsigned char readUChar(std::ifstream& in);
short readShort(std::ifstream& in);
unsigned short readUShort(std::ifstream& in);
int readInt(std::ifstream& in);
unsigned int readUInt(std::ifstream& in);
float readFloat(std::ifstream& in);
double readDouble(std::ifstream& in);

/**
 * \func resize3DVector
 * \brief Resizes a 3D vector to the given dimensions.
 */
void resize3DVector(imVector& vect, int width, int height, int depth);

void setAll3DVector(imVector& vect, double val);

void squareElements3DVector(imVector& vect);

double vectorSum(std::vector<double> const& vect);

double vectorSumSq(std::vector<double> const& vect);

bool allElementsEquals(std::vector<double> const& vect);

void convert1DArrayTo3DVector(const double *array, int width, int height, int depth, imVector& dest);

bool convertBinaryToVector( const char *filename, std::vector<imVector>& dest, std::vector<double> *trueVals, std::vector<std::string> *names, 
                            std::vector<int> *trues, short& sizeByte, short& xSize, short& ySize, short& zSize);
bool convertBinaryToVector(const char *filename, std::vector<imVector>& dest);
bool convertBinaryToVector(const char *filename, std::vector<imVector>& dest, std::vector<double> *trueVals);
bool convertBinaryToVector( const char *filename, std::vector<imVector>& dest, std::vector<double> *trueVals, std::vector<std::string> *names, 
                            std::vector<int> *trues);


bool convertBinaryToVectorTest( const char *filename, std::vector<imVector>& dest, std::vector<double> *trueVals, 
                                short sizeByte, short xSize, short ySize, short zSize);

};


#endif // ____ConvNetCommon__