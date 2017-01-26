/***************************************
*
*
*	TrainingImageSplitterFileCreator takes a set of input images of specified classes, splits them into specified sized subimages and converts the subimages to a single binary file.
*
* 	All input images must be of the same size, specified in command line arguments. The format of the
*	output file is this:
*
*		sizeByte xsize ysize zsize \0 numClasses trueVal   classname_c_str   image1 trueVal1 image2 trueVal2... imageN trueValN
*		short	 short short short    int         uint   char[] ended by \0         ushort          ushort
*
*	The sizeByte says how large/what type each input is. 
*		1 - unsigned byte 		-1 - signed byte
*		2 - unsigned short 		-2 - signed short
* 		4 - unsigned int 		-4 - signed int
* 		5 - float
*		6 - double
*
*
*
*	Usage: 
*		See usage statement in main
*
****************************************/


#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unordered_map>
#include <random>
#include <cstdlib>

using namespace std;
using namespace cv;

#define MAX_ADJUST 0.3

struct imstruct{
	string name;
	unsigned long count = 0;
};

short xsize = -1, ysize = -1, zsize = -1;

int imageNum = 0;
int stride = 1;
int globalStride = 1;
unsigned long byteCount = 0;
unsigned long imageCount = 0;
unsigned long unalteredImageCount = 0;
unsigned short __globalTrueVal;
unordered_map<unsigned short, int> trueMap;
vector<imstruct> counts;


bool __horizontalReflect = false;
bool __verticalReflect = false;
bool __rotateClock = false;
bool __rotateCClock = false;
bool __rotate180 = false;
bool __brightnessAlterations = false;
double __brightnessAlterationChance = 0.0;
int numWritesPerImage = 1;
bool __output = true;

default_random_engine gen(time(0));
uniform_real_distribution<double> distr(-MAX_ADJUST,MAX_ADJUST);
uniform_real_distribution<double> distr1(0,1);

int compareRev(const void* p1, const void* p2)
{
	imstruct* im1 = (imstruct*) p1;
	imstruct* im2 = (imstruct*) p2;
	if(im1->count > im2->count) return -1;
	if(im1->count == im2->count) return 0;
	return 1;
}

string getParallelName(const vector<string>& names, const vector<int>& trueVals, int trueVal)
{
	for(int i = 0; i < names.size(); i++)
	{
		if(trueVals[i] == trueVal)
			return names[i];
	}
	return string("");
}

int getMaxNameSize(const vector<string>& names)
{
	int max = 0;
	for(int i = 0; i < names.size(); i++)
		if(names[i].length() > max)
			max = names[i].length();
	return max;
}

unsigned char clamp (int val, int min, int max)
{
	if(val < min)
		return min;
	else if( val > max)
		return max;
	return val;
}

Mat* adjustBrightness(const Mat& image)
{
	Mat* mat = new Mat();
	image.copyTo(*mat);
	cvtColor(*mat, *mat, CV_BGR2HSV);

	default_random_engine gen(time(0));
	uniform_real_distribution<double> distr(-MAX_ADJUST,MAX_ADJUST);

	double adjustment = distr(gen);
	for(int i = 0; i < mat->rows; i++)
	{
		for(int j = 0; j < mat->cols; j++)
		{
			Vec3b& pix = mat->at<Vec3b>(i,j);

			int newVal = pix[2] + adjustment * 255;
			pix[2] = clamp(newVal, 0, 255);
		}
	}
	cvtColor(*mat, *mat, CV_HSV2BGR);
	return mat;
}

template<typename type>
unsigned long writeImage(Mat& image, ofstream& outfile)
{
	//Mat image = imread(inPathandName,1); //color image
	//cout << image << endl;
	//printf("x %d, y %d, z %d, ir %d, ic %d id %d",xsize,ysize,zsize,image.rows,image.cols,image.depth());
	if(image.rows != ysize || image.cols != xsize)// || image.depth() != zsize)
		return 0;
	//cout << "after return" << endl;

	unsigned long origImageCount = imageCount;

	type pixel[3];
	long size = sizeof(type) * 3;
    

	//write image normal
	if(image.type() == CV_8UC3)
	{
		for(int i=0; i < xsize; i++)
		{
			for(int j=0; j < ysize; j++)
			{
				const Vec3b& curPixel = image.at<Vec3b>(i,j);
				pixel[0] = (type)curPixel[0];
				pixel[1] = (type)curPixel[1];
				pixel[2] = (type)curPixel[2];

				//cout << "writing" << endl;
				outfile.write(reinterpret_cast<const char *>(pixel),size);
			}
		}
		if(__output)
			outfile.write(reinterpret_cast<const char *>(&__globalTrueVal),sizeof(unsigned short));
		unalteredImageCount++;

		//horizontal reflection
		if(__horizontalReflect)
		{
			for(int i = xsize-1; i >= 0; i--)
			{
				for(int j = 0; j < ysize; j++)
				{
					const Vec3b& curPixel = image.at<Vec3b>(i,j);
					pixel[0] = (type)curPixel[0];
					pixel[1] = (type)curPixel[1];
					pixel[2] = (type)curPixel[2];
					outfile.write(reinterpret_cast<const char *>(pixel),size);
				}
			}
			if(__output)
				outfile.write(reinterpret_cast<const char *>(&__globalTrueVal),sizeof(short));
		}

		//vertical reflection
		if(__verticalReflect)
		{
			for(int i = 0; i < xsize; i++)
			{
				for(int j = ysize-1; j >= 0; j--)
				{
					const Vec3b& curPixel = image.at<Vec3b>(i,j);
					pixel[0] = (type)curPixel[0];
					pixel[1] = (type)curPixel[1];
					pixel[2] = (type)curPixel[2];
					outfile.write(reinterpret_cast<const char *>(pixel),size);
				}
			}
			if(__output)
				outfile.write(reinterpret_cast<const char *>(&__globalTrueVal),sizeof(short));
		}

		//90 deg clockwise rotation
		if(__rotateClock)
		{
			for(int j=0; j < ysize; j++)
			{
				for(int i = xsize-1; i >= 0; i--)
				{
					const Vec3b& curPixel = image.at<Vec3b>(i,j);
					pixel[0] = (type)curPixel[0];
					pixel[1] = (type)curPixel[1];
					pixel[2] = (type)curPixel[2];
					outfile.write(reinterpret_cast<const char *>(pixel),size);
				}
			}
			if(__output)
				outfile.write(reinterpret_cast<const char *>(&__globalTrueVal),sizeof(short));
		}

		//90 deg counterclockwise rotation
		if(__rotateCClock)
		{
			for(int j = ysize-1; j >= 0; j--)
			{
				for(int i = 0; i < xsize; i++)
				{
					const Vec3b& curPixel = image.at<Vec3b>(i,j);
					pixel[0] = (type)curPixel[0];
					pixel[1] = (type)curPixel[1];
					pixel[2] = (type)curPixel[2];
					outfile.write(reinterpret_cast<const char *>(pixel),size);
				}
			}
			if(__output)
				outfile.write(reinterpret_cast<const char *>(&__globalTrueVal),sizeof(short));
		}

		if(__rotate180)
		{
			for(int i = xsize-1; i >= 0; i--)
			{
				for(int j = ysize-1; j >= 0; j--)
				{
					const Vec3b& curPixel = image.at<Vec3b>(i,j);
					pixel[0] = (type)curPixel[0];
					pixel[1] = (type)curPixel[1];
					pixel[2] = (type)curPixel[2];
					outfile.write(reinterpret_cast<const char *>(pixel),size);
				}
			}
			if(__output)
				outfile.write(reinterpret_cast<const char *>(&__globalTrueVal),sizeof(short));
		}

		if(__brightnessAlterations)
		{
			double num = distr1(gen);
			if(num <= __brightnessAlterationChance)
			{
				Mat* mat = adjustBrightness(image);
				//uncomment if you want to see the difference it makes in the images
				// namedWindow("Display window", WINDOW_AUTOSIZE);
				// namedWindow("Display window2", WINDOW_AUTOSIZE);
				// imshow("Display window",image);
				// imshow("Display window2",*mat);
				// waitKey(1);
				// getchar();
				for(int i=0; i < xsize; i++)
				{
					for(int j=0; j < ysize; j++)
					{
						const Vec3b& curPixel = mat->at<Vec3b>(i,j);
						pixel[0] = (type)curPixel[0];
						pixel[1] = (type)curPixel[1];
						pixel[2] = (type)curPixel[2];

						//cout << "writing" << endl;
						outfile.write(reinterpret_cast<const char *>(pixel),size);
					}
				}
				if(__output)
					outfile.write(reinterpret_cast<const char *>(&__globalTrueVal),sizeof(unsigned short));

				delete(mat);

				imageCount++;
				counts.back().count++;
				byteCount += (xsize * ysize * 3 * sizeof(type) + sizeof(unsigned short));
				unordered_map<unsigned short, int>::const_iterator got = trueMap.find(__globalTrueVal);
				if(got == trueMap.end()) // not found
					trueMap[__globalTrueVal] = 1;
				else // found
					trueMap[__globalTrueVal]++;
			}
		}


		unordered_map<unsigned short, int>::const_iterator got = trueMap.find(__globalTrueVal);
		if(got == trueMap.end()) // not found
			trueMap[__globalTrueVal] = numWritesPerImage;
		else // found
			trueMap[__globalTrueVal]+=numWritesPerImage;
		byteCount  += (xsize * ysize * 3 * sizeof(type) + sizeof(unsigned short)) * numWritesPerImage; //extra for the ushort trueVal
		imageCount += numWritesPerImage;
		counts.back().count += numWritesPerImage;
		if(imageCount % 100000 < numWritesPerImage)
		{
			printf("Images: %ld, GB: %lf\n", imageCount, byteCount/1.0e9);
			if(byteCount/1.0e9 > 30)
			{
				outfile.close();
				exit(0);
			}
		}
	}
	else
	{
		cout << "Unsupported image type" << endl;
	}
	return imageCount - origImageCount;
}

template<typename type>
void breakUpImage(const char* imageName, ofstream& outfile)
{
	Mat image = imread(imageName,1);
	int numThisImage = 0;
	int numrows = image.rows;
	int numcols = image.cols;
	// printf("%s rows: %d, cols: %d.     ",imageName, numrows,numcols);
	if(numrows < ysize || numcols < xsize)
	{
		printf("The image %s is too small in at least one dimension. Minimum size is %dx%d.\n",imageName,xsize,ysize);
		return;
	}

	// cout << "Breaking with stride = " << stride << " and true val " << __globalTrueVal << endl;
	counts.resize(counts.size() + 1);
	counts.back().name = imageName;
	for(int i=0; i <= numrows-ysize; i+=stride)
	{
		for(int j=0; j<= numcols-xsize; j+=stride)
		{
			Mat out = image(Range(i,i+ysize),Range(j,j+xsize));
			numThisImage += writeImage<type>(out,outfile);
			//numThisImage+=numWritesPerImage; // numThisFullImage += numWritesPerSubImage;
			//imageNum++;
		}
	}
	// printf("%d images created.\n", numThisImage);	
}

int checkExtensions(const char* filename)
{
	const string name = filename;
	if(name.rfind(".jpg")  == name.length() - 4) return 1;
	if(name.rfind(".jpeg") == name.length() - 5) return 1;
	if(name.rfind(".png")  == name.length() - 4) return 1;
	return 0;
}

template<typename type>
void getImages(const char* folder, ofstream& outfile)
{
	cout << "Getting images from " << folder << endl;
	const char* inPath = folder;
	bool isDirectory;
	struct stat s;
	if(stat(inPath,&s) == 0)
	{
		if(s.st_mode & S_IFDIR) // directory
		{
			isDirectory = true;
		}
		else if (s.st_mode & S_IFREG) // file
		{
			isDirectory = false;
		}
		else
		{
			printf("We're not sure what the file you inputted was.\nExiting\n");
			return;
		}
	}
	else
	{
		cout << "Error getting status of folder or file. \"" << folder << "\"\nExiting\n";
		return;
	}
	
	if(isDirectory)
	{
		DIR *directory;
		struct dirent *file;
		if((directory = opendir(inPath)))// != NULL)
		{
			string pathName = inPath;
			if(pathName.rfind("/") != pathName.length()-1)
			{
				pathName.append(1,'/');
			}
			char inPathandName[250];
			while((file = readdir(directory)))// != NULL)
			{
				if(strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") != 0)
				{
					if(checkExtensions(file->d_name))
					{
						sprintf(inPathandName,"%s%s",pathName.c_str(),file->d_name);
						//cout << "going to write an image" << endl;
						//writeImage<type>(inPathandName,outfile);
						breakUpImage<type>(inPathandName,outfile);
					}
				}
			}
			closedir(directory);
		}
	}
	else
	{
		if(checkExtensions(inPath))
		{
			breakUpImage<type>(inPath,outfile);
			//writeImage<type>(inPath,outfile);
		}

	}
}

int main (int argc, char** argv)
{
	if(argc < 3)
	{
		cout << "Usage: (Required to come first):\n   ./TrainingImageSplitterFileCreator ImageConfigFile outfileName";
		cout << "\nOptional arguments (must come after required args. Case sensitive):\n";
		cout << "   stride=<int>             Stride for folders without strides specified in config. Defaults to 1.\n";
		cout << "   -rot_clock               For all images, adds a copy rotated 90 deg clockwise\n";
		cout << "   -rot_cclock              For all images, adds a copy rotated 90 deg counterclockwise\n";
		cout << "   -rot_180                 For all images, adds a copy rotated 180 deg\n";
		cout << "   -horizontal              For all images, adds a copy horizontally reflected\n";
		cout << "   -vertical                For all images, adds a copy vertically reflected\n";
		cout << "   -all                     Activates -rot_clock, -rot_cclock, -rot_180, -horizontal, -vertical -brightChance=0.5\n";
		cout << "   -no_output               Doesn't make output. This will tell you how big the file will be.\n";
		cout << "   -brightChance=<double>   Chance that an image will have a brightness altered version added to the binary.\n";
		cout << "                            Between 0 and 1. Defaults to 0.\n";
		return 0;
	}

	if(argc > 3)
	{
		for(int i = 3; i < argc; i++)
		{
			string arg(argv[i]);
			if(arg.find("stride=") != string::npos)
				globalStride = stoi(arg.substr(arg.find("=")+1));
			else if(arg.find("-rot_clock") != string::npos)
            {
				__rotateClock = true;
                numWritesPerImage++;
            }
			else if(arg.find("-rot_cclock") != string::npos)
            {
				__rotateCClock = true;
                numWritesPerImage++;
            }
			else if(arg.find("-rot_180") != string::npos)
            {
				__rotate180 = true;
                numWritesPerImage++;
            }
			else if(arg.find("-horizontal") != string::npos)
            {
				__horizontalReflect = true;
                numWritesPerImage++;
            }
			else if(arg.find("-vertical") != string::npos)
            {
				__verticalReflect = true;
                numWritesPerImage++;
            }
			else if(arg.find("-all") != string::npos)
			{
				__rotateClock = true;	__rotateCClock = true;	__rotate180 = true;
				__horizontalReflect = true;		__verticalReflect = true;
				__brightnessAlterationChance = .5;
				__brightnessAlterations = true;
                numWritesPerImage += 5;
			}
			else if(arg.find("-no_output") != string::npos)
			{
				__output = false;
			}
			else if(arg.find("-brightChance=") != string::npos)
			{
				__brightnessAlterationChance = stod(arg.substr(arg.find('=')+1));
				__brightnessAlterations = true;
			}
			else 
			{
				printf("Unknown arg %s. Aborting.\n", argv[i]);
			}
		}

	}

	ifstream imageConfig;
	ofstream outfile;
	imageConfig.open(argv[1]);
	if(__output)
	{
		outfile.open(argv[2], ios::trunc | ios::binary);
		if(!imageConfig.is_open())
		{
			cout << "Could not open the ImageConfigFile" << endl;
			return -1;
		}
	}
	string line;

	//get image sizes
	getline(imageConfig,line);

	int locx = line.find(" ");
	xsize = stoi(line.substr(0,locx));
	int locy = line.find(" ",locx+1);
	ysize = stoi(line.substr(locx+1, locy));
	int locz = line.find(" ",locy+1);
	zsize = stoi(line.substr(locy+1,locz));

	//get sizeByte
	getline(imageConfig,line);
	short sizeByte = stoi(line);

	cout << "SizeByte: " << sizeByte << " x: " << xsize << " y: " << ysize << " z: " << zsize << endl;

	/*
	double testd = 559.236;
	int testi = -400001;
	unsigned int testui = 5098;
	unsigned short testus = 4098;
	char testc = -120;
	unsigned char testuc = 230;
	float testf = -4.59;
	outfile.write(reinterpret_cast<const char *>(&testd),sizeof(double));
	outfile.write(reinterpret_cast<const char *>(&testi),sizeof(int));
	outfile.write(reinterpret_cast<const char *>(&testui),sizeof(unsigned int));
	outfile.write(reinterpret_cast<const char *>(&testus),sizeof(unsigned short));
	outfile.write(reinterpret_cast<const char *>(&testc),sizeof(char));
	outfile.write(reinterpret_cast<const char *>(&testuc),sizeof(unsigned char));
	outfile.write(reinterpret_cast<const char *>(&testf),sizeof(float));
	*/

	char slash0 = '\0';

	outfile.write(reinterpret_cast<const char *>(&sizeByte),sizeof(short));
	outfile.write(reinterpret_cast<const char *>(&xsize),sizeof(short));
	outfile.write(reinterpret_cast<const char *>(&ysize),sizeof(short));
	outfile.write(reinterpret_cast<const char *>(&zsize),sizeof(short));
	outfile.write(reinterpret_cast<const char *>(&slash0),sizeof(char));

	byteCount += 4 * sizeof(short) + sizeof(char);

	bool endtrueVals = false;

	vector<string> names;
	vector<int> trues;
	int numClasses = 0;

	while(getline(imageConfig, line))
	{
		if(line.size() == 0 || line[0] == '#' || line[0] == '\n')
			continue;
		if(line[0] == '$') // set a trueval
		{
			if(endtrueVals)
			{
				printf("All trueVals must come before the image folders.\n");
				return -1;
			}

			int space1 = line.find(' ');
			int space2 = line.find(' ', space1+1);
			int trueVal = stoi(line.substr(space1+1, space2 - space1 - 1));
			trues.push_back(trueVal);
			// string name = line.substr(space2 + 1);
			names.push_back(line.substr(space2 + 1));
			numClasses++;
		}
		else
		{
			if(!endtrueVals)
			{
				//write the number of classes
				outfile.write(reinterpret_cast<const char *>(&numClasses),sizeof(int));
				byteCount += sizeof(int);

				//write the trueval and name for each class
				for(int i = 0; i < numClasses; i++)
				{
					int trueVal = trues[i];
					char *out_name;
					out_name = new char[names[i].length()+1];
					
					for(int j= 0; j < names[i].length(); j++)
					{
						out_name[j] = names[i][j];
						// printf("%c\n", name[i]);
					}
					out_name[names[i].length()] = '\0';
					printf("Class %d, %s\n", trueVal, out_name);
					outfile.write(reinterpret_cast<const char *>(&trueVal),sizeof(int));
					outfile.write(out_name, sizeof(char) * (names[i].length() + 1)); // +1 for the '\0'

					byteCount += sizeof(int) + (names[i].length()+1) * sizeof(char);
					delete out_name;
				}
				endtrueVals = true;

				// getchar();
			}
			
			int comma1 = line.find(',');
			int comma2 = line.find(',',comma1+1);
			string folder = line.substr(0,comma1);
			unsigned short trueVal;
			if(comma2 != string::npos)
			 	trueVal = stoi(line.substr(comma1+1));
			else
				trueVal = stoi(line.substr(comma1+1,comma2));
			__globalTrueVal = trueVal;
			if(comma2 == string::npos)
				stride = globalStride;
			else
			{
				string stri = line.substr(comma2+1);
				if(stri.find("stride=") != string::npos)
				{
					stride = stoi(stri.substr(stri.find('=') + 1));
				}
				else
					stride = stoi(stri);
			}

			if(sizeByte == 1)
					getImages<unsigned char>(folder.c_str(),outfile);
			else if(sizeByte == -1)
					getImages<char>(folder.c_str(),outfile);
			else if(sizeByte == 2)
					getImages<unsigned short>(folder.c_str(),outfile);
			else if(sizeByte == -2)
					getImages<short>(folder.c_str(),outfile);
			else if(sizeByte == 4)
					getImages<unsigned int>(folder.c_str(),outfile);
			else if(sizeByte == -4)
					getImages<int>(folder.c_str(),outfile);
			else if(sizeByte == 5)
					getImages<float>(folder.c_str(),outfile);
			else if(sizeByte == 6)
					getImages<double>(folder.c_str(),outfile);
		}
	}

	imageConfig.close();
	outfile.close();

	

	printf("Total Images: %ld, GB: %lf\n", imageCount, byteCount/1.0e9);
	//cout << "Total: " << imageCount << " images created.";
    if(numWritesPerImage != 1)
        cout << " (" << unalteredImageCount << " without transformations)";
    cout << endl;

	double sum = 0;
	for( auto it = trueMap.begin(); it != trueMap.end(); it++)
	{
		sum += it->second;
	}
	cout << "Distribution:" << endl;
	int nameSize = getMaxNameSize(names);
	for( auto it = trueMap.begin(); it != trueMap.end(); it++)
	{
		cout << "True val " << it->first << ", ";
		cout << setw(nameSize) << left << getParallelName(names, trues, it->first) << ": ";
		cout << setw(6) << right << it->second << "   " << it->second/sum * 100 << "%\n";
	}

	printf("Type y to see the distribution by image. Type any other char to quit.\n");
	char cont = getchar();
	getchar();
	if(cont != 'y' && cont != 'Y')
		return 0;

	qsort(counts.data(), counts.size(), sizeof(imstruct), compareRev);
	for(int i = 0; i < counts.size(); i++)
	{
		printf("%s - %lu images. %lf%%\n", counts[i].name.c_str(), counts[i].count, 100.0 * counts[i].count/imageCount);
	}

	printf("Type y to see the distribution by folder. Type any other char to quit\n");
	cont = getchar();
	if(cont != 'y' && cont != 'Y')
		return 0;

	vector<vector<imstruct> > folderCounts(1);
	folderCounts.back() = counts;
	int f = 1;
	while(folderCounts.back().size() > 1)
	{
		folderCounts.resize(folderCounts.size() + 1);
		for(int i = 0; i < folderCounts[f-1].size(); i++)
		{
			bool found = false;
			string folder = folderCounts[f-1][i].name.substr(0, folderCounts[f-1][i].name.rfind('/'));
			for(int j = 0; j < folderCounts[f].size(); j++)
			{
				if(folderCounts[f][j].name == folder)
				{
					folderCounts[f][j].count += folderCounts[f-1][i].count;
					found = true;
					break;
				}
			}
			if(!found)
			{
				imstruct s;
				s.name = folder;
				s.count = folderCounts[f-1][i].count;
				folderCounts[f].push_back(s);
			}
		}

		qsort(folderCounts[f].data(), folderCounts[f].size(), sizeof(imstruct), compareRev);
		for(int j = 0; j < folderCounts[f].size(); j++)
		{
			printf("%s - %lu images. %lf%%\n", folderCounts[f][j].name.c_str(), folderCounts[f][j].count, 100.0 * folderCounts[f][j].count/imageCount);
		}
		printf("\n");

		f++;
	}

	return 0;
}







