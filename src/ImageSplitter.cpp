// ImageSplitter takes an image or all images in a folder and makes each possible 32x32 image from it/them.
// only takes jpeg, jpg, png
// use by ./ImageSplitter imageOrFolderName outputDirectory baseOutputName extension (stride)

#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef WIN32
# include <io.h>
#else
# include <unistd.h>
#endif

using namespace cv;
using namespace std;

string baseOutputName,extension;
char *inPath, *outPath;
int imageNum = 0;
int stride = 1;

void breakUpImage(char* imageName)
{
	Mat image = imread(imageName,1);
	int numThisImage = 0;
	int numrows = image.rows;
	int numcols = image.cols;
	printf("%s rows: %d, cols: %d\t",imageName, numrows,numcols);
	if(numrows < 32 || numcols < 32)
	{
		printf("The image %s is too small in at least one dimension. Minimum size is 32x32.\n",imageName);
		return;
	}
	for(int i=0; i <= numrows-32; i+=stride)
	{
		for(int j=0; j<= numcols-32; j+=stride)
		{
			//Mat out = image.create(i+96, j+32, CV_8UC3);
			Mat out = image(Range(i,i+32),Range(j,j+32));
			char outNumString[20];
			string outPathandName = outPath;
			if(outPathandName.rfind("/") != outPathandName.length()-1)
			{
				outPathandName.append(1,'/');
			}
			outPathandName.append(baseOutputName);
			sprintf(outNumString,"%d",imageNum++);
			outPathandName.append(outNumString);
			outPathandName.append(extension);
			imwrite(outPathandName,out);
			numThisImage++;
		}
	}
	printf("%d images created.\n", numThisImage);
	
}

int checkExtensions(char* filename)
{
	string name = filename;
	if(name.rfind(".jpg")  == name.length() - 4) return 1;
	if(name.rfind(".jpeg") == name.length() - 5) return 1;
	if(name.rfind(".png")  == name.length() - 4) return 1;
	return 0;
}

int main(int argc, char** argv)
{
	if(argc != 5 && argc != 6)
	{
		printf("use format: ./ImageSplitter imageOrFolderPath outputFolderPath baseOutputName extension (stride=1)\n");
		return -1;
	}
	baseOutputName = argv[3];
	extension = argv[4];
	inPath = argv[1];
	outPath = argv[2];
	if(argc == 6)
	{
		stride = atoi(argv[5]);
	}
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
			return -1;
		}
	}
	else
	{
		printf("Error getting status of folder or file.\nExiting\n");
		return -1;
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
						breakUpImage(inPathandName);
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
			breakUpImage(inPath);
		}

	}

	return 0;
	
}