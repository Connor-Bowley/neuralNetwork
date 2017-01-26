//CNN code
#include "ConvNetCL.h"

//general
#include <string>
#include <iostream>
#include <vector>
#include <fstream>

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//for looking at files and directories
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

//BOINC
#ifdef _BOINC_APP_
#include "diagnostics.h"
#include "filesys.h"
#include "boinc_api.h"
#include "mfile.h"
#include "proc_control.h"
#endif

using namespace cv;
using namespace std;

typedef vector<vector<vector<double> > > imVector;

const char* inPath;
char *outPath;
int stride = 1;

int cnnWidth, cnnHeight;

vector<string> filenames;
int curImage = 0;
int curRow = 0;
bool readimVector = false;


//VECTOR FUNCTIONS
void resize3DVector(vector<vector<vector<double> > > &vect, int width, int height, int depth)
{
	vect.resize(width);
	for(int i=0; i < width; i++)
	{
		vect[i].resize(height);
		for(int j=0; j < height; j++)
		{
			vect[i][j].resize(depth);
		}
	}
}

void setAll3DVector(vector<vector<vector<double> > > &vect, double val)
{
	for(int i=0; i< vect.size(); i++)
	{
		for(int j=0; j< vect[i].size(); j++)
		{
			for(int k=0; k< vect[i][j].size(); k++)
			{
				vect[i][j][k] = val;
			}
		}
	}
}

double vectorSum(const vector<double>& vect)
{
	double sum=0;
	for(int i=0; i<vect.size(); i++)
		sum += vect[i];
	return sum;
}
//END - VECTOR FUNCTIONS

//BOINC FUNCTIONS
std::string getBoincFilename(std::string filename) throw(std::runtime_error) {
    std::string resolved_path = filename;
	#ifdef _BOINC_APP_
	    if(boinc_resolve_filename_s(filename.c_str(), resolved_path)) {
	        printf("Could not resolve filename %s\n",filename.c_str());
	        throw std::runtime_error("Boinc could not resolve filename");
	    }
	#endif
    return resolved_path;
}

void writeCheckpoint(imVector& fullImage) throw(std::runtime_error)
{
	boinc_begin_critical_section();
	string resolved_checkpoint_name = getBoincFilename("checkpoint.yml");
	FileStorage outfile(resolved_checkpoint_name, FileStorage::WRITE);
	if(!outfile.isOpened())
		throw std::runtime_error("Checkpoint file could not be opened for writing.");

	printf("Writing Checkpoint: image: %d, last row completed: %d\n", curImage, curRow);

	outfile << "CUR_IMAGE" << curImage;
	outfile << "CUR_ROW" << curRow; // this is the compeleted row

	outfile << "FILENAMES" << "[:";
	for(int i = 0; i < filenames.size(); i++)
		outfile << filenames[i].c_str();
	outfile << "]";

	outfile << "IMVECTOR" << "[:";
	for(int i =0; i < fullImage.size(); i++)
		for(int j = 0; j < fullImage[i].size(); j++)
			for(int k = 0; k < fullImage[i][j].size(); k++)
				outfile << fullImage[i][j][k];
	outfile << "]";

	int numrows = fullImage.size();
	int numcols = fullImage[0].size();
	int depth = fullImage[0][0].size();
	// cout << "writing numrows" << endl;
	outfile << "NUMROWS" << numrows;
	outfile << "NUMCOLS" << numcols;
	outfile << "DEPTH" << depth;

	

	outfile.release();
	boinc_end_critical_section();

	// getchar();
}

bool readCheckpoint()
{
	string resolved_checkpoint_name = getBoincFilename("checkpoint.yml");
	FileStorage infile(resolved_checkpoint_name, FileStorage::READ);
	if(!infile.isOpened())
		return false;

	infile["CUR_IMAGE"] >> curImage;
	infile["CUR_ROW"]   >> curRow;
	curRow += stride; //to get us to the next row
	//infile["FILENAMES"] >> filenames;
	FileNode nameNode = infile["FILENAMES"];
	for(FileNodeIterator it = nameNode.begin(); it != nameNode.end(); it++)
	{
		string str;
		(*it) >> str;
		// printf("%s\n",str.c_str());
		filenames.push_back(str);

	}
 
 	infile.release();
	return true;
}

void readSavedimVector(imVector& fullImage)
{
	string resolved_checkpoint_name = getBoincFilename("checkpoint.yml");
	FileStorage infile(resolved_checkpoint_name, FileStorage::READ);
	if(!infile.isOpened())
	{
		printf("Error opening checkpoint file to read predicted image.\n");
		return;
	}

	int numrows, numcols, depth;
	infile["NUMROWS"] >> numrows;
	infile["NUMCOLS"] >> numcols;
	infile["DEPTH"]   >> depth;

	vector<double> flatVec;
	// infile["IMVECTOR"] >> flatVec;
	resize3DVector(fullImage,numrows,numcols,depth);

	// double* flat = flatVec.data();
	FileNode imVec = infile["IMVECTOR"];
	FileNodeIterator imit = imVec.begin();
	for(int i = 0; i < numrows; i++)
		for(int j = 0; j < numcols; j++)
			for(int k = 0; k < depth; k++)
			{
				// fullImage[i][j][k] = *(flat++);
				*imit >> fullImage[i][j][k];
				imit++;
			}

	infile.release();

}

double getFraction(double currow, double rowsInImage)
{
	double frac = 0;
	double fracFor1Image = 1.0/filenames.size();

	//fraction from completed images
	frac += curImage*fracFor1Image;

	//fraction from completion of current image
	frac += currow/rowsInImage * fracFor1Image;

	return frac;
}
//END - BOINC FUNCTIONS

//FUNCTIONS FOR READING FILES AND DIRECTORIES
int checkExtensions(const char* filename)
{
	string name = filename;
	if(name.rfind(".jpg")  == name.length() - 4) return 1;
	if(name.rfind(".jpeg") == name.length() - 5) return 1;
	if(name.rfind(".png")  == name.length() - 4) return 1;
	if(name.rfind(".JPG")  == name.length() - 4) return 1;
	if(name.rfind(".JPEG") == name.length() - 5) return 1;
	if(name.rfind(".PNG")  == name.length() - 4) return 1;
	return 0;
}

bool getFiles(const char* inPath)
{
	bool isDirectory;
	struct stat s;
	if(stat(inPath,&s) == 0)
	{
		if(s.st_mode & S_IFDIR) // directory
			isDirectory = true;
		else if (s.st_mode & S_IFREG) // file
			isDirectory = false;
		else
		{
			printf("We're not sure what the file you inputted was.\nExiting\n");
			return false;
		}
	}
	else
	{
		printf("Error getting status of folder or file to run over.\nExiting\n");
		return false;
	}
	
	if(isDirectory)
	{
		DIR *directory;
		struct dirent *file;
		if((directory = opendir(inPath)))// != NULL)
		{
			string pathName = inPath;
			if(pathName.rfind("/") != pathName.length()-1)
				pathName.append(1,'/');
			char inPathandName[250];
			while((file = readdir(directory)))// != NULL)
			{
				if(strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") != 0)
				{
					if(checkExtensions(file->d_name))
					{
						sprintf(inPathandName,"%s%s",pathName.c_str(),file->d_name);
						string ipan(inPathandName);
						if(ipan.find("_prediction") == string::npos)
							filenames.push_back(ipan);
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
			string ip(inPath);
			filenames.push_back(ip);
		}
	}
	return true;
}
// END - FUNCTIONS FOR READING FILES AND DIRECTORIES

//FUNCTIONS USED TO BREAK UP AN IMAGE
bool allElementsEquals(vector<double>& array)
{
	for(int i=1; i < array.size(); i++)
	{
		if(array[0] != array[i])
			return false;
	}
	return true;
}

void squareElements(vector<vector<vector<double> > >& vect)
{
	for(int i=0; i < vect.size(); i++)
		for(int j=0; j < vect[i].size(); j++)
			for(int k=0; k < vect[i][j].size(); k++)
				vect[i][j][k] = vect[i][j][k] * vect[i][j][k];
}

void breakUpImage(const char* imageName, Net& net)
{
	//cout << "starting breakUpImage" << endl;
	Mat image = imread(imageName,1);
	int numrows = image.rows;
	int numcols = image.cols;
	printf("%s rows: %d, cols: %d\n",imageName, numrows,numcols);


	vector<vector< vector<double> > > fullImage; //2 dims for width and height, last dim for each possible category
	int numClasses = net.getNumClasses();
	if(readimVector)
		readSavedimVector(fullImage);
	else
	{
		resize3DVector(fullImage,numrows,numcols,net.getNumClasses());
		setAll3DVector(fullImage,0);
	}

	vector<Mat> imageRow(0);
	vector<int> calcedClasses(0);
	vector<vector<double> > confidences(0);//for the confidence for each category for each image
		//the outer vector is the image, the inner vector is the category, the double is output(confidence) of the softmax

	int numrowsmcnn = numrows - cnnHeight;
	int numcolsmcnn = numcols - cnnWidth;
	if(numrows < cnnHeight || numcols < cnnWidth)
	{
		printf("The image %s is too small in at least one dimension. Minimum size is %dx%d.\n",imageName,cnnHeight,cnnWidth);
		return;
	}
	for( ; curRow <= numrowsmcnn; curRow += stride) //curRow will be 0 unless set by checkpoint
	{
		imageRow.resize(0);

		//get all subimages from a row
		for(int j=0; j<= numcolsmcnn; j+=stride) //NOTE: each j is a different subimage
		{
			const Mat out = image(Range(curRow,curRow+cnnHeight),Range(j,j+cnnWidth));
			imageRow.push_back(out);
		}

		//set them as the data in the net
		net.setData(imageRow);
		net.run();
		net.getConfidences(confidences); //gets the confidence for each category for each image

		int localCurImage = 0;
		for(int j=0; j<= numcolsmcnn; j+=stride) //NOTE: each iteration of this loop is a different subimage
		{
			for(int ii=curRow; ii < curRow+cnnHeight && ii < numrows; ii++)
				for(int jj=j; jj < j+cnnWidth && jj < numcols; jj++)
					for(int cat = 0; cat < confidences[localCurImage].size(); cat++)
						fullImage[ii][jj][cat] += confidences[localCurImage][cat];
			localCurImage++;
		}

		//update fraction done and see if we need to checkpoint
		double fraction = getFraction(curRow,numrowsmcnn);
		printf("Row %d of %d (%d) \t Image %d of %lu. \t %lf%% total completion.\n",curRow,numrowsmcnn,numrows,curImage+1,filenames.size(),fraction*100.0);
		#ifdef _BOINC_APP_
		boinc_fraction_done(fraction);
		if(boinc_time_to_checkpoint())
		{
			writeCheckpoint(fullImage);
			boinc_checkpoint_completed();
		}
		#endif
		// writeCheckpoint(fullImage);
	}
	curRow = 0; // this is set here so the next image will be right.

	// cout << "Went through whole image" << endl;

	//now we have the confidences for every pixel in the image
	//so get the category for each pixel and make a new image from it
	squareElements(fullImage);
	vector<Mat*> outputMats(numClasses);
	for(int m = 0; m < numClasses; m++)
		outputMats[m] = new Mat(numrows,numcols,CV_8UC3);
	for(int k = 0; k < numClasses; k++)
	{
		for(int i=0; i < numrows; i++)
		{
			for(int j=0; j < numcols; j++)
			{
				//square ratios
				double sumsq = vectorSum(fullImage[i][j]);
				for(int n=0; n < fullImage[i][j].size(); n++)
					fullImage[i][j][n] /= sumsq;

				//write the pixel
				Vec3b& outPix = outputMats[k]->at<Vec3b>(i,j);
				if(allElementsEquals(fullImage[i][j]))
				{
					outPix[0] = 0; outPix[1] = 255; outPix[2] = 0; // green
				}
				else
				{
					double white = 255 * fullImage[i][j][k];
					outPix[0] = white; // blue
					outPix[1] = white; // green
					outPix[2] = white; // red
				}
			}
		}
	}

	// cout << "Lets write the outputs" << endl;

	char outName[255];
	string origName(imageName);
	size_t dot = origName.rfind('.');
	const char *noExtension = origName.substr(0,dot).c_str();
	const char *extension = origName.substr(dot).c_str();

	for(int k = 0; k < numClasses; k++)
	{
		sprintf(outName,"%s_prediction_class%d%s",noExtension,k,extension);
		cout << "writing " << outName << endl;
		imwrite(outName, *(outputMats[k]));
		delete outputMats[k];
	}
}
//END - FUNCTIONS USED TO BREAK UP AN IMAGE

int main(int argc, char** argv)
{
	if(argc < 3  || 5 < argc)
	{
		printf("use format: ./ConvNetFullImageDriverCL_BOINC cnnConfig.txt imageOrFolderPath (stride=1) (device=0)\n");
		return -1;
	}

	printf("Initializing BOINC\n");

	#ifdef _BOINC_APP_
	boinc_init_diagnostics(BOINC_DIAG_MEMORYLEAKCHECKENABLED);

	BOINC_OPTIONS options;
	boinc_options_defaults(options);
	options.multi_thread = true;  // for multiple threads in OpenCL
	options.multi_process = true; // for multiple processes in OpenCL?
	options.normal_thread_priority = true; // so GPUs will run at full speed
	boinc_init_options(&options);
	boinc_init();
	#endif

	inPath = getBoincFilename(string(argv[2])).c_str();

	int device = 0;
	for(int i = 3; i < argc; i++)
	{
		string arg(argv[i]);
		if(arg.find("stride=") != string::npos)
			stride = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("device=") != string::npos)
			device = stoi(arg.substr(arg.find('=')+1));
	}

	string netPath = getBoincFilename(argv[1]);
	Net net(netPath.c_str());

	cnnWidth = net.getInputWidth();
	cnnHeight = net.getInputHeight();

	if(!net.setDevice(device) || !net.finalize())
	{
		printf("Net was unable to finalize on device %d\n", device);
		boinc_finish(-1);
		return -1;
	}

	bool noCheckpoint = true;

	#ifdef _BOINC_APP_
	if(readCheckpoint())
	{
		printf("Continuing from Checkpoint\n");
		noCheckpoint = false;
		readimVector = true;
	}
	else
	{
		printf("No Checkpoint found. Starting from beginning\n");
	}
	#endif

	if(noCheckpoint) // otherwise files will be put in vector by readCheckpoint
	{
		bool success = getFiles(inPath);
		if(!success)
		{
			printf("Error getting images to run over.\n");
			boinc_finish(-1);
			return -1;
		}
	}

	printf("CurImage: %d. filenamesSize %lu\n",curImage,filenames.size());
	for(; curImage < filenames.size(); curImage++)
	{
		breakUpImage(filenames[curImage].c_str(), net);
	}

	boinc_finish(0);
	return 0;

}





