#include "ConvNetSeam.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace cv;
using namespace std;

const char *kernelPath = "../kernels/Seamcarve_kernels.cl";

int owidth = -1, oheight = -1;
int owidth_g = -1, oheight_g = -1;

unsigned int window_width,        window_width_g;
int *image;                cl_mem image_g;
char *vdirs, *hdirs;       cl_mem vdirs_g, hdirs_g;
float *greyscale;          //cl_mem greyscale_g;
float *vals;               cl_mem vals_g;
float *vcosts, *hcosts;    cl_mem vcosts_g, hcosts_g;
int *vseam, *hseam;        cl_mem vseam_g, hseam_g;
cl_mem vmin_g, hmin_g;

bool __seam_inited = false, __seam_inited_g = false, __seam_opencl_inited = false;
bool singleCPU = false;
vector<cl_device_id> __deviceIds_seam;
int __q_seam;

cl_context __context_seam;
cl_program __seamcarve_program;
cl_uint __device_seam = 0;
cl_device_id __device_id_seam = NULL;
cl_platform_id __platform_seam = NULL;

//OpenCL kernels
cl_kernel vcolumnCalcCosts, vcalcSeamToRemove, vseamremove, vcolumnCalcCostsRev, vcalcSeamToRemoveRev;
cl_kernel hcolumnCalcCosts, hcalcSeamToRemove, hseamremove;


//OpenCL helper functions
void CheckError (cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::exit (1);
	}
}

std::string LoadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	//cout << result << endl;
	return result;
}

cl_program CreateProgram (std::string& source, cl_context& context, int programNum)
{
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}

static inline int POSITION3(int x, int y, int z) //assumes 3 channels
{
	return ((y * window_width * 3) + x*3) + z;
}

static inline int POSITION(int x, int y) {
    return ((y * window_width) + x);
}

void printArray(float* array, unsigned long size)
{
	for(unsigned long i = 0; i < size; i ++)
		printf("%f, ", array[i]);
	printf("\n");
}

void printArray(int* array, unsigned long size)
{
	for(unsigned long i = 0; i < size; i ++)
		printf("%d, ", array[i]);
	printf("\n");
}

void printArray(char* array, unsigned long size)
{
	for(unsigned long i = 0; i < size; i ++)
		printf("%d, ", array[i]);
	printf("\n");
}

void seamcarve_cleanup_gpu()
{
	if(__seam_inited_g)
	{
		clReleaseMemObject(image_g);
		clReleaseMemObject(vdirs_g);
		clReleaseMemObject(hdirs_g);
		// clReleaseMemObject(greyscale_g);
		clReleaseMemObject(vals_g);
		clReleaseMemObject(vcosts_g);
		clReleaseMemObject(hcosts_g);
		clReleaseMemObject(vseam_g);
		clReleaseMemObject(hseam_g);
		// clReleaseMemObject(vmin_g);
		// clReleaseMemObject(hmin_g);

		__seam_inited_g = false;
	}
}

void seamcarve_cleanup_cpu()
{
	// printf("Cleaning mem\n");
	if(__seam_inited)
	{
		delete image;
		delete vdirs;
		delete hdirs;
		delete greyscale;
		delete vals;
		delete vcosts;
		delete hcosts;
		delete vseam;
		delete hseam;

		__seam_inited = false;
	}
}

void seamcarve_cleanup_OpenCL()
{
	clReleaseKernel(vcolumnCalcCosts);
	clReleaseKernel(vcalcSeamToRemove);
	clReleaseKernel(vcolumnCalcCostsRev);
	clReleaseKernel(vcalcSeamToRemoveRev);
	clReleaseKernel(vseamremove);
	clReleaseKernel(hcolumnCalcCosts);
	clReleaseKernel(hcalcSeamToRemove);
	clReleaseKernel(hseamremove);

	clReleaseProgram(__seamcarve_program);

	clReleaseContext(__context_seam);

	__seam_opencl_inited = false;
}

void seamcarve_cleanup()
{
	seamcarve_cleanup_cpu();
	seamcarve_cleanup_gpu();
	seamcarve_cleanup_OpenCL();
}

void __seamcarve_init_OpenCL()
{
	cl_int error;
	cl_uint platformIdCount;
	cl_uint deviceIdCount;
	vector<cl_platform_id> platformIds;

	//platforms
	clGetPlatformIDs(0,nullptr, &platformIdCount);
	platformIds.resize(platformIdCount);
	clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);
	if(__platform_seam != NULL)
		platformIds[0] = __platform_seam;
	//devices
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);
	__deviceIds_seam.resize(deviceIdCount);
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, __deviceIds_seam.data(), nullptr);

	//context
	const cl_context_properties contextProperties[] = 
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties>(platformIds[0]),
		0,0
	};
	__context_seam = clCreateContext(contextProperties, deviceIdCount, __deviceIds_seam.data(),
		nullptr, nullptr, &error);
	CheckError(error);


	int q = 0;
	if(__device_seam != -1)
		q = __device_seam;
	else
	{
		__device_seam = 0;
		for(int i = 1; i < deviceIdCount; i++)
		{
			cl_device_type type;
			CheckError(clGetDeviceInfo(__deviceIds_seam[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, nullptr));
			if(type == CL_DEVICE_TYPE_GPU)
				q = i;
		}
		__device_seam = q;
	}

	if(__platform_seam != NULL)
	{
		__q_seam = 0;
		__deviceIds_seam[0] = __device_id_seam;
	}

	__q_seam = q;

	


	//create/compile program
	// #ifndef _BOINC_APP_
		string loadedKernel = LoadKernel(kernelPath);
	// #else
	// 	string loadedKernel = LoadKernel("Seamcarve_kernels.cl");
	// #endif
	__seamcarve_program = CreateProgram(loadedKernel, __context_seam, 0);
	const cl_device_id* deviceToBuild = &__deviceIds_seam[q];
	CheckError(clBuildProgram(__seamcarve_program, 1, deviceToBuild, nullptr, nullptr, nullptr));

	//create kernels
	vcolumnCalcCosts = clCreateKernel(__seamcarve_program, "vcolumnCalcCosts", &error); CheckError(error);
	vcalcSeamToRemove = clCreateKernel(__seamcarve_program, "vcalcSeamToRemove", &error); CheckError(error);
	vseamremove = clCreateKernel(__seamcarve_program, "vseamremove", &error); CheckError(error);

	vcolumnCalcCostsRev = clCreateKernel(__seamcarve_program, "vcolumnCalcCostsRev", &error); CheckError(error);
	vcalcSeamToRemoveRev = clCreateKernel(__seamcarve_program, "vcalcSeamToRemoveRev", &error); CheckError(error);

	hcolumnCalcCosts = clCreateKernel(__seamcarve_program, "hrowCalcCosts", &error); CheckError(error);
	hcalcSeamToRemove = clCreateKernel(__seamcarve_program, "hcalcSeamToRemove", &error); CheckError(error);
	hseamremove = clCreateKernel(__seamcarve_program, "hseamremove", &error); CheckError(error);

		__seam_opencl_inited = true;

}


void seamcarve_setDevice(int deviceNum)
{
	__device_seam = deviceNum;
	seamcarve_cleanup();
	__seam_opencl_inited = false;
}

void seamcarve_setDevice(cl_device_id dev, cl_platform_id plat)
{
	__device_id_seam = dev;
	__platform_seam = plat;
	seamcarve_cleanup();
	__seam_opencl_inited = false;
}

void __doMem(unsigned long window_size, unsigned int window_height)
{
	// printf("doing cpu mem\n");
	if(__seam_inited) // if something, delete it
	{	
		// printf("Cleaning mem\n");
		seamcarve_cleanup_cpu();
	}
	__seam_inited = true;

	owidth = window_width;
	oheight = window_height;

	image = new int[window_size * 3];
	vdirs = new char[window_size]; // 0 is right, 1 is down-right, -1 is up-right
	// printf("hdirs should init with size of %lu\n", window_size);
	hdirs = new char[window_size];
	greyscale = new float[window_size];
	vals = new float[window_size];
	vcosts = new float[window_size];
	hcosts = new float[window_size];
	vseam = new int[window_height];
	hseam = new int[window_width];
}

void __doMem_OpenCL(unsigned long window_size_g, unsigned int window_height_g)
{
	// printf("doing cl mem width %u, height %u, size %lu\n",window_width_g, window_height_g, window_size_g);
	if(!__seam_opencl_inited)
		__seamcarve_init_OpenCL();
	else if(__seam_inited_g)
		seamcarve_cleanup_gpu();

	window_width = window_width_g;
	__doMem(window_size_g, window_height_g);
	owidth = window_width_g;
	oheight = window_height_g;

	__seam_inited_g = true;

	owidth_g = window_width_g;
	oheight_g = window_height_g;

	cl_int error;

	image_g = clCreateBuffer(__context_seam, CL_MEM_READ_WRITE,
		sizeof(float) * window_size_g * 3, nullptr, &error); CheckError(error);
	vdirs_g = clCreateBuffer(__context_seam, CL_MEM_READ_WRITE,
		sizeof(char) * window_size_g, nullptr, &error); CheckError(error);
	hdirs_g = clCreateBuffer(__context_seam, CL_MEM_READ_WRITE,
		sizeof(char) * window_size_g, nullptr, &error); CheckError(error);
	// greyscale_g = clCreateBuffer(__context_seam, CL_MEM_READ_WRITE,
	// 	sizeof(float) * window_size_g, nullptr, &error); CheckError(error);
	vals_g = clCreateBuffer(__context_seam, CL_MEM_READ_WRITE,
		sizeof(float) * window_size_g, nullptr, &error); CheckError(error);
	vcosts_g = clCreateBuffer(__context_seam, CL_MEM_READ_WRITE,
		sizeof(float) * window_size_g, nullptr, &error); CheckError(error);
	hcosts_g = clCreateBuffer(__context_seam, CL_MEM_READ_WRITE,
		sizeof(float) * window_size_g, nullptr, &error); CheckError(error);
	vseam_g = clCreateBuffer(__context_seam, CL_MEM_READ_WRITE,
		sizeof(int) * window_height_g, nullptr, &error); CheckError(error);
	hseam_g = clCreateBuffer(__context_seam, CL_MEM_READ_WRITE,
		sizeof(int) * window_width_g, nullptr, &error); CheckError(error);
	vmin_g = clCreateBuffer(__context_seam, CL_MEM_READ_WRITE,
		sizeof(float), nullptr, &error); CheckError(error);
	hmin_g = clCreateBuffer(__context_seam, CL_MEM_READ_WRITE,
		sizeof(float), nullptr, &error); CheckError(error);
}


void calcGreyscale(const Mat& source, int window_height)
{
	//most of the stuff is global pointers so it'll be ok

	//get average of all color channels and use as value for each pix.
	//also put unaltered into the image ptr

	//x is the row (height), y is the col(width)
	for(int y = 0; y < window_width; y++)
		for(int x = 0; x < window_height; x++)
		{
			const Vec3b& pix = source.at<Vec3b>(x,y);
			int pos = POSITION(y,x);
			image[POSITION3(y,x,0)] = pix[0];
			image[POSITION3(y,x,1)] = pix[1];
			image[POSITION3(y,x,2)] = pix[2];

			greyscale[pos] = (pix[0] + pix[1] + pix[2])/3;
		}
	// printf("done greyscale\n");
}

void calcGradient(int window_height)
{
	//calc gradient (val) for each pixel
	//y is row (height), x is column (width)
	for(int y = 0; y < window_height; y++)
		for(int x = 0; x < window_width; x++)
		{
			float result = 0;
			int pos = POSITION(x,y);
			if(x > 0) 					result += fabs(greyscale[pos] - greyscale[POSITION(x-1, y)]);
			if(x  < window_width - 1)	result += fabs(greyscale[pos] - greyscale[POSITION(x+1, y)]);
            if(y > 0)                   result += fabs(greyscale[pos] - greyscale[POSITION(x, y-1)]);
            if(y < window_height - 1)   result += fabs(greyscale[pos] - greyscale[POSITION(x, y+1)]);

            vals[pos] = result;
		}
	// printf("done gradient\n");
}

//seamcarve both ways, vertical then horizontal
bool seamcarve_both_vth(int vseams, int hseams, const Mat& source, Mat& dest)
{
	Mat temp;
	bool good = seamcarve_vf(vseams,source,temp);
	if(!good) return false;
	return seamcarve_hf(hseams,temp,dest);
}

//seamcarve both ways, vertical then horizontal
bool seamcarve_both_vth_cpu(int vseams, int hseams, const Mat& source, Mat& dest)
{
	Mat temp;
	bool good = seamcarve_vf_cpu(vseams,source,temp);
	if(!good) return false;
	return seamcarve_hf_cpu(hseams,temp,dest);
}

//seamcarve both ways, horizontal then vertical
bool seamcarve_both_htv(int hseams, int vseams, const Mat& source, Mat& dest)
{
	Mat temp;
	bool good = seamcarve_hf(hseams,source,temp);
	if(!good) return false;
	return seamcarve_vf(vseams,temp,dest);
}

//seamcarve both ways, horizontal then vertical
bool seamcarve_both_htv_cpu(int hseams, int vseams, const Mat& source, Mat& dest)
{
	Mat temp;
	bool good = seamcarve_hf_cpu(hseams,source,temp);
	if(!good) return false;
	return seamcarve_vf_cpu(vseams,temp,dest);
}

bool seamcarve_hf(int numSeams, const Mat& source, Mat& dest)
{
	// printf("seamcarve_hf %d x %d\n",source.rows, source.cols);
	cl_int error;

	window_width_g = source.cols;
	window_width = source.cols;
	unsigned int window_height_g = source.rows;
	unsigned long window_size_g = source.rows * source.cols;
	if(owidth_g != window_width_g || oheight_g != window_height_g)
	{
		__doMem_OpenCL(window_size_g,window_height_g);
	}
	// printf("hf mem good\n");

	if(window_height_g < numSeams)
		return false;

	cl_command_queue queue = clCreateCommandQueue(__context_seam, __deviceIds_seam[__q_seam], 0, &error);
	CheckError(error);

	calcGreyscale(source, window_height_g);
	calcGradient(window_height_g);

	CheckError(clEnqueueWriteBuffer(queue, vals_g, CL_FALSE, 0,
		sizeof(float) * window_size_g, vals, 0, nullptr, nullptr));
	CheckError(clEnqueueWriteBuffer(queue, image_g, CL_FALSE, 0,
		sizeof(float) * window_size_g * 3, image, 0, nullptr, nullptr));
	clFinish(queue);

	//set some args that never change
	clSetKernelArg(hcolumnCalcCosts, 0, sizeof(cl_mem), &hcosts_g);
	clSetKernelArg(hcolumnCalcCosts, 1, sizeof(cl_mem), &hdirs_g);
	clSetKernelArg(hcolumnCalcCosts, 2, sizeof(cl_mem), &vals_g);
	clSetKernelArg(hcolumnCalcCosts, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(hcolumnCalcCosts, 4, sizeof(unsigned int), &window_height_g);

	clSetKernelArg(hcalcSeamToRemove, 0, sizeof(cl_mem), &hcosts_g);
	clSetKernelArg(hcalcSeamToRemove, 1, sizeof(cl_mem), &hdirs_g);
	clSetKernelArg(hcalcSeamToRemove, 2, sizeof(cl_mem), &hseam_g);
	clSetKernelArg(hcalcSeamToRemove, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(hcalcSeamToRemove, 4, sizeof(unsigned int), &window_height_g);
	clSetKernelArg(hcalcSeamToRemove, 6, sizeof(cl_mem), &hmin_g);

	clSetKernelArg(hseamremove, 0, sizeof(cl_mem), &image_g);
	clSetKernelArg(hseamremove, 1, sizeof(cl_mem), &vals_g);
	clSetKernelArg(hseamremove, 2, sizeof(cl_mem), &hseam_g);
	clSetKernelArg(hseamremove, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(hseamremove, 4, sizeof(unsigned int), &window_height_g);


	size_t globalWorkSize[] = {0,0,0};
	unsigned int count = 0;
	clFinish(queue); //make sure stuff has copied by now
	while(count < numSeams)
	{
		// printf("In hf %d\n", count);
		//calculate columns costs and dirs
		clSetKernelArg(hcolumnCalcCosts, 5, sizeof(unsigned int), &count);
		globalWorkSize[0] = window_height_g - count;
		CheckError(clEnqueueNDRangeKernel(queue, hcolumnCalcCosts, 1, 
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

		// //calculate seam to remove
		clSetKernelArg(hcalcSeamToRemove, 5, sizeof(unsigned int), &count);
		globalWorkSize[0] = 1;
		CheckError(clEnqueueNDRangeKernel(queue, hcalcSeamToRemove, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

		// //remove one seam
		clSetKernelArg(hseamremove, 5, sizeof(unsigned int), &count);
		globalWorkSize[0] = window_width;
		CheckError(clEnqueueNDRangeKernel(queue, hseamremove, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

		count++;
	}
	// printf("done while\n");

	//pull from cl_mem object
	CheckError(clEnqueueReadBuffer(queue, image_g, CL_TRUE, 0,
		sizeof(float) * window_size_g * 3, image, 0, nullptr, nullptr));

	//put in dest mat
	// printf("creating dest\n");
	dest.create(source.rows-count,source.cols, CV_8UC3);
	// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
	//i is row, j is col
	for(int i = 0; i < dest.rows; i++)
	{
		for(int j = 0; j < dest.cols; j++)
		{
			// printf("j %d, i %d\n", j,i);
			Vec3b& pix = dest.at<Vec3b>(i,j);
			pix[0] = image[POSITION3(j,i,0)];
			pix[1] = image[POSITION3(j,i,1)];
			pix[2] = image[POSITION3(j,i,2)];
		}
	}
	clReleaseCommandQueue(queue);
	return true;
}

bool seamcarve_hf_cpu(int numSeams, const Mat& source, Mat& dest)
{
	window_width = source.cols;
	int window_height = source.rows;
	unsigned long window_size = source.rows * source.cols;

	//only redo the memory stuff if it changes
	if(owidth != window_width || oheight != window_height)
	{
		__doMem(window_size,window_height);
	}

	if(window_height < numSeams)
		return false;

	calcGreyscale(source, window_height);//only need height bc width is global
	calcGradient(window_height);

	//x is col (width), y is row (height)

	int count = 0;
	// printf("Starting loop\n");
	while(count < numSeams)
	{
		//show current image
		// dest.create(source.rows-count,source.cols, CV_8UC3);
		// printf("rows: %d, cols: %d\n", source.rows-count,source.cols);
		// //i is row, j is col
		// for(int i = 0; i < dest.rows; i++)
		// {
		// 	for(int j = 0; j < dest.cols; j++)
		// 	{
		// 		Vec3b& pix = dest.at<Vec3b>(i,j);
		// 		pix[0] = image[POSITION(j,i,0)];
		// 		pix[1] = image[POSITION(j,i,1)];
		// 		pix[2] = image[POSITION(j,i,2)];
		// 	}
		// }

		// imshow("test",dest);
		// waitKey(1);
		// printf("Press enter for next frame\n");
		// getchar();

		//init right col
		for(int y = 0; y < window_height - count; y++)
		{
			int pos = POSITION(window_width - 1, y);
			hcosts[pos] = vals[pos];
		}

		//calc rest of costs and dirs
		for(int x = window_width - 2; x >= 0; x--)
		{

			//do top pixel
			if(hcosts[POSITION(x+1,0)] < hcosts[POSITION(x+1,1)])
			{
				hcosts[POSITION(x,0)] = vals[POSITION(x,0)] + hcosts[POSITION(x+1,0)];
				hdirs[POSITION(x,0)] = 0; // right
			}
			else
			{
				hcosts[POSITION(x,0)] = vals[POSITION(x,0)] + hcosts[POSITION(x+1,1)];
				hdirs[POSITION(x,0)] = 1; //down
			}

			//middle
			int y;
			for(y = 1; y < window_height - count - 1; y++)
			{
				float cost_up   = hcosts[POSITION(x+1, y-1)];
				float cost_left = hcosts[POSITION(x+1, y  )]; //really goes to the right
				float cost_down = hcosts[POSITION(x+1, y+1)];
				int mypos = POSITION(x,y);

				if(cost_up < cost_left && cost_up < cost_down) // cost_up is min
				{
					hcosts[mypos] = vals[mypos] + cost_up;
					hdirs[mypos] = -1;
				}
				else if(cost_down < cost_left) // down is min
				{
					hcosts[mypos] = vals[mypos] + cost_down;
					hdirs[mypos] = 1;
				}
				else //straight left/right is min
				{
					hcosts[mypos] = vals[mypos] + cost_left;
					hdirs[mypos] = 0;
				}
			}
			// printf("Bottom\n");
			//bottom pixel
			int pos = POSITION(x,y);
			if(hcosts[POSITION(x+1, y)] < hcosts[POSITION(x+1,y-1)])
			{
				hcosts[pos] = vals[pos] + hcosts[POSITION(x+1,y)];
				hdirs[pos] = 0;
			}
			else
			{
				hcosts[pos] = vals[pos] + hcosts[POSITION(x+1,y-1)];
				hdirs [pos] = -1;
			}
		}//end calc costs and dirs

		// printf("Calc seam\n");

		//calc seams to remove
		float min_val = 20000000;
		for(int y = 0; y < window_height - count; y++)
		{
			if(hcosts[POSITION(0,y)] < min_val)
			{
				min_val = hcosts[POSITION(0,y)];
				hseam[0] = y;
			}
		}
		for(int x = 1; x < window_width; x++)
			hseam[x] = hseam[x-1] + hdirs[POSITION(x-1,hseam[x-1])];

		//remove one horizontal seam
		for(int x = 0; x < window_width; x++)
		{
			int y;
			for(y = hseam[x]; y < window_height - count - 1; y++)
			{
				image[POSITION3(x,y,0)] = image[POSITION3(x,y+1,0)];
				image[POSITION3(x,y,1)] = image[POSITION3(x,y+1,1)];
				image[POSITION3(x,y,2)] = image[POSITION3(x,y+1,2)];

				vals[POSITION(x,y)] = vals[POSITION(x,y+1)];
			}

			image[POSITION3(x,y,0)] = 0;
			image[POSITION3(x,y,1)] = 0;
			image[POSITION3(x,y,2)] = 0;

			vals[POSITION(x,y)] = 0;
		}
		count++;
	}


	//put new image in dest mat
	dest.create(source.rows-count,source.cols, CV_8UC3);
	// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
	//i is row, j is col
	for(int i = 0; i < dest.rows; i++)
	{
		for(int j = 0; j < dest.cols; j++)
		{
			Vec3b& pix = dest.at<Vec3b>(i,j);
			pix[0] = image[POSITION3(j,i,0)];
			pix[1] = image[POSITION3(j,i,1)];
			pix[2] = image[POSITION3(j,i,2)];
		}
	}

	return true;
}

bool seamcarve_vf(int numSeams, const Mat& source, Mat& dest)
{
	// printf("seamcarve_vf %d x %d\n",source.rows, source.cols);
	cl_int error;

	window_width_g = source.cols;
	window_width = source.cols;
	unsigned int window_height_g = source.rows;
	unsigned long window_size_g = source.rows * source.cols;
	if(owidth_g != window_width_g || oheight_g != window_height_g)
	{
		__doMem_OpenCL(window_size_g,window_height_g);
	}

	if(window_width_g < numSeams)
		return false;

	cl_command_queue queue = clCreateCommandQueue(__context_seam, __deviceIds_seam[__q_seam], 0, &error);
	CheckError(error);

	calcGreyscale(source, window_height_g);
	calcGradient(window_height_g);

	CheckError(clEnqueueWriteBuffer(queue, vals_g, CL_FALSE, 0,
		sizeof(float) * window_size_g, vals, 0, nullptr, nullptr));
	CheckError(clEnqueueWriteBuffer(queue, image_g, CL_FALSE, 0,
		sizeof(float) * window_size_g * 3, image, 0, nullptr, nullptr));
	clFinish(queue);

	//set some args that never change
	clSetKernelArg(vcolumnCalcCosts, 0, sizeof(cl_mem), &vcosts_g);
	clSetKernelArg(vcolumnCalcCosts, 1, sizeof(cl_mem), &vdirs_g);
	clSetKernelArg(vcolumnCalcCosts, 2, sizeof(cl_mem), &vals_g);
	clSetKernelArg(vcolumnCalcCosts, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(vcolumnCalcCosts, 4, sizeof(unsigned int), &window_height_g);

	clSetKernelArg(vcalcSeamToRemove, 0, sizeof(cl_mem), &vcosts_g);
	clSetKernelArg(vcalcSeamToRemove, 1, sizeof(cl_mem), &vdirs_g);
	clSetKernelArg(vcalcSeamToRemove, 2, sizeof(cl_mem), &vseam_g);
	clSetKernelArg(vcalcSeamToRemove, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(vcalcSeamToRemove, 4, sizeof(unsigned int), &window_height_g);
	clSetKernelArg(vcalcSeamToRemove, 6, sizeof(cl_mem), &vmin_g);

	clSetKernelArg(vseamremove, 0, sizeof(cl_mem), &image_g);
	clSetKernelArg(vseamremove, 1, sizeof(cl_mem), &vals_g);
	clSetKernelArg(vseamremove, 2, sizeof(cl_mem), &vseam_g);
	clSetKernelArg(vseamremove, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(vseamremove, 4, sizeof(unsigned int), &window_height_g);


	size_t globalWorkSize[] = {0,0,0};
	unsigned int count = 0;
	clFinish(queue); //make sure stuff has copied by now
	while(count < numSeams)
	{
		//calculate columns costs and dirs
		clSetKernelArg(vcolumnCalcCosts, 5, sizeof(unsigned int), &count);
		globalWorkSize[0] = window_width_g - count;
		CheckError(clEnqueueNDRangeKernel(queue, vcolumnCalcCosts, 1, 
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

		//calculate seam to remove
		clSetKernelArg(vcalcSeamToRemove, 5, sizeof(unsigned int), &count);
		globalWorkSize[0] = 1;
		CheckError(clEnqueueNDRangeKernel(queue, vcalcSeamToRemove, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

		//remove one seam
		clSetKernelArg(vseamremove, 5, sizeof(unsigned int), &count);
		globalWorkSize[0] = window_height_g;
		CheckError(clEnqueueNDRangeKernel(queue, vseamremove, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

		count++;
	}

	//pull from cl_mem object
	CheckError(clEnqueueReadBuffer(queue, image_g, CL_TRUE, 0,
		sizeof(float) * window_size_g * 3, image, 0, nullptr, nullptr));

	//put in dest mat
	dest.create(source.rows,source.cols-count, CV_8UC3);
	// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
	//i is row, j is col
	for(int i = 0; i < dest.rows; i++)
	{
		for(int j = 0; j < dest.cols; j++)
		{
			Vec3b& pix = dest.at<Vec3b>(i,j);
			pix[0] = image[POSITION3(j,i,0)];
			pix[1] = image[POSITION3(j,i,1)];
			pix[2] = image[POSITION3(j,i,2)];
		}
	}
	clReleaseCommandQueue(queue);
	return true;
}

/*bool seamcarve_vfRandom(int numSeams, const Mat& source, Mat& dest)
{
	cl_int error;

	window_width_g = source.cols;
	window_width = source.cols;
	unsigned int window_height_g = source.rows;
	unsigned long window_size_g = source.rows * source.cols;
	if(owidth_g != window_width_g || oheight_g != window_height_g)
	{
		__doMem_OpenCL(window_size_g,window_height_g);
	}

	if(window_width_g < numSeams)
		return false;

	cl_command_queue queue = clCreateCommandQueue(__context_seam, __deviceIds_seam[__q_seam], 0, &error);
	CheckError(error);

	calcGreyscale(source, window_height_g); // this puts the Mat into the image array

	CheckError(clEnqueueWriteBuffer(queue, vals_g, CL_FALSE, 0,
		sizeof(float) * window_size_g, vals, 0, nullptr, nullptr));
	CheckError(clEnqueueWriteBuffer(queue, image_g, CL_FALSE, 0,
		sizeof(float) * window_size_g * 3, image, 0, nullptr, nullptr));

	clSetKernelArg(vseamremove, 0, sizeof(cl_mem), &image_g);
	clSetKernelArg(vseamremove, 1, sizeof(cl_mem), &vals_g);
	clSetKernelArg(vseamremove, 2, sizeof(cl_mem), &vseam_g);
	clSetKernelArg(vseamremove, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(vseamremove, 4, sizeof(unsigned int), &window_height_g);

	clFinish(queue);

	
	unsigned int count = 0;
	size_t globalWorkSize[] = {0,0,0};
	default_random_engine gen(time(NULL));
	uniform_int_distribution<int> disR(-100000,100000);
	// time_t lastTime = time(NULL);
	while(count < numSeams)
	{
		if(count % 20 == 0)
		{
			default_random_engine gen2(time(NULL) + disR(gen));	
			gen = gen2;
			printf("new gen\n");
		}
		
		uniform_int_distribution<int> distr(0,window_width_g - count);
		vseam[0] = distr(gen);
		printf("%d\n", vseam[0]);
		for(int i = 1; i < window_height_g; i++)
		{
			// int xmin = vseam[i-1] == 0 ? 0 : -1;
			// int xmax = vseam[i-1] == window_width_g - count - 1 ? 0 : 1;
			// uniform_int_distribution<int> dis2(xmin, xmax);
			vseam[i] = vseam[i-1];// + dis2(gen);

			if(vseam[i] != vseam[0])
				printf("%d %d\n",i, vseam[i]);
		}
		CheckError(clEnqueueWriteBuffer(queue, vseam_g, CL_TRUE, 0,
			sizeof(int) * window_height_g, vseam, 0, nullptr, nullptr));

		//remove one seam
		clSetKernelArg(vseamremove, 5, sizeof(unsigned int), &count);
		globalWorkSize[0] = window_height_g;
		CheckError(clEnqueueNDRangeKernel(queue, vseamremove, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

		count++;
	}

	//pull from cl_mem object
	CheckError(clEnqueueReadBuffer(queue, image_g, CL_TRUE, 0,
		sizeof(float) * window_size_g * 3, image, 0, nullptr, nullptr));

	//put in dest mat
	dest.create(source.rows,source.cols-count, CV_8UC3);
	// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
	//i is row, j is col
	for(int i = 0; i < dest.rows; i++)
	{
		for(int j = 0; j < dest.cols; j++)
		{
			Vec3b& pix = dest.at<Vec3b>(i,j);
			pix[0] = image[POSITION3(j,i,0)];
			pix[1] = image[POSITION3(j,i,1)];
			pix[2] = image[POSITION3(j,i,2)];
		}
	}
	clReleaseCommandQueue(queue);
	return true;
}*/

bool seamcarve_vfRev(int numSeams, const Mat& source, Mat& dest)
{
	// printf("seamcarve_vf %d x %d\n",source.rows, source.cols);
	cl_int error;

	window_width_g = source.cols;
	window_width = source.cols;
	unsigned int window_height_g = source.rows;
	unsigned long window_size_g = source.rows * source.cols;
	if(owidth_g != window_width_g || oheight_g != window_height_g)
	{
		__doMem_OpenCL(window_size_g,window_height_g);
	}

	if(window_width_g < numSeams)
		return false;

	cl_command_queue queue = clCreateCommandQueue(__context_seam, __deviceIds_seam[__q_seam], 0, &error);
	CheckError(error);

	calcGreyscale(source, window_height_g);
	calcGradient(window_height_g);

	CheckError(clEnqueueWriteBuffer(queue, vals_g, CL_FALSE, 0,
		sizeof(float) * window_size_g, vals, 0, nullptr, nullptr));
	CheckError(clEnqueueWriteBuffer(queue, image_g, CL_FALSE, 0,
		sizeof(float) * window_size_g * 3, image, 0, nullptr, nullptr));
	clFinish(queue);

	//set some args that never change
	clSetKernelArg(vcolumnCalcCostsRev, 0, sizeof(cl_mem), &vcosts_g);
	clSetKernelArg(vcolumnCalcCostsRev, 1, sizeof(cl_mem), &vdirs_g);
	clSetKernelArg(vcolumnCalcCostsRev, 2, sizeof(cl_mem), &vals_g);
	clSetKernelArg(vcolumnCalcCostsRev, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(vcolumnCalcCostsRev, 4, sizeof(unsigned int), &window_height_g);

	clSetKernelArg(vcalcSeamToRemoveRev, 0, sizeof(cl_mem), &vcosts_g);
	clSetKernelArg(vcalcSeamToRemoveRev, 1, sizeof(cl_mem), &vdirs_g);
	clSetKernelArg(vcalcSeamToRemoveRev, 2, sizeof(cl_mem), &vseam_g);
	clSetKernelArg(vcalcSeamToRemoveRev, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(vcalcSeamToRemoveRev, 4, sizeof(unsigned int), &window_height_g);
	clSetKernelArg(vcalcSeamToRemoveRev, 6, sizeof(cl_mem), &vmin_g);

	clSetKernelArg(vseamremove, 0, sizeof(cl_mem), &image_g);
	clSetKernelArg(vseamremove, 1, sizeof(cl_mem), &vals_g);
	clSetKernelArg(vseamremove, 2, sizeof(cl_mem), &vseam_g);
	clSetKernelArg(vseamremove, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(vseamremove, 4, sizeof(unsigned int), &window_height_g);


	size_t globalWorkSize[] = {0,0,0};
	unsigned int count = 0;
	clFinish(queue); //make sure stuff has copied by now
	while(count < numSeams)
	{
		//calculate columns costs and dirs
		clSetKernelArg(vcolumnCalcCostsRev, 5, sizeof(unsigned int), &count);
		globalWorkSize[0] = window_width_g - count;
		CheckError(clEnqueueNDRangeKernel(queue, vcolumnCalcCostsRev, 1, 
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

		//calculate seam to remove
		clSetKernelArg(vcalcSeamToRemoveRev, 5, sizeof(unsigned int), &count);
		globalWorkSize[0] = 1;
		CheckError(clEnqueueNDRangeKernel(queue, vcalcSeamToRemoveRev, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

		//remove one seam
		clSetKernelArg(vseamremove, 5, sizeof(unsigned int), &count);
		globalWorkSize[0] = window_height_g;
		CheckError(clEnqueueNDRangeKernel(queue, vseamremove, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
		clFinish(queue);

		count++;
	}

	//pull from cl_mem object
	CheckError(clEnqueueReadBuffer(queue, image_g, CL_TRUE, 0,
		sizeof(float) * window_size_g * 3, image, 0, nullptr, nullptr));

	//put in dest mat
	dest.create(source.rows,source.cols-count, CV_8UC3);
	// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
	//i is row, j is col
	for(int i = 0; i < dest.rows; i++)
	{
		for(int j = 0; j < dest.cols; j++)
		{
			Vec3b& pix = dest.at<Vec3b>(i,j);
			pix[0] = image[POSITION3(j,i,0)];
			pix[1] = image[POSITION3(j,i,1)];
			pix[2] = image[POSITION3(j,i,2)];
		}
	}
	clReleaseCommandQueue(queue);
	return true;
}

bool seamcarve_vf_cpu(int numSeams, const Mat& source, Mat& dest)
{
	window_width = source.cols;
	unsigned int window_height = source.rows;
	unsigned long window_size = source.rows * source.cols;

	//only redo the memory stuff if it changes
	if(owidth != window_width || oheight != window_height)
	{
		__doMem(window_size,window_height);
	}

	if(window_width < numSeams)
		return false;

	calcGreyscale(source, window_height);
	calcGradient(window_height);

	// namedWindow("test",WINDOW_AUTOSIZE);

	//printf("Starting loop\n");

	unsigned int count = 0;
	while(count < numSeams)
	{
		//show current image
		// dest.create(source.rows,source.cols-count, CV_8UC3);
		// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
		// //i is row, j is col
		// for(int i = 0; i < dest.rows; i++)
		// {
		// 	for(int j = 0; j < dest.cols; j++)
		// 	{
		// 		Vec3b& pix = dest.at<Vec3b>(i,j);
		// 		pix[0] = image[POSITION(j,i,0)];
		// 		pix[1] = image[POSITION(j,i,1)];
		// 		pix[2] = image[POSITION(j,i,2)];
		// 	}
		// }

		// imshow("test",dest);
		// waitKey(1);
		// printf("Press enter for next frame\n");
		// getchar();

		//x is the col, y is the row

		//init bottom row
		// printf("Init bottom row. %d cols\n", (window_width - count));
		for(int x = 0; x < (window_width - count); x++)
		{
			int pos = POSITION(x,window_height - 1);
			//printf("%d\n", pos);
			vcosts[pos] = vals[pos];
			//dirs[pos] = 0; // doesn't really matter
		}

		//calc rest of costs and dirs
		for(int y = window_height - 2; y >= 0; y--)
		{
			//do left side
			if(vcosts[POSITION(0, y+1)] < vcosts[POSITION(1,y+1)])
			{
				vcosts[POSITION(0, y)] = vals[POSITION(0,y)] + vcosts[POSITION(0, y+1)];
				vdirs[POSITION(0,y)] = 0; //up
			}
			else
			{
				vcosts[POSITION(0, y)] = vals[POSITION(0,y)] + vcosts[POSITION(1, y+1)];
				vdirs[POSITION(0,y)] = 1;
			}

			//middle
			int x;
			for(x = 1; x < window_width - count - 1; x++)
			{
				float cost_left  = vcosts[POSITION(x-1, y+1)];
				float cost_up    = vcosts[POSITION(x  , y+1)];
				float cost_right = vcosts[POSITION(x+1, y+1)];
				int mypos = POSITION(x,y);
				// if(mypos == 304)
				// 	printf("Got 304\n");

				if(cost_left < cost_up  && cost_left < cost_right) // cost_left is min
				{
					vcosts[mypos] = vals[mypos] + cost_left;
					vdirs[mypos] = -1;
				}
				else if(cost_right < cost_up && cost_right < cost_left) //cost_right is min
				{
					vcosts[mypos] = vals[mypos] + cost_right;
					vdirs[mypos] = 1;
				}
				else // up is min
				{
					vcosts[mypos] = vals[mypos] + cost_up;
					vdirs[mypos] = 0;
				}
			}

			//right side
			int pos = POSITION(x, y);
			if (vcosts[POSITION(x, y+1)] < vcosts[POSITION(x-1, y+1)]) {
                vcosts[pos] = vals[pos] + vcosts[POSITION(x, y+1)];
                vdirs [pos] = 0;
            } else {
                vcosts[pos] = vals[pos] + vcosts[POSITION(x-1, y+1)];
                vdirs [pos] = -1;
            }		
        }//end calc costs and dirs

        //calc seam to remove
        float min_val = 20000000;
        for(int x = 0; x < window_width - count; x++)
        {
        	//printf("vcosts[%d] = %f\n", x,vcosts[POSITION(x,0)]);
        	if(vcosts[POSITION(x,0)] < min_val)
        	{
        		min_val = vcosts[POSITION(x,0)];
        		vseam[0] = x;
        	}
        }
        for(int y = 1; y < window_height; y++)
        	vseam[y] = vseam[y-1] + vdirs[POSITION(vseam[y-1], y-1)];

        //remove one seam
        for(int y = 0; y < window_height; y++)
        {
        	int x;
        	// printf("x %d, y %d\n", vseam[y], y);
        	for(x = vseam[y]; x < window_width - count - 1; x++)
        	{
        		image[POSITION3(x,y,0)] = image[POSITION3(x+1,y,0)];
        		image[POSITION3(x,y,1)] = image[POSITION3(x+1,y,1)];
        		image[POSITION3(x,y,2)] = image[POSITION3(x+1,y,2)];

        		vals[POSITION(x,y)] = vals[POSITION(x+1,y)];
        	}

        	image[POSITION3(x,y,0)] = 0;
        	image[POSITION3(x,y,1)] = 0;
        	image[POSITION3(x,y,2)] = 0;

        	vals[POSITION(x,y)] = 0;
        }
        count++;
	}

	dest.create(source.rows,source.cols-count, CV_8UC3);
	// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
	//i is row, j is col
	for(int i = 0; i < dest.rows; i++)
	{
		for(int j = 0; j < dest.cols; j++)
		{
			Vec3b& pix = dest.at<Vec3b>(i,j);
			pix[0] = image[POSITION3(j,i,0)];
			pix[1] = image[POSITION3(j,i,1)];
			pix[2] = image[POSITION3(j,i,2)];
		}
	}

	//put new image into mat
	// dest.create(source.rows,source.cols-numSeams, CV_8UC3);
	// for(int i = 0; i < dest.rows; i++)
	// {
	// 	for(int j = 0; j < dest.cols; j++)
	// 	{
	// 		Vec3b& pix = dest.at<Vec3b>(j,i);
	// 		pix[0] = image[POSITION(i,j,0)];
	// 		pix[1] = image[POSITION(i,j,1)];
	// 		pix[2] = image[POSITION(i,j,2)];
	// 	}
	// }

	return true;
}

bool __seamcarve_both_RorS(int vseams, int hseams, const Mat& source, Mat& dest, bool isScaled)
{
	cl_int error;

	window_width_g = source.cols;
	window_width = source.cols;
	unsigned int window_height_g = source.rows;
	unsigned long window_size_g = source.rows * source.cols;
	if(owidth_g != window_width_g || oheight_g != window_height_g)
	{
		__doMem_OpenCL(window_size_g,window_height_g);
	}

	if(window_width_g < vseams || window_height_g < hseams)
		return false;

	cl_command_queue queue = clCreateCommandQueue(__context_seam, __deviceIds_seam[__q_seam], 0, &error);
	CheckError(error);

	calcGreyscale(source, window_height_g);
	calcGradient(window_height_g);

	CheckError(clEnqueueWriteBuffer(queue, vals_g, CL_FALSE, 0,
		sizeof(float) * window_size_g, vals, 0, nullptr, nullptr));
	CheckError(clEnqueueWriteBuffer(queue, image_g, CL_FALSE, 0,
		sizeof(float) * window_size_g * 3, image, 0, nullptr, nullptr));
	clFinish(queue);

	//set some args that never change
	clSetKernelArg(vcolumnCalcCosts, 0, sizeof(cl_mem), &vcosts_g);
	clSetKernelArg(vcolumnCalcCosts, 1, sizeof(cl_mem), &vdirs_g);
	clSetKernelArg(vcolumnCalcCosts, 2, sizeof(cl_mem), &vals_g);
	clSetKernelArg(vcolumnCalcCosts, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(vcolumnCalcCosts, 4, sizeof(unsigned int), &window_height_g);

	clSetKernelArg(vcalcSeamToRemove, 0, sizeof(cl_mem), &vcosts_g);
	clSetKernelArg(vcalcSeamToRemove, 1, sizeof(cl_mem), &vdirs_g);
	clSetKernelArg(vcalcSeamToRemove, 2, sizeof(cl_mem), &vseam_g);
	clSetKernelArg(vcalcSeamToRemove, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(vcalcSeamToRemove, 4, sizeof(unsigned int), &window_height_g);
	clSetKernelArg(vcalcSeamToRemove, 6, sizeof(cl_mem), &vmin_g);


	clSetKernelArg(vseamremove, 0, sizeof(cl_mem), &image_g);
	clSetKernelArg(vseamremove, 1, sizeof(cl_mem), &vals_g);
	clSetKernelArg(vseamremove, 2, sizeof(cl_mem), &vseam_g);
	clSetKernelArg(vseamremove, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(vseamremove, 4, sizeof(unsigned int), &window_height_g);

	clSetKernelArg(hcolumnCalcCosts, 0, sizeof(cl_mem), &hcosts_g);
	clSetKernelArg(hcolumnCalcCosts, 1, sizeof(cl_mem), &hdirs_g);
	clSetKernelArg(hcolumnCalcCosts, 2, sizeof(cl_mem), &vals_g);
	clSetKernelArg(hcolumnCalcCosts, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(hcolumnCalcCosts, 4, sizeof(unsigned int), &window_height_g);

	clSetKernelArg(hcalcSeamToRemove, 0, sizeof(cl_mem), &hcosts_g);
	clSetKernelArg(hcalcSeamToRemove, 1, sizeof(cl_mem), &hdirs_g);
	clSetKernelArg(hcalcSeamToRemove, 2, sizeof(cl_mem), &hseam_g);
	clSetKernelArg(hcalcSeamToRemove, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(hcalcSeamToRemove, 4, sizeof(unsigned int), &window_height_g);
	clSetKernelArg(hcalcSeamToRemove, 6, sizeof(cl_mem), &hmin_g);


	clSetKernelArg(hseamremove, 0, sizeof(cl_mem), &image_g);
	clSetKernelArg(hseamremove, 1, sizeof(cl_mem), &vals_g);
	clSetKernelArg(hseamremove, 2, sizeof(cl_mem), &hseam_g);
	clSetKernelArg(hseamremove, 3, sizeof(unsigned int), &window_width_g);
	clSetKernelArg(hseamremove, 4, sizeof(unsigned int), &window_height_g);


	size_t globalWorkSize[] = {0,0,0};
	unsigned int vcount = 0, hcount = 0;
	float hmin, vmin;
	clFinish(queue); //make sure stuff has copied by now
	while(vcount < vseams || hcount < hseams)
	{
		//calculate columns costs and dirs
		clSetKernelArg(vcolumnCalcCosts, 5, sizeof(unsigned int), &vcount);
		globalWorkSize[0] = window_width_g - vcount;
		CheckError(clEnqueueNDRangeKernel(queue, vcolumnCalcCosts, 1, 
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

		clSetKernelArg(hcolumnCalcCosts, 5, sizeof(unsigned int), &hcount);
		globalWorkSize[0] = window_height_g - hcount;
		CheckError(clEnqueueNDRangeKernel(queue, hcolumnCalcCosts, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

		clFinish(queue);

		//calculate seam to remove
		clSetKernelArg(vcalcSeamToRemove, 5, sizeof(unsigned int), &vcount);
		globalWorkSize[0] = 1;
		CheckError(clEnqueueNDRangeKernel(queue, vcalcSeamToRemove, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

		clSetKernelArg(hcalcSeamToRemove, 5, sizeof(unsigned int), &hcount);
		CheckError(clEnqueueNDRangeKernel(queue, hcalcSeamToRemove, 1,
			nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));

		clFinish(queue);

		CheckError(clEnqueueReadBuffer(queue, vmin_g, CL_FALSE, 0,
			sizeof(float), &vmin, 0, nullptr, nullptr));
		CheckError(clEnqueueReadBuffer(queue, hmin_g, CL_FALSE, 0,
			sizeof(float), &hmin, 0, nullptr, nullptr));
		clFinish(queue);

		if(isScaled)
			hmin *= (source.rows - hcount)/(source.cols - (float)vcount);

		if(hcount < hseams && hmin < vmin)
		{
			//remove one seam
			clSetKernelArg(hseamremove, 5, sizeof(unsigned int), &hcount);
			globalWorkSize[0] = window_height_g;
			CheckError(clEnqueueNDRangeKernel(queue, hseamremove, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			clFinish(queue);
			hcount++;
		}
		else //remove vertical
		{
			//remove one seam
			clSetKernelArg(vseamremove, 5, sizeof(unsigned int), &vcount);
			globalWorkSize[0] = window_height_g;
			CheckError(clEnqueueNDRangeKernel(queue, vseamremove, 1,
				nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr));
			clFinish(queue);
			vcount++;
		}

	}

	//pull from cl_mem object
	CheckError(clEnqueueReadBuffer(queue, image_g, CL_TRUE, 0,
		sizeof(float) * window_size_g * 3, image, 0, nullptr, nullptr));

	//put in dest mat
	dest.create(source.rows-hcount,source.cols-vcount, CV_8UC3);
	// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
	//i is row, j is col
	for(int i = 0; i < dest.rows; i++)
	{
		for(int j = 0; j < dest.cols; j++)
		{
			Vec3b& pix = dest.at<Vec3b>(i,j);
			pix[0] = image[POSITION3(j,i,0)];
			pix[1] = image[POSITION3(j,i,1)];
			pix[2] = image[POSITION3(j,i,2)];
		}
	}
	clReleaseCommandQueue(queue);
	return true;
}

bool __seamcarve_both_RorS_cpu(int vseams, int hseams, const Mat& source, Mat& dest, bool isScaled)
{
	window_width = source.cols;
	int window_height = source.rows;
	unsigned long window_size = source.rows * source.cols;

	//only redo the memory stuff if it changes
	if(owidth != window_width || oheight != window_height)
	{
		__doMem(window_size,window_height);
	}

	if(window_width < vseams || window_height < hseams)
		return false;

	calcGreyscale(source,window_height);
	calcGradient(window_height);

	int vcount = 0, hcount = 0;
	while(vcount < vseams || hcount < hseams)
	{
		float vmin = 20000000, hmin = 20000000;
		//calc horizontal stuff
		if(hcount < hseams)
		{
			//init right col
			for(int y = 0; y < window_height - hcount; y++)
			{
				int pos = POSITION(window_width - 1, y);
				hcosts[pos] = vals[pos];
			}

			//calc rest of costs and dirs
			for(int x = window_width - 2; x >= 0; x--)
			{

				//do top pixel
				if(hcosts[POSITION(x+1,0)] < hcosts[POSITION(x+1,1)])
				{
					hcosts[POSITION(x,0)] = vals[POSITION(x,0)] + hcosts[POSITION(x+1,0)];
					hdirs[POSITION(x,0)] = 0; // right
				}
				else
				{
					hcosts[POSITION(x,0)] = vals[POSITION(x,0)] + hcosts[POSITION(x+1,1)];
					hdirs[POSITION(x,0)] = 1; //down
				}

				//middle
				int y;
				for(y = 1; y < window_height - hcount - 1; y++)
				{
					float cost_up   = hcosts[POSITION(x+1, y-1)];
					float cost_left = hcosts[POSITION(x+1, y  )]; //really goes to the right
					float cost_down = hcosts[POSITION(x+1, y+1)];
					int mypos = POSITION(x,y);

					if(cost_up < cost_left && cost_up < cost_down) // cost_up is min
					{
						hcosts[mypos] = vals[mypos] + cost_up;
						hdirs[mypos] = -1;
					}
					else if(cost_down < cost_left) // down is min
					{
						hcosts[mypos] = vals[mypos] + cost_down;
						hdirs[mypos] = 1;
					}
					else //straight left/right is min
					{
						hcosts[mypos] = vals[mypos] + cost_left;
						hdirs[mypos] = 0;
					}
				}
				// printf("Bottom\n");
				//bottom pixel
				int pos = POSITION(x,y);
				if(hcosts[POSITION(x+1, y)] < hcosts[POSITION(x+1,y-1)])
				{
					hcosts[pos] = vals[pos] + hcosts[POSITION(x+1,y)];
					hdirs[pos] = 0;
				}
				else
				{
					hcosts[pos] = vals[pos] + hcosts[POSITION(x+1,y-1)];
					hdirs [pos] = -1;
				}
			}//end calc costs and dirs

			// printf("Calc seam\n");

			//calc seams to remove
			for(int y = 0; y < window_height - hcount; y++)
			{
				if(hcosts[POSITION(0,y)] < hmin)
				{
					hmin = hcosts[POSITION(0,y)];
					hseam[0] = y;
				}
			}
		}
		//end calc horizontal stuff

		//calc vertical stuff
		if(vcount < vseams)
		{
			//init bottom row
			// printf("Init bottom row. %d cols\n", (window_width - count));
			for(int x = 0; x < (window_width - vcount); x++)
			{
				int pos = POSITION(x,window_height - 1);
				//printf("%d\n", pos);
				vcosts[pos] = vals[pos];
				//dirs[pos] = 0; // doesn't really matter
			}

			//calc rest of costs and dirs
			for(int y = window_height - 2; y >= 0; y--)
			{
				//do left side
				if(vcosts[POSITION(0, y+1)] < vcosts[POSITION(1,y+1)])
				{
					vcosts[POSITION(0, y)] = vals[POSITION(0,y)] + vcosts[POSITION(0, y+1)];
					vdirs[POSITION(0,y)] = 0; //up
				}
				else
				{
					vcosts[POSITION(0, y)] = vals[POSITION(0,y)] + vcosts[POSITION(1, y+1)];
					vdirs[POSITION(0,y)] = 1;
				}

				//middle
				int x;
				for(x = 1; x < window_width - vcount - 1; x++)
				{
					float cost_left  = vcosts[POSITION(x-1, y+1)];
					float cost_up    = vcosts[POSITION(x  , y+1)];
					float cost_right = vcosts[POSITION(x+1, y+1)];
					int mypos = POSITION(x,y);
					// if(mypos == 304)
					// 	printf("Got 304\n");

					if(cost_left < cost_up  && cost_left < cost_right) // cost_left is min
					{
						vcosts[mypos] = vals[mypos] + cost_left;
						vdirs[mypos] = -1;
					}
					else if(cost_right < cost_up && cost_right < cost_left) //cost_right is min
					{
						vcosts[mypos] = vals[mypos] + cost_right;
						vdirs[mypos] = 1;
					}
					else // up is min
					{
						vcosts[mypos] = vals[mypos] + cost_up;
						vdirs[mypos] = 0;
					}
				}

				//right side
				int pos = POSITION(x, y);
				if (vcosts[POSITION(x, y+1)] < vcosts[POSITION(x-1, y+1)]) {
	                vcosts[pos] = vals[pos] + vcosts[POSITION(x, y+1)];
	                vdirs [pos] = 0;
	            } else {
	                vcosts[pos] = vals[pos] + vcosts[POSITION(x-1, y+1)];
	                vdirs [pos] = -1;
	            }		
	        }//end calc costs and dirs

	        //calc seam to remove
	        for(int x = 0; x < window_width - vcount; x++)
	        {
	        	//printf("vcosts[%d] = %f\n", x,vcosts[POSITION(x,0)]);
	        	if(vcosts[POSITION(x,0)] < vmin)
	        	{
	        		vmin = vcosts[POSITION(x,0)];
	        		vseam[0] = x;
	        	}
	        }
	    }
		//end calc vertical stuff

		//hseam[0] holds the lowest h energy, vseam[0] holds the lowest v energy
		if(isScaled && hcount < hseams && vcount < vseams)
		{
			//scale so it doesn't matter if it's not square for the energy
			//if not scaled, and there are more rows than cols, then vmin will prob be larger
			//hmin -> cols and vmin -> rows
			// printf("Scaling vmin %f org hmin %f ",vmin,hmin); 
			hmin *= (source.rows - hcount)/(source.cols - (float)vcount);
			// printf("new hmin %f den %d num %d\n",hmin,(source.cols - vcount),(source.rows - hcount));
		}
		if((hcount < hseams && hmin < vmin))// || vcount >= vseams) //cheaper to do horizontal
		{
			for(int x = 1; x < window_width; x++)
				hseam[x] = hseam[x-1] + hdirs[POSITION(x-1,hseam[x-1])];

			//remove one horizontal seam
			for(int x = 0; x < window_width; x++)
			{
				int y;
				for(y = hseam[x]; y < window_height - hcount - 1; y++)
				{
					image[POSITION3(x,y,0)] = image[POSITION3(x,y+1,0)];
					image[POSITION3(x,y,1)] = image[POSITION3(x,y+1,1)];
					image[POSITION3(x,y,2)] = image[POSITION3(x,y+1,2)];

					vals[POSITION(x,y)] = vals[POSITION(x,y+1)];
				}

				image[POSITION3(x,y,0)] = 0;
				image[POSITION3(x,y,1)] = 0;
				image[POSITION3(x,y,2)] = 0;

				vals[POSITION(x,y)] = 0;
			}
			hcount++;
		}
		else //cheaper to do vertical
		{
			for(int y = 1; y < window_height; y++)
	        	vseam[y] = vseam[y-1] + vdirs[POSITION(vseam[y-1], y-1)];

	        //remove one seam
	        for(int y = 0; y < window_height; y++)
	        {
	        	int x;
	        	// printf("x %d, y %d\n", vseam[y], y);
	        	for(x = vseam[y]; x < window_width - vcount - 1; x++)
	        	{
	        		image[POSITION3(x,y,0)] = image[POSITION3(x+1,y,0)];
	        		image[POSITION3(x,y,1)] = image[POSITION3(x+1,y,1)];
	        		image[POSITION3(x,y,2)] = image[POSITION3(x+1,y,2)];

	        		vals[POSITION(x,y)] = vals[POSITION(x+1,y)];
	        	}

	        	image[POSITION3(x,y,0)] = 0;
	        	image[POSITION3(x,y,1)] = 0;
	        	image[POSITION3(x,y,2)] = 0;

	        	vals[POSITION(x,y)] = 0;
	        }
	        vcount++;
		}
	}
	dest.create(source.rows-hcount,source.cols-vcount, CV_8UC3);
	// printf("rows: %d, cols: %d\n", source.rows,source.cols-count);
	//i is row, j is col
	for(int i = 0; i < dest.rows; i++)
	{
		for(int j = 0; j < dest.cols; j++)
		{
			Vec3b& pix = dest.at<Vec3b>(i,j);
			pix[0] = image[POSITION3(j,i,0)];
			pix[1] = image[POSITION3(j,i,1)];
			pix[2] = image[POSITION3(j,i,2)];
		}
	}

	return true;
}

bool seamcarve_both_raw(int vseams, int hseams, const Mat& source, Mat& dest)
{
	return __seamcarve_both_RorS(vseams,hseams,source,dest,false);
}

bool seamcarve_both_scaled(int vseams, int hseams, const Mat& source, Mat& dest)
{
	return __seamcarve_both_RorS(vseams,hseams,source,dest,true);
}

bool seamcarve_both_raw_cpu(int vseams, int hseams, const Mat& source, Mat& dest)
{
	return __seamcarve_both_RorS_cpu(vseams,hseams,source,dest,false);
}

bool seamcarve_both_scaled_cpu(int vseams, int hseams, const Mat& source, Mat& dest)
{
	return __seamcarve_both_RorS_cpu(vseams,hseams,source,dest,true);
}
