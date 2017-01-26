

#ifndef ____ConvNetSeam__
#define ____ConvNetSeam__

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//OpenCL
#ifdef __APPLE__
 	#include "OpenCL/opencl.h"
#else
 	#include "CL/cl.h"
#endif

// boinc
#ifdef _BOINC_APP_
	#include "diagnostics.h"
	#include "filesys.h"
	#include "boinc_api.h"
	#include "mfile.h"
	#include "proc_control.h"
#endif

bool seamcarve_vf      (int numSeams, const cv::Mat& source, cv::Mat& dest);
bool seamcarve_vf_cpu  (int numSeams, const cv::Mat& source, cv::Mat& dest);
bool seamcarve_vfRev   (int numSeams, const cv::Mat& source, cv::Mat& dest);
bool seamcarve_vfRandom(int numSeams, const cv::Mat& source, cv::Mat& dest);

bool seamcarve_hf    (int numSeams, const cv::Mat& source, cv::Mat& dest);
bool seamcarve_hf_cpu(int numSeams, const cv::Mat& source, cv::Mat& dest);

bool seamcarve_both_vth    (int vseams, int hseams, const cv::Mat& source, cv::Mat& dest);
bool seamcarve_both_vth_cpu(int vseams, int hseams, const cv::Mat& source, cv::Mat& dest);

bool seamcarve_both_htv    (int hseams, int vseams, const cv::Mat& source, cv::Mat& dest);
bool seamcarve_both_htv_cpu(int hseams, int vseams, const cv::Mat& source, cv::Mat& dest);

bool seamcarve_both_raw    (int vseams, int hseams, const cv::Mat& source, cv::Mat& dest);
bool seamcarve_both_raw_cpu(int vseams, int hseams, const cv::Mat& source, cv::Mat& dest);

bool seamcarve_both_scaled    (int vseams, int hseams, const cv::Mat& source, cv::Mat& dest);
bool seamcarve_both_scaled_cpu(int vseams, int hseams, const cv::Mat& source, cv::Mat& dest);

void seamcarve_cleanup();

void seamcarve_setDevice(int deviceNum);
void seamcarve_setDevice(cl_device_id device, cl_platform_id platform);

#endif /* defined(____ConvNetSeam__)*/
