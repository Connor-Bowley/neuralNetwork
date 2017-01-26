/*
Checkpoint txt looks like
curFrame
framenum1 classIndex1
framenum2 classIndex2
...

curFrame, framenum, classIndex are ints
*/
//ConvNet
#include <ConvNetCL.h>
#include <ConvNetSeam.h>
#include <ConvNetEvent.h>

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//Other
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <time.h>

//BOINC
#ifdef _BOINC_APP_
#include "diagnostics.h"
#include "filesys.h"
#include "boinc_api.h"
#include "mfile.h"
#include "proc_control.h"
#include "boinc_opencl.h"
#endif

using namespace std;
using namespace cv;

struct DoneFrame{
	int framenum;
	int classIndex;
	vector<double> confidences;

	DoneFrame(int framenum, int classIndex, vector<double>& confidences);
	DoneFrame();
};

DoneFrame::DoneFrame()
{
	this->framenum = -1;
	this->classIndex = -1;
}

DoneFrame::DoneFrame(int framenum, int classIndex, vector<double>& confidences)
{
	this->framenum = framenum;
	this->classIndex = classIndex;
	this->confidences = confidences;
}

vector<DoneFrame> doneFrames;
int curFrame = 0;
int numClasses;

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

void writeCheckpoint() throw(std::runtime_error)
{
	if(doneFrames.size() == 0)
		return;
	string resolved_checkpoint_name = getBoincFilename("checkpoint.txt");
	ofstream check(resolved_checkpoint_name.c_str(), ios::trunc);
	check << curFrame << endl;
	check << doneFrames.size() << endl;
	check << doneFrames[0].confidences.size() << endl;
	for(size_t i = 0; i < doneFrames.size(); i++)
	{
		check << doneFrames[i].framenum << " " << doneFrames[i].classIndex;
		for(int j = 0; j < doneFrames[i].confidences.size(); j++)
			check << " " << doneFrames[i].confidences[j];
		check << endl;
	}
	check.close();

}

bool readCheckpoint()
{
	string line, resolved_checkpoint_name = getBoincFilename("checkpoint.txt");
	int framenum, classIndex;
	
	ifstream check(resolved_checkpoint_name.c_str());
	if(!check.is_open())
		return false;
	//the first line is our current frame
	getline(check, line);
	curFrame = stoi(line);
	//the second line is the length of the vector
	getline(check, line);
	// doneFrames.resize(stoi(line)); //don't need to do b/c push_back()
	//the third line is the number of classes
	getline(check, line);
	int numClasses = stoi(line);
	vector<double> curConf(numClasses);
	//the rest of the lines are frame numbers and calculated class indexes
	while(getline(check, line))
	{
		stringstream ss(line);
		ss >> framenum >> classIndex;
		for(int i = 0; i < numClasses; i++)
			ss >> curConf[i];
		doneFrames.push_back(DoneFrame(framenum, classIndex,curConf));
	}
	check.close();
	return true;
}

//Other functions

bool getNextFrame(VideoCapture& video, Mat& frame, int& framenum, int jump = 1)
{
	// firstGot should regulate itself so it'll reset when a video runs out of frames
	static bool firstGot = false; 
	bool moreFrames = true;
	if(firstGot) // not first frame
	{
		bool val1 = true;
		for(int i = 0; i < jump; i++)
			if(!video.grab())
				val1 = false;
		bool val2 = video.retrieve(frame);
		framenum += jump;
		if(!val1 || !val2)
		{
			firstGot = false; // this means video ended
			moreFrames = false;
		}
	}
	else // first frame
	{
		bool val = video.grab();
		firstGot = true;
		if(!val)
		{
			printf("first frame val failed on grab\n");
			firstGot = false;
			moreFrames = false;
		}
		val = video.retrieve(frame);
		if(!val)
		{
			printf("first frame failed on retreive\n");
			firstGot = false;
			moreFrames = false;
		}
		framenum++;
	}
	return moreFrames;
}

int main(int argc, char** argv)
{
	if(argc == 1)
	{
		printf("Flags:\n");
		printf("REQUIRED\n");
		printf(" -cnn=<trained_cnn_file>        Sets CNN to be used for running\n");
		printf(" -video=<videoName>             Sets video to run CNN over\n");
		printf(" -video_start_time=<int>        The start of the video from the DB in seconds.\n");
		printf(" -video_id=<int>                The unique id for the video.\n");
		printf(" -cnn_config_id=<int>           The unique id for the cnn_config\n");
		printf(" One of the following describing how CNN was trained:\n");
		printf("   -carveDown_both_scaled\n");
		printf("   -carveDown_both_raw\n");
		printf("   -carveDown_vth\n");
		printf("   -carveDown_htv\n");
		printf("   -scaleDown\n");
		printf("   -distortDown\n");
		printf("   -random_crop\n");
		printf("OPTIONAL\n");
		printf(" -device=<device_num>           Device to run CNN (and seamcarving if applicable) on. Default 0. Can't use if running on BOINC\n");
		printf(" -jump=<uint>                   Will run on every \"jump-th\" frame. Default 1.\n");
		printf(" -batchSize=<uint>              How many frames that will be seamcarved before running through cnn. Default 10.\n");
		return 0;
	}

	string cnn = "..";
	string video_path = "..";
	int scaleType = -674;
	unsigned int jump = 1;
	int device = 0;
	unsigned int batchSize = 10;
	int fps = 10;
	int video_start_time = -1;
	int video_id = -1;
	int cnn_config_id = -1;

	//get cmd line args
	for(int i = 1; i < argc; i++)
	{
		string arg(argv[i]);
		if(arg.find("-cnn=") != string::npos)
			cnn = arg.substr(arg.find('=')+1);
		else if(arg.find("-video=") != string::npos)
			video_path = arg.substr(arg.find('=')+1);
		else if(arg.find("-video_id=") != string::npos)
			video_id = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-cnn_config_id=") != string::npos)
			cnn_config_id = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-video_start_time=") != string::npos)
			video_start_time = stoi(arg.substr(arg.find('=')+1));
		else if(arg == "-carveDown_both_scaled")
			scaleType = CARVE_DOWN_BOTH_SCALED;
		else if(arg == "-carveDown_both_raw")
			scaleType = CARVE_DOWN_BOTH_RAW;
		else if(arg == "-carveDown_vth")
			scaleType = CARVE_DOWN_VTH;
		else if(arg == "-carveDown_htv")
			scaleType = CARVE_DOWN_HTV;
		else if(arg == "-scaleDown")
			scaleType = SCALE_DOWN;
		else if(arg == "-distortDown")
			scaleType = DISTORT_DOWN;
		else if(arg == "random_crop")
			scaleType = RANDOM_CROP;
		else if(arg.find("-device=") != string::npos)
			device = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-jump=") != string::npos)
			jump = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-batchSize=") != string::npos)
			batchSize = stoi(arg.substr(arg.find('=')+1));
		else
		{
			printf("Unknown arg \"%s\"\n", argv[i]);
			return 0;
		}
	}

	if(cnn == ".." || video_path == ".." || video_id == -1 || video_start_time == -1 || scaleType == -674 || cnn_config_id == -1)
	{
		printf("You must have all the required args.\n");
		return 0;
	}

	#ifdef _BOINC_APP_
	int retval;
	cl_device_id cldevice;
	cl_platform_id platform;
	boinc_init_diagnostics(BOINC_DIAG_MEMORYLEAKCHECKENABLED);
	BOINC_OPTIONS options;
	boinc_options_defaults(options);
	options.multi_thread = true;  // for multiple threads in OpenCL
	options.multi_process = true; // for multiple processes in OpenCL?
	options.normal_thread_priority = true; // so GPUs will run at full speed
	boinc_init_options(&options);
	boinc_init();

	retval = boinc_get_opencl_ids(argc, argv, -1, &cldevice, &platform);
    if (retval) {
        fprintf(stderr, 
            "Error: boinc_get_opencl_ids() failed with error %d\n", 
            retval
        );
        return 1;
    }

	#endif

	string resolved_cnn = getBoincFilename(cnn);
	string resolved_video_path = getBoincFilename(video_path);


	//set up net
	Net net(resolved_cnn.c_str());
	
	#ifdef _BOINC_APP_
		net.setDevice(cldevice,platform);
	#else
		net.setDevice(device);
	#endif
	if(!net.finalize())
	{
		#ifdef _BOINC_APP_
		boinc_finish(-1);
		#endif
		return -1;
	}
	seamcarve_setDevice(device);

	// int inputHeight = net.getInputHeight();
	int inputWidth = net.getInputWidth();

	int inputSize = inputWidth;
	cv::Size cvSize = cv::Size(inputSize, inputSize);

	#ifdef _BOINC_APP_
	if(readCheckpoint())
	{
		printf("Continuing from checkpoint\n");
	}
	else
	{
		printf("No Checkpoint found. Starting from beginning\n");
	}
	#endif

	//set up video
	VideoCapture video(resolved_video_path.c_str());
	if(!video.isOpened())
	{
		cout << "Could not open video: " << resolved_video_path << endl;
		#ifdef _BOINC_APP_
		boinc_finish(-1);
		#endif
		return -1;
	}

	//get to current frame
	for(int i = 0; i < curFrame; i++)
		video.grab();

	bool cont = true;
	vector<int> calcClasses;
	vector<vector<double> > curConfidences;
	Mat frame;

	int frameWidth = video.get(CV_CAP_PROP_FRAME_WIDTH);
	int frameHeight = video.get(CV_CAP_PROP_FRAME_HEIGHT);
	//used for random crop
	default_random_engine gen(time(NULL));
	uniform_int_distribution<int> disI(0,frameHeight - inputSize);
	uniform_int_distribution<int> disJ(0,frameWidth - inputSize);

	//used for seamcarving
	int vseams = frameWidth  - inputSize;
	int hseams = frameHeight - inputSize;
	int numSeams = frameWidth - frameHeight;

	while(cont)
	{
		#ifdef _BOINC_APP_
		boinc_fraction_done(video.get(CV_CAP_PROP_POS_AVI_RATIO) * 100.0);
		if(boinc_time_to_checkpoint())
		{
			writeCheckpoint();
			boinc_checkpoint_completed();
		}
		#else
		printf("Percent done: %lf\n", video.get(CV_CAP_PROP_POS_AVI_RATIO) * 100.0);
		#endif
		//get next batchSize of preprocessed images
		vector<Mat> currentFrames;
		vector<int> currentFramenums;
		for(int i = 0; i < batchSize; i++)
		{
			if(!getNextFrame(video,frame,curFrame,jump))
			{
				cont = false;
				break;
			}
			Mat tempMat;
			if(scaleType == DISTORT_DOWN) // straight scale to size. Distort if necessary
			{
				resize(frame,tempMat,cvSize);
			}
			else if(scaleType == RANDOM_CROP)
			{
				int si = disI(gen);
				int sj = disJ(gen);
				Mat temp(frame,Range(si,si+inputSize),Range(sj,sj+inputSize));
				tempMat = temp;
			}
			else if(scaleType == SCALE_DOWN) // seamcarve to square. Scale to size
			{
				Mat temp;
				if(numSeams > 0) //width > height. landscape
				{
					//vertical seams, fast
					seamcarve_vf(numSeams,frame,temp);//bring us to square
					resize(temp, tempMat,cvSize);
				}
				else // height > width. portrait
				{
					//horizontal seams, fast
					seamcarve_hf(-numSeams, frame, temp);
					resize(temp, tempMat,cvSize);
				}
			}
			else if(scaleType == CARVE_DOWN_VTH) // seamcarve in both directions down to size. No normal scaling
				seamcarve_both_vth(vseams, hseams, frame, tempMat);
			else if(scaleType == CARVE_DOWN_HTV)
				seamcarve_both_htv(hseams, vseams, frame, tempMat);
			else if(scaleType == CARVE_DOWN_BOTH_RAW)
				seamcarve_both_raw(vseams, hseams, frame, tempMat);
			else if(scaleType == CARVE_DOWN_BOTH_SCALED)
				seamcarve_both_scaled(vseams, hseams, frame, tempMat);
			else
			{
				printf("Unknown scaleType %d\n", scaleType);
				return 0;
			}
			currentFrames.push_back(tempMat);
			currentFramenums.push_back(curFrame);

		}

		// for(int i = 0; i < currentFramenums.size(); i++)
		// {
		// 	printf("%d ", currentFramenums[i]);
		// }
		// printf("\n");

		//run them through CNN and store results
		net.setData(currentFrames);
		net.run();
		net.getCalculatedClasses(calcClasses);
		net.getConfidences(curConfidences);

		for(int i = 0; i < currentFramenums.size(); i++)
			doneFrames.push_back(DoneFrame(currentFramenums[i],calcClasses[i], curConfidences[i]));

	}

	//now everything has been run through the cnn and we need to generate Observations
	Observations obs;

	//in theory, doneFrames should be in order by frame number
	for(int i = 0; i < doneFrames.size();)
	{
		int curClassIndex = doneFrames[i].classIndex;
		int startFrame = doneFrames[i].framenum;
		i++;
		while(i < doneFrames.size() && doneFrames[i].classIndex == curClassIndex)
			i++;
		int endFrame = doneFrames[i-1].framenum;

		int curStartTime = video_start_time + startFrame / fps;
		int curEndTime   = video_start_time + endFrame / fps;

		obs.addEvent(net.getClassForTrueVal(curClassIndex), curStartTime, curEndTime);
	}

	/*
	Fields in database
		id: int(11) autoincrement id for cnn_observations
		event_type: enum(...)
		cnn: cnn_id???
		start_time: time (hh:mm:ss)
		video_id: int(11)
		end_time: time (hh:mm:ss)
	*/

	/* 
	Format output file

	cnn:string //assimilator will turn into cnn_id
	video_name:string //assimilator will turn into video_id
	EVENT
	EVENT
	EVENT

	where EVENT is from Event::toString 
	*/

	//list of calculated events with no smoothing / signal processing / etc.
	ofstream outfile(getBoincFilename("results.txt"));
	outfile << cnn_config_id << endl;
	outfile << video_id << endl;
	outfile << obs.toString();
	outfile.close();

	//same format as checkpoint file
	ofstream rawout(getBoincFilename("raw_results.txt"));
	rawout << curFrame << endl;
	rawout << doneFrames.size() << endl;
	rawout << doneFrames[0].confidences.size() << endl;
	for(size_t i = 0; i < doneFrames.size(); i++)
	{
		rawout << doneFrames[i].framenum << " " << doneFrames[i].classIndex;
		for(int j = 0; j < doneFrames[i].confidences.size(); j++)
			rawout << " " << doneFrames[i].confidences[j];
		rawout << endl;
	}
	rawout.close();


	#ifdef _BOINC_APP_
	boinc_finish(0);
	#endif

	return 0;
}








