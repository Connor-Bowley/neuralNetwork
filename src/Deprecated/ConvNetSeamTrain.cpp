/***************************
*
* This program:
* 	1a. Brings in a video from database.
*	1b. Seamcarve frames.
*	2.  Trains a cnn over the images.
*	3.  Makes file saying where training images came from.
*
*
****************************/


//ConvNet
#include <ConvNetCL.h>
#include <ConvNetSeam.h>

//OpenCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//MySQL
#include <mysql.h>

//Other
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;
using namespace cv;

#define mysql_query_check(conn,query) __mysql_check(conn, query, __FILE__, __LINE__)

#define CLASSES_IN_OUT_FRAME 0
#define CLASSES_DETAILED 1
#define CLASSES_SUPER_DETAILED 2

#define DISTORT_DOWN 0
#define SCALE_DOWN 1
#define CARVE_DOWN 2

int detailLevel = CLASSES_IN_OUT_FRAME;

vector<string> class_names;
vector<int> class_true_vals;

//class definitions
struct Event
{
	string type;
	int starttime;
	int endtime;
	bool isOvernight = false; //this means the starttime is before midnight and endtime is after
};

class Observations
{
	vector<Event> events;

public:
	void addEvent(string type, string starttime, string endtime);
	void getEvents(string tim, vector<Event>& dest);
	void getEvents(int tim, vector<Event>& dest);
	void getAllEvents(vector<Event>& dest);
};

//Global variables
MYSQL *wildlife_db_conn = NULL;

//Helper functions
void __mysql_check(MYSQL *conn, string query, const char* file, const int line)
{
	mysql_query(conn, query.c_str());
	if(mysql_errno(conn) != 0)
	{
		ostringstream ex_msg;
		ex_msg << "ERROR in MYSQL query: '" << query << "'. Error: " << mysql_errno(conn) << " -- '" << mysql_error(conn) << "'. Thrown on " << file << ":" << line;
		cerr << ex_msg.str() << endl;
		exit(1);
	}
}

int getTime(string tim) // must be in format hh::mm::ss. Military time
{
	int t = 0;
	t += stoi(tim.substr(tim.rfind(':')+1)); // seconds
	t += 60 * stoi(tim.substr(tim.find(':')+1, 2)); //minutes
	t += 3600 * stoi(tim.substr(0,2)); //hours
	return t;
}

bool containsEvent(vector<Event> events, string type)
{
	for(int i = 0; i < events.size(); i++)
	{
		if(events[i].type == type)
			return true;
	}
	return false;
}


//Class Level functions (and getTime)
void Observations::addEvent(string type, string starttime, string endtime)
{
	Event event;
	event.type = type;
	event.starttime = getTime(starttime);
	event.endtime = getTime(endtime);
	if(event.endtime < event.starttime) // possible if the starttime is before midnight and endtime is after
		event.isOvernight = true;
	events.push_back(event);
}

void Observations::getEvents(int tim, vector<Event>& dest)
{
	dest.resize(0);
	//seconds in a day = 3600 * 24 = 86400
	tim %= 86400; //make sure we are within a valid time for a day
	for(int i = 0; i < events.size(); i++)
	{
		//check if time is within event time. if so add to dest
		if(events[i].isOvernight) 
		{
			if(events[i].starttime <= tim || tim  <= events[i].endtime)
				dest.push_back(events[i]);
		}
		else
		{
			if(events[i].starttime <= tim && tim  <= events[i].endtime)
				dest.push_back(events[i]);
		}
	}
}

void Observations::getEvents(string tim, vector<Event>& dest)
{
	getEvents(getTime(tim),dest);
}

void Observations::getAllEvents(vector<Event>& dest)
{
	dest.resize(0);
	for(int i = 0; i < events.size(); i++)
		dest.push_back(events[i]);
}

//Other Functions
void init_wildlife_database()
{
	wildlife_db_conn = mysql_init(NULL);

	//get database info from file
	string db_host, db_name, db_password, db_user;
	ifstream db_info_file("../wildlife_db_info");
	db_info_file >> db_host >> db_name >> db_user >> db_password;
	db_info_file.close();

	printf("parsed db info, host: '%s', name: '%s', user: '%s', pass: '%s'\n", db_host.c_str(), db_name.c_str(), db_user.c_str(), db_password.c_str());

	if(mysql_real_connect(wildlife_db_conn, db_host.c_str(),db_user.c_str(), db_password.c_str(), db_name.c_str(), 0, NULL, 0) == NULL)
	{
		printf("Error conneting to database: %d, '%s'\n",mysql_errno(wildlife_db_conn), mysql_error(wildlife_db_conn));
		exit(1);
	}
}

bool setupDetailLevel(int detail)
{
	if(0 <= detail && detail <= 1)
	{
		detailLevel = detail;
		if(detailLevel == 0)
		{
			class_names.push_back("parent behavior - not in frame");
			class_true_vals.push_back(0);

			class_names.push_back("parent behavior - in frame");
			class_true_vals.push_back(1);
		}
		else if(detailLevel == 1)
		{
			class_names.push_back("parent behavior - not in frame");
			class_true_vals.push_back(0);

			class_names.push_back("parent behavior - on nest");
			class_true_vals.push_back(1);

			class_names.push_back("parent behavior - flying");
			class_true_vals.push_back(2);

			class_names.push_back("parent behavior - walking");
			class_true_vals.push_back(3);
		}
		return true;
	}
	else
	{
		printf("Unknown detail level '%d'. Exiting.\n", detail);
		return false;
	}
}

int getTrueVal(const vector<Event>& events)
{
	if(detailLevel == CLASSES_IN_OUT_FRAME)
	{
		if(containsEvent(events, "parent behavior - not in frame"))
			return 0;
		else if(containsEvent(events, "parent behavior - in frame"))
			return 1;
	}
	else if(detailLevel == CLASSES_DETAILED)
	{
		if(containsEvent(events, "parent behavior - not in frame"))
			return 0;
		else if(containsEvent(events, "parent behavior - on nest"))
			return 1;
		else if(containsEvent(events, "parent behavior - flying"))
			return 2;
		else if(containsEvent(events, "parent behavior - walking"))
			return 3;
	}
	else if(detailLevel == CLASSES_SUPER_DETAILED)
	{

	}
	return -1; // error unknown detail level or no events found
		
}

//do not use this function on multiple VideoCaptures at once. Run one throught to the end before
//going on to the next
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
		bool val = video.read(frame);
		firstGot = true;
		if(!val)
		{
			firstGot = false;
			moreFrames = false;
		}
	}

	return moreFrames;
}

int main(int argc, const char **argv)
{
	if(argc == 1)
	{
		printf("Usage: ./ConvNetSeamTrain \n");
		printf(" -cnn=<cnn_config>        Sets CNN architecture. Required.\n");
		printf(" -outname=<name>          Sets name of output cnn. Required.\n");
		printf(" -video=<video_id>        Picks a video to use for training. Must be in database. Can be used multiple times.\n");
		printf(" -species_id=<species_num>   Sets species to grab videos of\n");
		printf(" -max_videos=<int>        Max videos to bring in for training.\n");
		printf(" -max_time=<double>       Max number of hours of video to train on\n");
		printf(" -testPercent=<0-100>     Percent of data to use as test data.");
		printf(" -carveDown               Image is seamcarved both horizontally and vertically down to size\n");
		printf(" -scaleDown               Image is seamcarved to square and scaled down to size\n");
		printf(" -distortDown             Image is scaled down to size. No seamcarving. Possible distortion.\n");
		//printf(" -images=<path_to_images> Picks path_to_images for training. Can be used multiple times.\n");
		printf(" -device=<device_num>     OpenCL device to run CNN on\n");
		printf(" -jump=<int>              How many frames to jump between using frames. If jump=10,\n");
		printf("                          it will calc on frames 0, 10, 20, etc. Defaults to 1.\n");
		//printf(" -non_expert              Sets it not to pull from expert observed\n");

		printf(" -train_as_is. Default\n");
		printf(" -train_equal_prop\n");
		printf(" -detail=<int>            0 is in or out of frame (default). 1 is out of frame, on nest, flying\n");
		return 0;
	}
	//variable declarations
	vector<int> video_ids;
	bool expert = true;
	int jump = 1;
	int species_id = -1;
	string cnn_path = "";
	string outname = "";
	int max_videos = -1;
	int max_time = -1;
	int device = -1;
	bool train_as_is = true;
	int detail = 0;
	double testPercent = 0;
	int scaleType = SCALE_DOWN;
	int inputSize; //assumes square input
	Size cvSize;

	//0a. Parse through command line args
	for(int i = 1; i < argc; i++)
	{
		string arg(argv[i]);
		if(arg.find("-cnn=") != string::npos)
			cnn_path = arg.substr(arg.find('=')+1);
		else if(arg.find("-outname=") != string::npos)
			outname = arg.substr(arg.find('=')+1);
		else if(arg.find("-video=") != string::npos)
			video_ids.push_back(stoi(arg.substr(arg.find('=')+1)));
		else if(arg.find("-species_id=") != string::npos)
			species_id = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-max_videos=") != string::npos)
			max_videos = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-max_time=") != string::npos)
			max_time = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-device=") != string::npos)
			device = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-jump=") != string::npos)
			jump = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-train_equal_prop") != string::npos)
			train_as_is = false;
		else if(arg.find("-detail=") != string::npos)
			detail = stoi(arg.substr(arg.find('=')+1));
		else if(arg.find("-testPercent=") != string::npos)
			testPercent = stod(arg.substr(arg.find('=')+1));
		else if(arg.find("-distortDown") != string::npos)
			scaleType = DISTORT_DOWN;
		else if(arg.find("-scaleDown") != string::npos)
			scaleType = SCALE_DOWN;
		else if(arg.find("-carveDown") != string::npos)
			scaleType = CARVE_DOWN;
		else
		{
			printf("Unknown arg: '%s'\n", argv[i]);
			return 0;
		}
	}

	if(cnn_path == "")
	{
		printf("You must supply a CNN config file.\n");
		return 0;
	}
	if(outname == "")
	{
		printf("You must supply a name for the output CNN.\n");
		return 0;
	}

	if(!setupDetailLevel(detail))
		return 0;

	Net net;
	net.load(cnn_path.c_str());
	inputSize = net.getInputWidth();
	cvSize = Size(inputSize,inputSize);

	if(max_time != -1)//turn hours to seconds
		max_time *= 3600;

	//0b. Open Wildlife Database
	init_wildlife_database();

	printf("database inited\n");

	//1a. Find videos in Database
	//from video_2 need id, archive_filename, start_time, duration_s

	ostringstream query;
	query << "SELECT id, archive_filename, start_time, duration_s FROM video_2 WHERE";
	if(expert)
		query << " expert_finished = 'FINISHED'";
	else
		query << " crowd_status = 'VALIDATED'";
	if(species_id != -1)
		query << " AND species_id = " << species_id;
	if(video_ids.size() > 0)
	{
		query << " AND ( ";
		for(int i = 0; i < video_ids.size(); i++)
		{
			if(i > 0)
				query << " OR";
			query << " id = " << video_ids[i];
		}
		query << " )";
	}
	if(max_videos != -1)
		query << " LIMIT " << max_videos;

	query << ";";

	//make MySQL query and get results
	printf("Running query\n'%s'\n", query.str().c_str());
	mysql_query_check(wildlife_db_conn, query.str());
	MYSQL_RES *video_2_result = mysql_store_result(wildlife_db_conn);
	printf("Query to video_2 made.\n");

	MYSQL_ROW video_row;
	int current_time = 0;
	vector<vector<Mat> > trainingData; // should probably find some way of reserving mem for this
	vector<vector<double> > training_trueVals;
	unsigned long totalAmountData = 0;
	while((video_row = mysql_fetch_row(video_2_result)))
	{
		//for each video, get variables
		int video_id = atoi(video_row[1]);
		string video_path = video_row[2];
		string video_name = video_path.substr(video_path.rfind('/')+1);
		string startDateAndTime = video_row[3];
		int starttime = getTime(startDateAndTime.substr(startDateAndTime.find(' ')+1));
		int duration = atoi(video_row[4]);

		printf("Video #%d, %s. Dur: %d\n", video_id, video_name.c_str(), duration);

		//if we are going to go over max time, continue and see if there is a shorter video
		if(max_time != -1 && current_time + duration > max_time)
			continue;
		current_time += duration;

		//wget video (in separate thread?)
		string sys_cmd = "wget -q http://wildlife.und.edu" + video_path;
		system(sys_cmd.c_str());

		//get observations from video and put in observations
		Observations observations;
		ostringstream expert_query;
		expert_query << "SELECT event_type, start_time, end_time FROM expert_observations"
			<< " WHERE video_id = " << video_id << ";";
		printf("Query to expert_observations:\n'%s'\n", expert_query.str().c_str());
		mysql_query_check(wildlife_db_conn, expert_query.str());
		MYSQL_RES *obs_result = mysql_store_result(wildlife_db_conn);
		
		MYSQL_ROW obs_row;
		while((obs_row = mysql_fetch_row(obs_result)))
		{
			string type = obs_row[0];
			string start = obs_row[1];
			string end = obs_row[2];
			observations.addEvent(type, start, end);
		}
		mysql_free_result(obs_result); // free result because now it is stored in observations

		//open video
		VideoCapture video(video_name.c_str());
		if(!video.isOpened()) // if not opening, try next video
		{
			printf("Could not open video '%s' from path '%s'\n", video_name.c_str(), video_path.c_str());
			continue;
		}

		//positive numSeams means width  > height - landscape
		//negative numSeams means height > width  - portrait
		int numSeams = video.get(CV_CAP_PROP_FRAME_WIDTH) - video.get(CV_CAP_PROP_FRAME_HEIGHT);

		Mat frame;
		int framenum = 0;
		vector<Event> curEvents;
		//go through video frame by frame.
		trainingData.resize(trainingData.size() + 1);
		training_trueVals.resize(training_trueVals.size() + 1);
		while(getNextFrame(video, frame, framenum, jump))
		{
			//seamcarve/scale frame and put in trainingData
			trainingData.back().resize(trainingData.size() + 1);
			if(scaleType == DISTORT_DOWN) // straight scale to size. Distort if necessary
			{
				resize(frame,trainingData.back().back(),cvSize);
			}
			else if(scaleType == SCALE_DOWN) // seamcarve to square. Scale to size
			{
				Mat temp;
				if(numSeams > 0) //width > height. landscape
				{
					//vertical seams, fast
					seamcarve_vf(numSeams,frame,temp);//bring us to square
					resize(temp, trainingData.back().back(),cvSize);
				}
				else // height > width. portrait
				{
					//horizontal seams, fast
					//seamcarve_hf(numSeams, frame, temp);
					resize(temp, trainingData.back().back(),cvSize);
				}
			}
			else if(scaleType == CARVE_DOWN) // seamcarve in both directions down to size. No normal scaling
			{
				//both types seams, fast
				//seamcarve_bf(frame, trainingData.back().back());
			}

			//get true val and put in
			observations.getEvents(starttime + framenum * .1, curEvents); //assuming 10 frames per second.
			int trueVal = getTrueVal(curEvents);
			training_trueVals.back().push_back(trueVal);
		}
		video.release();
			
		totalAmountData += trainingData.back().size();
		//rm video
		sys_cmd = "rm " + video_name;
		system(sys_cmd.c_str());
	}

	//free video_2_result
	mysql_free_result(video_2_result);

	if(scaleType == CARVE_DOWN || scaleType == SCALE_DOWN)
		seamcarve_cleanup();


	//2. Train CNN over seamcarved images
	//initialized up top
	net.setSaveName(outname);
	net.setClassNames(class_names,class_true_vals);
	if(train_as_is)
		net.setTrainingType(TRAIN_AS_IS);
	else
		net.setTrainingType(TRAIN_EQUAL_PROP);
	if(device != -1)
		net.setDevice(device);
	if(!net.finalize())
	{
		printf("Error finalizing Net: \n%s\n",net.getErrorLog().c_str());
		return 0;
	}

	printf("CNN Layer Sizes\n");
	net.printLayerDims();

	//add training and test data. get percentages as close as possible 
	//while using separate vidoes for each
	//if percent diff > 5 tell the user?
	unsigned long minSize = -1; // should be max ulong
	int minIndex = -1;
	if(testPercent == 0 || trainingData.size() < 2)
	{
		//put it all for training
		for(int i = 0; i < trainingData.size(); i++)
			net.addTrainingData(trainingData[i],training_trueVals[i]);
	}
	else
	{
		//for accountability reasons, try to keep training and test videos from separate videos
		//if we can't get percents right, use smallest video for test
		vector<bool> isTraining(trainingData.size(),true);
		testPercent *= .01; //make this a decimal again.
		double totalTest = 0;
		bool noTestFound = true;
		for(int i = 0; i < trainingData.size(); i++)
		{
			double curPercent = trainingData[i].size() / totalAmountData;
			if(curPercent + totalTest < testPercent)
			{
				isTraining[i] = false;
				totalTest += curPercent;
				noTestFound = false;
			}
			// if we can make this and only go over by 3% lets do it.
			else if(curPercent + totalTest < testPercent + .03)
			{
				isTraining[i] = false;
				totalTest += curPercent;
				break;
			}
			else
			{
				if(noTestFound && trainingData[i].size() < minSize)
				{
					minSize = trainingData[i].size();
					minIndex = i;
				}
			}
		}

		if(noTestFound)
		{
			isTraining[minIndex] = false;
			totalTest += trainingData[minIndex].size() / totalAmountData;
		}

		//now we know who is training and who is test
		for(int i = 0; i < trainingData.size(); i++)
		{
			if(isTraining[i])
				net.addTrainingData(trainingData[i],training_trueVals[i]);
			else
				net.addTestData(trainingData[i],training_trueVals[i]);
		}
	}

	printf("Training Distribution\n");
	net.printTrainingDistribution();

	net.train(); //this will save using the save name because of net.setSaveName called above
}