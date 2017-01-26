#include "ConvNetEvent.h"
#include "ConvNetCommon.h"

using namespace convnet;

//Vectors for each classes define
void getClasses(int classLevel, std::vector<int>& dest)
{
	dest.clear();
	if(classLevel == CLASSES_ON_OFF_OUT)
	{
		dest.push_back(PARENT_BEHAVIOR__ON_NEST);
		dest.push_back(PARENT_BEHAVIOR__OFF_NEST);
		dest.push_back(PARENT_BEHAVIOR__NOT_IN_VIDEO);
	}
	else if(classLevel == CLASSES_DETAILED)
	{
		dest.push_back(PARENT_BEHAVIOR__NOT_IN_VIDEO);
		dest.push_back(PARENT_BEHAVIOR__ON_NEST);
		dest.push_back(PARENT_BEHAVIOR__FLYING);
		dest.push_back(PARENT_BEHAVIOR__WALKING);
	}
	else 
	{
		printf("Class level not currently supported\n");
		exit(0);
	}
}

Event::Event()
{
	starttime = -1;
	endtime = -1;
}

bool Event::equals(const Event &other)
{
	if(type == other.type && starttime == other.starttime && endtime == other.endtime)
		return true;
	return false;
}

Event::Event(std::string type, long starttime, long endtime)
{
	this->type = type;
	this->starttime = starttime;
	this->endtime = endtime;
	if(endtime < starttime)
		this->isOvernight = true;
	else
		this->isOvernight = false;
	this->starttime_string = getTime(starttime);
	this->endtime_string = getTime(endtime);
}

Event::Event(std::string event)
{
	std::string line;
	std::stringstream ss(event);
	getline(ss,line); //just says EVENT
	getline(ss,line); 
	std::string type = line;
	getline(ss,line);
	std::string starttime = line;
	getline(ss, line);
	std::string endtime = line;
	Event(type,getTime(starttime),getTime(endtime));
}

Event::Event(const Event& other)
{
	type = other.type;
	starttime = other.starttime;
	endtime = other.starttime;
	isOvernight = other.isOvernight;
	starttime_string = other.starttime_string;
	endtime_string = other.endtime_string;
}

std::string Event::toString() const
{
	std::string out = "EVENT\n";
	out += type; out += "\n";
	out += starttime_string; out += "\n";
	out += endtime_string; out += "\n";

	return out;
}


bool containsEvent(std::vector<Event> events, std::string type)
{
	for (auto&& e : events) {
		if(e.type == type)
			return true;
	}

	return false;
}

Observations::Observations()
{}

Observations::Observations(std::string obs)
{
	this->load(obs);
}

bool Observations::equals(const Observations& other)
{
	if(events.size() != other.events.size())
		return false;

	std::vector<Event> e1 = events;
	std::vector<Event> e2 = other.events;

	while(e1.size() > 0)
	{
		bool found = false;
		for(unsigned int i = 0; i < e2.size(); i++)
		{
			if(e1[0].equals(e2[i]))
			{
				e1.erase(e1.begin());
				e2.erase(e2.begin()+i);
				found = true;
				break;
			}
		}
		if(!found)
			return false;
	}
	return true;
}

void Observations::load(std::string obs)
{
	std::stringstream ss(obs);
	std::string line;
	getline(ss,line);
	std::stringstream event; //should say EVENT
	event << line << '\n';
	while(getline(ss,line))
	{
		if(line == "EVENT")
		{
			events.push_back(Event(event.str()));
			event.clear();
			event << line << '\n';
		}
		else
			event << line << '\n';
	}
}

//Class Level functions (and getTime)
void Observations::addEvent(std::string type, int starttime, int endtime)
{
	events.push_back(Event(type,starttime,endtime));
}

void Observations::addEvent(std::string type, std::string starttime, std::string endtime)
{
	Event event(type, getTime(starttime), getTime(endtime));
	events.push_back(event);
	// printf("event: %s, start %s|%d, end %s|%d\n", event.type.c_str(), starttime.c_str(), event.starttime, endtime.c_str(), event.endtime);
}

void Observations::getEvents(int tim, std::vector<Event>& dest)
{
	dest.clear();
	//seconds in a day = 3600 * 24 = 86400
	tim %= 86400; //make sure we are within a valid time for a day
	for(unsigned int i = 0; i < events.size(); i++)
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

void Observations::getEvents(std::string tim, std::vector<Event>& dest)
{
	getEvents(getTime(tim),dest);
}

void Observations::getAllEvents(std::vector<Event>& dest)
{
	dest.resize(0);
	for(unsigned int i = 0; i < events.size(); i++)
		dest.push_back(events[i]);
}

std::string Observations::toString() const
{
	std::stringstream ss;
	for(size_t i = 0; i < events.size(); i++)
	{
		ss << events[i].toString();
	}
	return ss.str();
}
