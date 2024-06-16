//Author: Laurie Bose
//Date: 2021

#include <scamp5.hpp>
#include <list>
#include "REGISTER_ENUMS.hpp"
#include <string>
using namespace SCAMP5_PE;
using namespace std;
#include <sstream>

namespace TIMING_STATS
{
	list<uint32_t> timing_values;
	list<uint32_t> timing_samples;
	list<string> timing_nametags;
	bool enabled = true;
	vs_stopwatch timer;

	vs_stopwatch frame_timer;
	const int frame_time_sample_count = 10;
	int frame_time_sample_index = 0;
	int frame_time_samples[frame_time_sample_count];

	string current_nametag;
	int current_samples;

	void print_and_clear(bool clear_text_console)
	{
		list<uint32_t>::iterator itvalues = timing_values.begin();
		list<uint32_t>::iterator itsamples = timing_samples.begin();
		list<string>::iterator itnametags = timing_nametags.begin();

		string output = "!clear ";
		std::ostringstream output_stream;

		uint32_t total_timings = 0;
		while(itvalues != timing_values.end())
		{
			total_timings += (*itvalues);
			itvalues++;
		}
		itvalues = timing_values.begin();

		while(itnametags != timing_nametags.end())
		{
			int tmp_val = (100*(*itvalues))/ (*itsamples);
			double average_value = 0.01*tmp_val;
			int percent_value = (100*(*itvalues))/total_timings;
			output_stream << *itnametags << " %:" << percent_value << " tot:" << (*itvalues) << " avg:" << average_value << " samples:" << (*itsamples) << '\n';

			itnametags++;
			itvalues++;
			itsamples++;
		}

		output_stream << "total timed: " << total_timings << '\n';

		timing_values.clear();
		timing_samples.clear();
		timing_nametags.clear();

		output_stream << "total frame: " << frame_time_samples[frame_time_sample_index] << '\n';

		int average_fps = 0;
		for(int n = 0 ; n < frame_time_sample_count ; n++)
		{
			average_fps +=frame_time_samples[n];
		}
		average_fps = 1000000*frame_time_sample_count/average_fps;
		output_stream << "FPS " << average_fps << '\n';

		vs_post_text("!clear");
		vs_post_text("%s\n",(output_stream.str()).c_str());
	}
}

void tick(string nametag,int samples)
{
	if(TIMING_STATS::enabled)
	{
		TIMING_STATS::current_nametag = nametag;
		TIMING_STATS::current_samples = samples;
		TIMING_STATS::timer.reset();
	}
}

void tick(string nametag)
{
	if(TIMING_STATS::enabled)
	{
		TIMING_STATS::current_nametag = nametag;
		TIMING_STATS::current_samples = 1;
		TIMING_STATS::timer.reset();
	}
}

void tock()
{
	if(TIMING_STATS::enabled)
	{
		uint32_t time = TIMING_STATS::timer.get_usec();
		TIMING_STATS::timer.reset();

		bool new_entry = true;
		list<uint32_t>::iterator itvalues = TIMING_STATS::timing_values.begin();
		list<uint32_t>::iterator itsamples = TIMING_STATS::timing_samples.begin();
		list<string>::iterator itnametags = TIMING_STATS::timing_nametags.begin();

		//search for existing entry
		while(itnametags != TIMING_STATS::timing_nametags.end())
		{
			if(*itnametags == TIMING_STATS::current_nametag)
			{
				//update existing entry
				*itvalues = (*itvalues + time);
				*itsamples += TIMING_STATS::current_samples;
				new_entry = false;
				itnametags = TIMING_STATS::timing_nametags.end();
			}
			else
			{
				itnametags++;
				itvalues++;
				itsamples++;
			}
		}

		//add new entry
		if(new_entry)
		{
			TIMING_STATS::timing_nametags.push_back(TIMING_STATS::current_nametag);
			TIMING_STATS::timing_samples.push_back(TIMING_STATS::current_samples);
			TIMING_STATS::timing_values.push_back(time);
		}
	}
}

void tick_frame()
{
	TIMING_STATS::frame_timer.reset();
}

void tock_frame()
{
	TIMING_STATS::frame_time_samples[TIMING_STATS::frame_time_sample_index] = TIMING_STATS::frame_timer.get_usec();
	TIMING_STATS::frame_time_sample_index = (TIMING_STATS::frame_time_sample_index+1)%TIMING_STATS::frame_time_sample_count;
}

