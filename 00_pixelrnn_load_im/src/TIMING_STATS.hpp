//Author: Laurie Bose
//Date: 2021

#include <scamp5.hpp>
#include <list>
#include <algorithm>
#include "REGISTER_ENUMS.hpp"
using namespace SCAMP5_PE;

namespace TIMING_STATS
{
	void print_and_clear(bool clear_text_console);
	extern bool enabled;
}

void tick(std::string nametag,int samples);

void tick(std::string nametag);

void tock();

void tick_frame();

void tock_frame();
