//Author: Laurie Bose
//Date: 2021

#include <scamp5.hpp>
#include <math.h>
#include "REGISTER_ENUMS.hpp"
#include "MISC_FUNCTIONS.hpp"

using namespace SCAMP5_PE;

#ifndef AREG_POOLING
#define AREG_POOLING

void MAX_POOL_F(int iterations,bool maxpool_dirx, bool maxpool_diry,bool blocking);  //DESTROYS CONTENT IN R0

#endif
