#ifndef SCAMP5D_LOAD_IMAGE_SEQUENCE_HPP
#define SCAMP5D_LOAD_IMAGE_SEQUENCE_HPP

#include <cstdio>
#include <scamp5.hpp>

using namespace SCAMP5_PE;

extern int image_index;

void load_dreg_image(DREG target_dreg,const uint8_t*image_buffer,uint16_t n_rows,uint16_t n_cols);
void load_dataset_image(const char*filepath_format);
//void load_dataset_image(const char*filepath_format,int index);
#endif
