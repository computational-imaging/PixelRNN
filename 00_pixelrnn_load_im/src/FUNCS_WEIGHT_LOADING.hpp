//Author: Laurie Bose + Haley So
#include <scamp5.hpp>
#include "MISC_FUNCTIONS.hpp"
#include "IMG_TF.hpp"

// ToDo: done!
void load_weights_into_grid_F(int filters,int filter_size,const void *filter_weights_pointer, int grid_cell_size,int grid_posx, int grid_posy, int grid_size, int grid_padding)
{
	int8_t (*filter_weights)[16][1][filter_size*filter_size] = (int8_t (*)[16][1][filter_size*filter_size]) (filter_weights_pointer);

	scamp5_load_in(F,0);
	scamp5_load_in(120);

	int gridx = 0;
	int gridy = 0;

	// for each of the 16 filters
	// loaded with the first weight in the bottom left corner of each 8x8 block
	for(int n = 0 ; n < filters; n++)
	{
		// Set grid position in the plane
		int offx = grid_posx-gridx*(grid_cell_size+grid_padding);
		int offy = grid_posy+gridy*(grid_cell_size+grid_padding)-1;

		MISC_FUNCTIONS::load_centered_rect_into_DREG(DENUM::R1, offx, offy, 64, 64);

		// For each of the filter weights
		// load in -120, 0, 120 as weights
		for(int i = 0 ; i < filter_size*filter_size; i++)
		{
			scamp5_load_pattern(R2,(offy-i/filter_size),(offx-i%filter_size),248,248);
			scamp5_kernel_begin();
				AND(R3,R1,R2);
			scamp5_kernel_end();

			if((*filter_weights)[n][0][i] == 1)
			{
				scamp5_kernel_begin();
					WHERE(R3);
						add(F,F,IN);
					ALL();
				scamp5_kernel_end();
			}
			else
			{
				if((*filter_weights)[n][0][i] == -1)
				{
					scamp5_kernel_begin();
						WHERE(R3);
							sub(F,F,IN);
						ALL();
					scamp5_kernel_end();
				}
			}
		}

		gridx++;
		if(gridx >= grid_size)
		{
			gridy++;
			gridx = 0;
		}
	}
}

void load_checkerboard_grid_R10(int filters,int filter_size, int grid_cell_size,int grid_posx, int grid_posy, int grid_sizex,int grid_padding)
{
	scamp5_kernel_begin();
		CLR(R10);
	scamp5_kernel_end();

	int gridx = 0;
	int gridy = 0;
	for(int n = 0 ; n < filters; n++)
	{
		int offx = grid_posx-gridx*(grid_cell_size+grid_padding);
		int offy = grid_posy+gridy*(grid_cell_size+grid_padding);

		if((gridy%2 == 0 && gridx%2 == 0) || (gridy%2 == 1 && gridx%2 == 1))
		{
			MISC_FUNCTIONS::load_centered_rect_into_DREG(DENUM::R11, offx,offy, grid_cell_size, grid_cell_size);
			scamp5_kernel_begin();
				OR(R1,R10,R11);
				MOV(R10,R1);
			scamp5_kernel_end();
		}

		gridx++;
		if(gridx == grid_sizex)
		{
			gridx = 0;
			gridy++;
		}
	}
}

void load_FC_weights_into_grid_E(int test, int filters,int neurons,const void *FC_weights_pointer,int switch_xy,int weight_orderx, int weight_ordery, int grid_cell_size,int grid_posx, int grid_posy, int grid_sizex,int grid_padding)
{
	const int input_image_size = 16;
	int total_weights = neurons*filters*input_image_size*input_image_size;

	int8_t (*FC_weights)[total_weights] = (int8_t (*)[total_weights]) (FC_weights_pointer);

	scamp5_kernel_begin();
		CLR(R1);
		CLR(R2);
	scamp5_kernel_end();

	for(int n = 0 ; (test == 0 && n < neurons) || (test > 0 && n < 16); n++)
	{
		int gridx = 0;
		int gridy = 0;
		for(int f = 0 ; f < filters; f++)
		{
			int offx = grid_posx-gridx*(grid_cell_size+grid_padding);
			int offy = grid_posy+gridy*(grid_cell_size+grid_padding);

			int img_pooling = 4;

			int weight_start_index = f*input_image_size*input_image_size+n*4096;
			if(test != 0)
			{
				weight_start_index = f*input_image_size*input_image_size+(test-1)*4096;
			}

			for(int x = 0 ; x < input_image_size ; x++)
			{
				for(int y = 0 ; y < input_image_size ; y++)
				{
					int weight_index = weight_start_index;
					int index_x = weight_orderx ? x : (input_image_size-1-x);
					int index_y = weight_ordery ? y : (input_image_size-1-y);

					if(!switch_xy)
					{
						weight_index+=index_x+index_y*input_image_size;
					}
					else
					{
						weight_index+=index_y+index_x*input_image_size;
					}


					int weight_value = (*FC_weights)[weight_index];

					int pos_x = offx-grid_cell_size/2+x*img_pooling+n%img_pooling;
					int pos_y = offy-grid_cell_size/2+y*img_pooling+(n/img_pooling);

					if(weight_value != 0)
					{
						scamp5_load_point(R11,pos_y,pos_x);
						if(weight_value == 1)
						{
							scamp5_kernel_begin();
								OR(R3,R1,R11);
								MOV(R1,R3);
							scamp5_kernel_end();
						}
						else
						{
							scamp5_kernel_begin();
								OR(R3,R2,R11);
								MOV(R2,R3);
							scamp5_kernel_end();
						}
					}
				}
			}

			gridx++;
			if(gridx == grid_sizex)
			{
				gridx = 0;
				gridy++;
			}
		}
	}

	scamp5_load_in(E,0);
	scamp5_load_in(120);
	scamp5_kernel_begin();
		WHERE(R1);
			add(E,E,IN);
		WHERE(R2);
			sub(E,E,IN);
		ALL();
	scamp5_kernel_end();
}
