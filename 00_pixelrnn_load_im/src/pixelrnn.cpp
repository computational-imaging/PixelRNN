// haley so & laurie bose
// 2023

#include <sstream>
#include "scamp5.hpp"
#include <math.h>
#include "IMG_TF.hpp"
#include "AREG_POOLING.hpp"
#include "MISC_FUNCTIONS.hpp"
#include "FUNCS_DIGIT_EXTRACTION.hpp"
#include "FUNCS_WEIGHT_LOADING.hpp"
#include "WEIGHTS_CAMBRIDGE.hpp"
#include "WEIGHTS_LIPS.hpp"
#include "WEIGHTS_MNIST.hpp"
//#include "WEIGHT_FC.hpp"
#include "load_dataset_image.hpp"


//void load_dreg_image(DREG target_dreg,const uint8_t*image_buffer,uint16_t n_rows);
extern int default_convolution_filters[];
uint8_t OUTPUT[4096];
int result[10];
int max_index=0;

const int filters = 16;
const int filter_size = 5;
int input_img_size = 64;
int grid_posx = 256-(1+64/2), grid_posy = 64/2;
int weight_grid_width = 256/64;
int grid_cell_padding = 0;

int frame_t = 0;
int class_i = 0; // which class
int video_i = 0;
bool play = false;

//------------------------HAND GESTURE RECOGNITION-------------------------------
char filepath[256] = "D:/BackUp/Desktop/PixelRNN/cambridge_video_test_set_duplicated_vminvmax_set/";
char filepath_save[256] = "D:/BackUp/Desktop/PixelRNN/test_outputs/";
const char* video_names[9][10] ={
		{ "/Set1_1_0002/", "/Set1_1_0014/", "/Set1_1_0015/", "/Set1_1_0019/", "/Set2_1_0008/", "/Set4_1_0009/", "/Set5_1_0001/", "/Set5_1_0002/", "/Set5_1_0005/", "/Set5_1_0006/"},
		{ "/Set1_2_0004/", "/Set1_2_0006/", "/Set2_2_0014/", "/Set3_2_0005/", "/Set3_2_0008/", "/Set3_2_0009/", "/Set3_2_0012/", "/Set5_2_0007/", "/Set5_2_0011/", "/Set5_2_0019/"},
		{ "/Set1_3_0004/", "/Set2_3_0002/", "/Set2_3_0003/", "/Set2_3_0011/", "/Set2_3_0014/", "/Set3_3_0011/", "/Set3_3_0017/", "/Set4_3_0007/", "/Set4_3_0018/", "/Set5_3_0007/"},
		{ "/Set1_4_0008/", "/Set2_4_0001/", "/Set3_4_0004/", "/Set4_4_0002/", "/Set5_4_0001/", "/Set5_4_0005/", "/Set5_4_0007/", "/Set5_4_0008/", "/Set5_4_0011/", "/Set5_4_0014/"},
		{ "/Set1_5_0010/", "/Set2_5_0000/", "/Set2_5_0017/", "/Set3_5_0001/", "/Set3_5_0005/", "/Set3_5_0007/", "/Set3_5_0019/", "/Set4_5_0003/", "/Set4_5_0010/", "/Set5_5_0000/"},
		{ "/Set1_6_0000/", "/Set2_6_0000/", "/Set2_6_0008/", "/Set3_6_0013/", "/Set4_6_0004/", "/Set4_6_0006/", "/Set5_6_0005/", "/Set5_6_0007/", "/Set5_6_0008/", "/Set5_6_0017/"},
		{ "/Set1_7_0015/", "/Set1_7_0016/", "/Set1_7_0017/", "/Set2_7_0000/", "/Set2_7_0002/", "/Set2_7_0018/", "/Set3_7_0006/", "/Set3_7_0017/", "/Set4_7_0012/", "/Set5_7_0002/"},
		{ "/Set1_8_0003/", "/Set1_8_0009/", "/Set1_8_0017/", "/Set2_8_0002/", "/Set3_8_0002/", "/Set3_8_0003/", "/Set3_8_0017/", "/Set4_8_0006/", "/Set4_8_0019/", "/Set5_8_0000/"},
		{ "/Set1_9_0002/", "/Set1_9_0005/", "/Set1_9_0008/", "/Set2_9_0007/", "/Set2_9_0008/", "/Set2_9_0009/", "/Set3_9_0000/", "/Set3_9_0016/", "/Set3_9_0019/", "/Set4_9_0006/"},
};


char class_i_string[256];
char frame_t_string[256];
std::string tmp_string_conv_save;

void pixelrnn()
{
	vs_init();
	const int display_size = 1;
	auto display00 = vs_gui_add_display("Input Image",0,0,1);
    auto display01 = vs_gui_add_display("Duplicated Input Binary",0,1,1);
    auto display02 = vs_gui_add_display("Conv1 Filter Weights",0,2,1);
    auto display03 = vs_gui_add_display("RNN Filter Weights",0,3,1);

    // First Conv Layer
    auto display11 = vs_gui_add_display("Conv1 output",1,0,1);
    auto display12 = vs_gui_add_display("ReLU",1,1,1);
    auto display13 = vs_gui_add_display("MaxPool",1,2,1);
    auto display14 = vs_gui_add_display("Binarized",1,3,1);

    auto display20 = vs_gui_add_display("Copied input and copied hidden",2,0,1);
	auto display21 = vs_gui_add_display("Gate convs applied",2,1,1);
    auto display22 = vs_gui_add_display("ft and ot",2,2,1);
	auto display23 = vs_gui_add_display("forget gate",2,3,1);

	auto display30 = vs_gui_add_display("Binarized copy on input/hidden",3,0,1);
	auto display31 = vs_gui_add_display("output ot",3,1,1);
	auto display32 = vs_gui_add_display("",3,2,1);
	auto display33 = vs_gui_add_display("",3,3,display_size);


	int img_threshold = 0;
	vs_gui_add_slider("img threshold", -120, 120, -12,&img_threshold);
	int conv1_threshold = 0;
	vs_gui_add_slider("conv1 threshold", -120, 120, 17,&conv1_threshold);
	int rnn_threshold = 0;
	vs_gui_add_slider("rnn threshold", -128, 127, 35,&rnn_threshold );

	int input_white = 5;
	int input_black = 0;
	vs_gui_add_slider("input_white",-120,120,15,&input_white);


	int input_white_rnn = 2;
	vs_gui_add_slider("input rnn white",-10,10,8,&input_white_rnn);


	int maxpool = 4;
	int maxpool_dirx = 0,maxpool_diry = 1;
	int maxpool_blocking = 1;

	auto load_weights_button = vs_gui_add_button("load_weights_button");
	vs_on_gui_update(load_weights_button,[&](int32_t new_value)
	{
		// ConvLayer 1 weights in Register B
		// Temporary registers: R1, R2, R3, F
		load_weights_into_grid_F(filters,filter_size,CONV1_WEIGHTS_5x5,input_img_size,grid_posx, grid_posy, weight_grid_width,grid_cell_padding);
		scamp5_kernel_begin();
			mov(B,F);
		scamp5_kernel_end();

		load_weights_into_grid_F(filters,filter_size,HIDDEN_WEIGHTS_5x5,input_img_size,grid_posx, grid_posy, weight_grid_width,grid_cell_padding);
		scamp5_kernel_begin();
			mov(A,F);
		scamp5_kernel_end();
	});

	bool print_timings = false;
	auto print_timings_bttn = vs_gui_add_button("Print Timings");
	vs_on_gui_update(print_timings_bttn,[&](int32_t new_value)
	{
		print_timings = true;
	});

	auto load_image_button = vs_gui_add_button("load image/ stop");
	vs_on_gui_update(load_image_button,[&](int32_t new_value)
	{
		if (play==true){
			play = false;
		}
		else{
			play = true;
		}

	});

	auto reset_button = vs_gui_add_button("reset");
	vs_on_gui_update(reset_button,[&](int32_t new_value)
	{
		video_i = 0;
		frame_t = 0;
		class_i = 0;
	});
	auto output_switch = vs_gui_add_switch("Save Outputs: ",false);

	vs_stopwatch timer;
	while(1)
	{
		vs_frame_loop_control();
		// Refresh binarized grid inputs
		scamp5_kernel_begin();
			MOV(R5,R10);
			REFRESH(R5);
		scamp5_kernel_end();

		////////////// REFRESH THE WEIGHTS CONV1 in B /////////////////////////
		vs_process_message();
		vs_wait_frame_trigger();

		scamp5_load_in(E,0);
		scamp5_load_in(80);
		scamp5_kernel_begin();
			mov(F,A);
			sub(F,F,IN);
			where(F);
				add(E,E,IN);
				add(E,E,IN);
			all();
			bus(F,A);
			sub(F,F,IN);
			where(F);
				sub(E,E,IN);
				sub(E,E,IN);
			all();
			mov(A,E);
		scamp5_kernel_end();

		scamp5_load_in(E,0);
		scamp5_load_in(80);
		scamp5_kernel_begin();
			mov(F,B);
			sub(F,F,IN);
			where(F);
				add(E,E,IN);
				add(E,E,IN);
			all();
			bus(F,B);
			sub(F,F,IN);
			where(F);
				sub(E,E,IN);
				sub(E,E,IN);
			all();
			mov(B,E);
			REFRESH(R5);
		scamp5_kernel_end();

		if (play){
			// LOAD IN VIDEO IMAGE:
			if (class_i < 10) {
				sprintf(class_i_string, "%c", 48+class_i);
			} else {
				sprintf(class_i_string, "1%c", 38+class_i);
			}

			if (frame_t < 10 ){
				sprintf(frame_t_string, "%c", 48+frame_t);
			} else {
				sprintf(frame_t_string, "1%c", 38+frame_t);
			}

			std::string tmp_string;
			tmp_string = std::string(filepath) + std::string(class_i_string) + std::string(video_names[class_i][video_i]) + std::string(frame_t_string) + std::string(".BMP");
			vs_post_text(tmp_string.c_str());
			vs_post_text("   *\n");

			load_dataset_image(tmp_string.c_str());
			scamp5_kernel_begin();
				mov(F,E);
			scamp5_kernel_end();
			scamp5_output_image(F,display00);

			// binarize F into R10
			scamp5_load_in(E,0);
			scamp5_load_in(img_threshold);
			scamp5_kernel_begin();
				CLR(R11);
//				WHERE(R10);
					sub(E,F,IN);
//				ALL();
				where(E);
//					MOV(R11,FLAG);
					MOV(R10,FLAG);
				all();
			scamp5_kernel_end();


			// print the class it should be in display 33
			std::array<uint8_t, 4> color = {255, 255, 255, 255};
			std::stringstream spost;
			std::string post_string;
			spost << "Should be class " << class_i;
			post_string = spost.str();
			vs_gui_display_text(display32, 190, 127, post_string.c_str(), color);



// *********************************************************************************************************************
// 			This is if we don't load in videos:
// *********************************************************************************************************************
//			////////////// LOAD IMAGE //////////////
//			scamp5_get_image(F,E,vs_gui_read_slider(VS_GUI_FRAME_GAIN));
//			//		scamp5_output_image(F,display00);
//
//			//////////// BLUR and DOWNSAMPLE to 64x64 /////////
//
//			if(vs_gui_read_slider(gauss_display_switch ) == 1)
//			{
//				scamp5_kernel_begin();
//					SET(R1);
//					SET(R2);
//					gauss(F, F, 1);
//				scamp5_kernel_end();
//			}
//			scamp5_output_image(F,display00);
//
//			IMGTF::SCALING::ANALOG::HALF_SCALE(F);
//			IMGTF::SCALING::ANALOG::HALF_SCALE(F);
//			// analog center crop
//			MISC_FUNCTIONS::load_rect_into_DREG(DENUM::R10,96,96,64,64);
//			scamp5_load_in(E,0);
//			scamp5_kernel_begin();
//				WHERE(R10);
//					mov(E,F);
//				ALL();
//				mov(F,E);
//			scamp5_kernel_end();
//
//			// binarize F into R11
//			scamp5_load_in(E,0);
//			scamp5_load_in(img_threshold);
//			scamp5_kernel_begin();
//				CLR(R11);
//				WHERE(R10);
//					sub(E,F,IN);
//				ALL();
//				where(E);
//					MOV(R11,FLAG);
//				all();
//			scamp5_kernel_end();
//
//			MISC_FUNCTIONS::shift_R11(-grid_posx+127,+grid_posy-128); // shifted to top left corner
//
//	//		scamp5_output_image(R11,display30);
//			// F has center 64x64 in analog
//			// R11 has F binarized in top left corner
//	//		scamp5_output_image(F,display10);
//	//		scamp5_output_image(R11,display11);
//
//			//////////// Duplicate out to the 16 registers /////////
//			duplicate_F_into_grid_C();
//			duplicate_R11_into_grid_R10(input_img_size, weight_grid_width,grid_cell_padding);


		//---------------------------------------------
		// duplicated grid binary
		scamp5_output_image(R10,display01);

		// Refresh binarized grid inputs
		scamp5_kernel_begin();
			MOV(R5,R10);
			REFRESH(R5);
		scamp5_kernel_end();



///////////////////////////////////////CONVOLUTION 1//////////////////////////////////
		load_DREG_into_F(DENUM::R5,input_white,input_black);
		scamp5_kernel_begin();
			MOV(R8,R5);
		scamp5_kernel_end();

		scamp5_load_in(0);
		scamp5_kernel_begin();
			mov(C,IN);
		scamp5_kernel_end();
		scamp5_load_in(40);
		scamp5_kernel_begin(); // subtract 40 from the filter weights and put into D
			sub(E,B,IN);
			where(E);
				MOV(R9,FLAG); // where B>0, set R9 to 1, else 0 --- setting areas where filter is 120 (aka 1)
			all();
		scamp5_kernel_end();

		scamp5_load_in(F,input_white);

		// can fine tune placement of filter values with these shifting operations
		int shift2 = 1;
		if(shift2 > 0){
			while(shift2--){
				scamp5_kernel_begin();
					WHERE(R9);			  // FLAG = R9
					DNEWS(R9,FLAG,north); // digital neighbor OR with neighbor
				scamp5_kernel_end();
			}
		}else
		if(shift2 < 0){
			while(shift2++){
				scamp5_kernel_begin();
					WHERE(R9);
					DNEWS(R9,FLAG,south);
				scamp5_kernel_end();
			}
		}

		shift2 = 0;
		if(shift2 > 0){
			while(shift2--){
				scamp5_kernel_begin();
					WHERE(R9);
					DNEWS(R9,FLAG,east);
				scamp5_kernel_end();
			}
		}else
		if(shift2 < 0){
			while(shift2++){
				scamp5_kernel_begin();
					WHERE(R9);
					DNEWS(R9,FLAG,west);
				scamp5_kernel_end();
			}
		}



		timer.reset();

		////////////////////////////////////////////////////////////
		//CREATE PATTERN USED TO CONSTRAIN FLOODING WHEN SPREADING FILTER WEIGHTS
		scamp5_load_pattern(R1,0,0,255,192);
		scamp5_load_pattern(R2,0,0,192,255);
		scamp5_kernel_begin();
			OR(R3,R1,R2);
			NOT(R10,R3);
		scamp5_kernel_end();

		////////////////////////////////////////////////////////////
		// starts iterating from the bottom left corner of filter... then at the end we shift the image to center the output

		scamp5_load_in(C, 0);
		int weight_index = 0;
		for(int y = 0 ; y < filter_size ; y++)
		{
			for(int x = 0 ; x < filter_size ; x++)
			{
				if(weight_index > 25) // convol_step begins as -1
				{
					y = filter_size;
					break;
				}

				scamp5_load_pattern(R11,8-weight_index/5,7-weight_index%5,224,224);
				scamp5_kernel_begin();
					AND(R4,R9,R11);
					MOV(R11,R4);
				scamp5_kernel_end();


				scamp5_flood(R11,R10,0,1);

				// C is the running sum
				scamp5_load_in(F,input_white);
				scamp5_kernel_begin();
					// replaces XOR function XOR(R1, R8, R11)
					ALL();
			    	NOT(R0,R11);
			    	NOR(R1,R8,R0);
			    	NOT(R0,R8);
			    	MOV(R2,R8);
			    	NOR(R2,R11,R0);
			    	OR(R1,R2);
			    	// end XOR

					NOT(R2, R1);	//XNOR
					AND(R3, R10, R2);
					WHERE(R3);
						add(C,C,F);
					ALL();
					AND(R3, R1, R10);
					WHERE(R3);
						sub(C,C,F);
					ALL();
				scamp5_kernel_end();

				// if x is not the edge, we move the image to the right
				if(x != filter_size-1)
				{
					scamp5_kernel_begin();
						WHERE(R8);
						DNEWS(R8,FLAG,east); // AND R8 and the west
						ALL();
					scamp5_kernel_end();
				}
				else // if it is the edge, west west west north
				{
					scamp5_kernel_begin();
						WHERE(R8);
						DNEWS(R8,FLAG,west);
						ALL();
						WHERE(R8);
						DNEWS(R8,FLAG,west);
						ALL();
						WHERE(R8);
						DNEWS(R8,FLAG,west);
						ALL();
						WHERE(R8);
						DNEWS(R8,FLAG,west);
						ALL();
						WHERE(R8);
						DNEWS(R8,FLAG,north);
						ALL();
					scamp5_kernel_end();
				}
				weight_index++;
			}
		}



		if(print_timings)
		{
			vs_post_text("CONVOL LAYER TIME %lu  microseconds\n",timer.get_usec());
		}

		scamp5_output_image(C, display11);

///////////////////////////////////////////////////////////////////////////////////////
		// RELU //
		timer.reset();
		scamp5_load_in(F,0);
		scamp5_kernel_begin();
			where(C);
				mov(F,C);
			all();
			mov(C,F);
		scamp5_kernel_end();
		scamp5_output_image(C, display12);

////////////////////////////////////////////////////////////////////////////////////////
		// MaxPool //
		scamp5_kernel_begin();
				mov(F,C);
		scamp5_kernel_end();
		AREG_POOLING::MAX_POOL_F(maxpool-1,maxpool_dirx == 1, maxpool_diry == 1, maxpool_blocking);
		scamp5_kernel_begin();
				mov(C,F);
		scamp5_kernel_end();
		scamp5_output_image(C, display13);


		int xshift = 1;
		if(xshift > 0){
			while(xshift--){
				scamp5_kernel_begin();
					movx(C,C,west);
				scamp5_kernel_end();
			}
		}else
		if(xshift < 0){
			while(xshift++){
				scamp5_kernel_begin();
					movx(C,C,east);
				scamp5_kernel_end();
			}
		}

		int yshift = -1;
		if(yshift > 0){
			while(yshift--){
				scamp5_kernel_begin();
					movx(C,C,south);
				scamp5_kernel_end();
			}
		}else
		if(yshift < 0){
			while(yshift++){
				scamp5_kernel_begin();
					movx(C,C,north);
				scamp5_kernel_end();
			}
		}

		if(print_timings)
		{
			vs_post_text("RELU MAXPOOL TIME %lu \n",timer.get_usec());
		}

		// Binarize

		scamp5_kernel_begin();
			mov(F,C);
		scamp5_kernel_end();
		scamp5_load_in(conv1_threshold);
		scamp5_kernel_begin();
			CLR(R11);
			sub(F,F,IN);
		scamp5_kernel_end();
		scamp5_load_in(E,-50);
		scamp5_load_in(50);
		scamp5_kernel_begin();
			where(F);
				MOV(R11,FLAG);
				mov(E, IN);
			all();
			MOV(R9,R11);
		scamp5_kernel_end();
		scamp5_output_image(R9, display14);

		// shift image so the kernel and image are "centered"
		// for 5x5 kernel where the first weight is the bottom left corner, we have to
		// shift the resulting image 2 to the right and 2 up
		scamp5_kernel_begin();
			CLR(R1,R3,R2);
			SET(R4); // enables east for DNEWS
			DNEWS0(R11,R9);
			DNEWS0(R9,R11);
			CLR(R4);
			SET(R3); // enables north for DNEWS
			DNEWS0(R11,R9);
			DNEWS0(R9,R11);
			CLR(R3);
		scamp5_kernel_end();


/////////////////////////////////////////////////////////////////////////////////////
		// Downsample the output

		// binarize and then downsample
		IMGTF::SCALING::DIGITAL::HALF_SCALE(R9);
		IMGTF::SCALING::DIGITAL::HALF_SCALE(R9);

		scamp5_kernel_begin();
			REFRESH(R9);
			MOV(R11, R9);
		scamp5_kernel_end();

		MISC_FUNCTIONS::shift_R11(-grid_posx+128+127,+grid_posy-65);
		scamp5_kernel_begin();
			MOV(R8, R11);
			MOV(R11, R9);
		scamp5_kernel_end();
		MISC_FUNCTIONS::shift_R11(-grid_posx+64+127,+grid_posy-65); 	// shifted to quadrant 6
//
//		scamp5_output_image(R9, display21); // this is centered binarized 64x64
//		scamp5_output_image(R11, display20); // this is the shifted to quad 6

/////////////////////////////////////////////////////////////////////////////////////
///////////////////////           RNN Layer               ///////////////////////////
///////////////////////                                   ///////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

		// A has hidden weights HIDDEN_WEIGHTS_5x5
		if(frame_t==0){
			// hidden state in D and R6
			scamp5_load_in(D,80);
			scamp5_kernel_begin();
				SET(R6);
			scamp5_kernel_end();

		}
		else{
			// snap operation to 80 and 0
			// use E and F to help snap
			scamp5_load_in(E,30);
			scamp5_kernel_begin();
				sub(F,D,E);
				CLR(R6);
			scamp5_kernel_end();
			scamp5_load_in(D,-80);
			scamp5_load_in(80);
			scamp5_kernel_begin();
				where(F);
					mov(D, IN);
					MOV(R6, FLAG);
				all();
			scamp5_kernel_end();

		}

		// R11 is the output of the CNN
		// D is the hidden state; R6 is the digital hidden state; hidden state is also binary btw
		// copy hidden into E and copy conv_output into E
		scamp5_load_in(E,-80);
		scamp5_load_in(80);
		MISC_FUNCTIONS::load_rect_into_DREG(DENUM::R10,128,64,64,64); // quadrant 6
		MISC_FUNCTIONS::load_rect_into_DREG(DENUM::R8,  64,64,64,64);   // quadrant 7
		scamp5_kernel_begin();
			AND(R4,R11,R10);
			WHERE(R4);
				mov(E, IN);
			ALL();
			WHERE(R8);
				mov(E, D);
			ALL();
			mov(F,E);
		scamp5_kernel_end();

		// shift F and combine into F
		int shift64 = 64;
		while(shift64--){
			scamp5_kernel_begin();
				movx(F,F,north);
			scamp5_kernel_end();
		}
		MISC_FUNCTIONS::load_rect_into_DREG(DENUM::R10,64,128,128,64);
		scamp5_kernel_begin();
			WHERE(R10);
				mov(E, F);
			ALL();
			mov(F, E);			// recombine into F -- F has the copied input and copied hidden states
		scamp5_kernel_end();


		// binary version/ snapping
		scamp5_load_in(30);
		scamp5_kernel_begin();
			CLR(R7);
			sub(E, F, IN);
			where(E);
				MOV(R7,FLAG);	// R7 has binary version
			all();
		scamp5_kernel_end();

		scamp5_load_in(F, -80);
		scamp5_load_in(80);
		scamp5_kernel_begin();
			WHERE(R7);
				mov(F, IN);
			ALL();
			mov(E,F);
		scamp5_kernel_end();
		scamp5_output_image(F, display20);
		// E and F both have the quad o,h,o,h, binarized
		// R7 has the digital version. R6 is the digital hidden state
///////////////////////////////////////////////////////////////////////

	// GATE CONVS
		// A has weights for rnn
		scamp5_load_in(40);
		scamp5_kernel_begin(); // subtract 40 from the filter weights and put into E
			sub(E,A,IN);
			where(E);
				MOV(R9,FLAG); // where E>0, set R9 to 1, else 0 --- setting areas where filter is 1
			all();
		scamp5_kernel_end();
		// R9 has weights where it is 1

		// can fine tune placement of filter values with these shifting operations
		shift2 = 1;
		if(shift2 > 0){
			while(shift2--){
				scamp5_kernel_begin();
					WHERE(R9);			  // FLAG = R9
					DNEWS(R9,FLAG,north); // digital neighbor OR with neighbor
				scamp5_kernel_end();
			}
		}else
		if(shift2 < 0){
			while(shift2++){
				scamp5_kernel_begin();
					WHERE(R9);
					DNEWS(R9,FLAG,south);
				scamp5_kernel_end();
			}
		}

		shift2 = 0;
		if(shift2 > 0){
			while(shift2--){
				scamp5_kernel_begin();
					WHERE(R9);
					DNEWS(R9,FLAG,east);
				scamp5_kernel_end();
			}
		}else
		if(shift2 < 0){
			while(shift2++){
				scamp5_kernel_begin();
					WHERE(R9);
					DNEWS(R9,FLAG,west);
				scamp5_kernel_end();
			}
		}



		timer.reset();
		// R9 has the weight where weight is 1

		////////////////////////////////////////////////////////////
		//CREATE PATTERN USED TO CONSTRAIN FLOODING WHEN SPREADING FILTER WEIGHTS

		scamp5_load_pattern(R1,0,0,255,192);
		scamp5_load_pattern(R2,0,0,192,255);
		scamp5_kernel_begin();
			OR(R3,R1,R2);
			NOT(R10,R3);
		scamp5_kernel_end();
//
//		scamp5_output_image(R3, display31);  // outline
//		scamp5_output_image(R10, display32); // PE regions
		////////////////////////////////////////////////////////////
		// starts with bottom left corner of filter...then we shift to center it
		scamp5_output_image(R7, display30);
		scamp5_load_in(C,0);
		scamp5_load_in(F,input_white_rnn);

		weight_index = 0;
		for(int y = 0 ; y < filter_size ; y++)
		{
			for(int x = 0 ; x < filter_size ; x++)
			{
				if(weight_index > 25) // convol_step begins as -1
				{
					y = filter_size;
					break;
				}

				scamp5_load_pattern(R11,8-weight_index/5,7-weight_index%5,224,224);

				scamp5_kernel_begin();
					AND(R4,R9,R11);		// R9 is where weight is 1, R11 is the pattern
					MOV(R11,R4);
				scamp5_kernel_end();
				scamp5_flood(R11,R10,0,1);		// R11 is the current weight

				// C is the running sum. add input_white
				scamp5_kernel_begin();
//					XOR(R1, R7, R11); // replace xnor operation. xnor R7 input/hidden with weight R11
					ALL();
			    	NOT(R0,R11);
			    	NOR(R1,R7,R0);
			    	NOT(R0,R7);
			    	MOV(R2,R7);
			    	NOR(R2,R11,R0);
			    	OR(R1,R2);
			    	// end XOR

					NOT(R2, R1);	// XNOR

					AND(R3, R10, R2);
					WHERE(R3);
						add(C,C,F);
					ALL();
					AND(R3, R1, R10);
					WHERE(R3);
						sub(C,C,F);
					ALL();
				scamp5_kernel_end();


				// if x is not the edge, we move the image to the right
				if(x != filter_size-1)
				{
					scamp5_kernel_begin();
						WHERE(R7);
						DNEWS(R7,FLAG,east); // AND R7 and the west
						ALL();
					scamp5_kernel_end();
				}
				else // if it is the edge, west west west north
				{
					//TODO PUT WEIGHTS IN THE RIGHT ORDER RATHER THAN DOING THIS
					scamp5_kernel_begin();
						WHERE(R7);
						DNEWS(R7,FLAG,west);
						ALL();
						WHERE(R7);
						DNEWS(R7,FLAG,west);
						ALL();
						WHERE(R7);
						DNEWS(R7,FLAG,west);
						ALL();
						WHERE(R7);
						DNEWS(R7,FLAG,west);
						ALL();
						WHERE(R7);
						DNEWS(R7,FLAG,north);
						ALL();
					scamp5_kernel_end();
				}
				weight_index++;
			}
		}

//		// shift output up 2 and right 2 to center the output
//		scamp5_kernel_begin();
//			movx(C,C,east);
//			movx(C,C,east);
//			movx(C,C,north);
//			movx(C,C,north);
//		scamp5_kernel_end();

		if(print_timings)
		{
			vs_post_text("GATE CONV LAYER TIME %lu  microseconds\n",timer.get_usec());
			print_timings = false;
		}


		scamp5_load_in(F,-80);
		MISC_FUNCTIONS::load_rect_into_DREG(DENUM::R10,64,64,127,127); // quad 6
		scamp5_kernel_begin();
			WHERE(R10);
				mov(F,C);
			ALL();
			mov(C,F);
		scamp5_kernel_end();
		scamp5_output_image(C, display21);


		///////////////////////////////// add the 2 channels together to complete the gate convs
		MISC_FUNCTIONS::load_rect_into_DREG(DENUM::R10,64,64,64,127);

		// add f results together = ft
		// add o results together = ot

		// copy C into F, shift and combine into E
		shift64 = 64;
		while(shift64--){
			scamp5_kernel_begin();
				movx(F,F,west);
			scamp5_kernel_end();
		}

		scamp5_load_in(E,0);
		scamp5_kernel_begin();
			WHERE(R10);
				add(E,F,C);
			ALL();
		scamp5_kernel_end();
		scamp5_output_image(E,display22);  // should have ft and ot in quad 7 and 11

		MISC_FUNCTIONS::load_rect_into_DREG(DENUM::R8,64,127,64,64);  // quad 11

		scamp5_load_in(F,0);
		scamp5_kernel_begin();
			WHERE(R8);
				mov(F,E);
			ALL();
		scamp5_kernel_end();
		scamp5_output_image(F,display31);  // output


//////////////////////////////////////////////////////////////////////
		// binarize/sign function forget gate
		// R6 is the hidden state

		MISC_FUNCTIONS::load_rect_into_DREG(DENUM::R8,64,64,64,64);  // quad 7

		scamp5_load_in(F,0);
		scamp5_load_in(rnn_threshold); // rnn threshold
		scamp5_kernel_begin();
			WHERE(R8);
				sub(F,E,IN);
			ALL();
			CLR(R11);
			where(F);
				MOV(R11,FLAG);		// R10 is 1s
			all();
			AND(R10, R8, R11);		// just in the square region
			AND(R6, R8, R6);
		scamp5_kernel_end();
		scamp5_output_image(R10, display23);

		// next_h multiplying version
		scamp5_load_in(D,-80);
		scamp5_load_in(80);
        scamp5_kernel_begin();
        	ALL();
//			XOR(R1, R10, R6);		// XOR forget gate and the hidden state (convert from 0 1 to -1 1)
        	NOT(R0,R10);
			NOR(R1,R6,R0);
			NOT(R0,R6);
			MOV(R2,R6);
			NOR(R2,R10,R0);
			OR(R1,R2); 		// end XOR
			NOT(R2, R1);	// XNOR

			AND(R10, R2, R8);
			WHERE(R10);
				mov(D, IN);
				MOV(R6, FLAG);
			ALL();
		scamp5_kernel_end();
		if (vs_gui_read_slider(output_switch ) == 1){
			if (frame_t==15){
				std::string tmp_string_conv_save;
				tmp_string_conv_save = std::string(filepath_save) + std::string(class_i_string) + std::string(video_names[class_i][video_i]) + std::string(frame_t_string) + std::string("_output.BMP");
				vs_gui_save_image(display31, tmp_string_conv_save.c_str() );
				vs_post_text(tmp_string_conv_save.c_str());
			}

		}
//		scamp5_output_image(D, display31);		 // new h


		// Increment file path for next image:
		if (frame_t<15){
			frame_t = frame_t + 1;
		}else{
			frame_t = 0;
			video_i = video_i + 1;

		}
		if (video_i > 9){
			play = false;
			class_i = class_i + 1;
			video_i = 0;
		}
		if (class_i > 8){
			play = false;
			video_i = 0;
			frame_t = 0;
			class_i = 0;
		}

		}


		if(vs_gui_is_on()){
			scamp5_output_image(B,display02); // display conv1 weights
			scamp5_output_image(A,display03); // display gate weights

		}



		vs_loop_counter_inc();
	}
}
