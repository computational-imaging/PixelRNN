// haley so & laurie bose
// 2023 pixelrnn

#include "scamp5.hpp"
#include <math.h>
#include "IMG_TF.hpp"
#include <sstream>
using namespace SCAMP5_PE;

volatile bool host_on;
int img_gain_level = 1;
int shiftx = 0;
int shifty = 0;
volatile int gauss_scale = 0;
int half_scale = 0;
uint32_t dreg_tf_time,dreg_translation_time,dreg_scaling_time,dreg_3skewrot_time,dreg_2skewrot_time;
uint32_t areg_tf_time,areg_translation_time,areg_scaling_time,areg_3skewrot_time,areg_2skewrot_time;

char filepatha[256] = "C:/Users/sohal/Desktop/cambridge_video_test_set_pad/";

const char* video_namesa[9][10] ={
{"/Set1_1_0014/", "/Set1_1_0015/", "/Set2_1_0008/", "/Set3_1_0006/", "/Set3_1_0007/", "/Set3_1_0017/", "/Set4_1_0009/", "/Set5_1_0005/", "/Set5_1_0006/", "/Set5_1_0008/"},
{"/Set1_2_0004/", "/Set1_2_0006/", "/Set1_2_0017/", "/Set2_2_0005/", "/Set3_2_0005/", "/Set3_2_0007/", "/Set3_2_0009/", "/Set3_2_0012/", "/Set4_2_0009/", "/Set5_2_0007/"},
{"/Set1_3_0004/", "/Set1_3_0016/", "/Set2_3_0001/", "/Set2_3_0002/", "/Set2_3_0011/", "/Set3_3_0004/", "/Set3_3_0007/", "/Set4_3_0008/", "/Set5_3_0007/", "/Set5_3_0016/"},
{"/Set1_4_0012/", "/Set2_4_0001/", "/Set2_4_0011/", "/Set2_4_0012/", "/Set3_4_0004/", "/Set4_4_0002/", "/Set5_4_0001/", "/Set5_4_0003/", "/Set5_4_0004/", "/Set5_4_0019/"},
{"/Set1_5_0008/", "/Set1_5_0010/", "/Set3_5_0001/", "/Set3_5_0005/", "/Set3_5_0012/", "/Set3_5_0019/", "/Set4_5_0000/", "/Set5_5_0009/", "/Set5_5_0012/", "/Set5_5_0016/"},
{"/Set1_6_0000/", "/Set1_6_0009/", "/Set1_6_0015/", "/Set3_6_0010/", "/Set4_6_0010/", "/Set5_6_0005/", "/Set5_6_0006/", "/Set5_6_0010/", "/Set5_6_0011/", "/Set5_6_0017/"},
{"/Set1_7_0004/", "/Set1_7_0016/", "/Set2_7_0000/", "/Set2_7_0002/", "/Set2_7_0018/", "/Set3_7_0006/", "/Set3_7_0007/", "/Set3_7_0009/", "/Set3_7_0016/", "/Set3_7_0017/"},
{"/Set1_8_0001/", "/Set1_8_0003/", "/Set1_8_0012/", "/Set2_8_0002/", "/Set3_8_0002/", "/Set3_8_0003/", "/Set3_8_0013/", "/Set4_8_0019/", "/Set5_8_0013/", "/Set5_8_0018/"},
{"/Set1_9_0002/", "/Set1_9_0005/", "/Set1_9_0008/", "/Set2_9_0007/", "/Set2_9_0014/", "/Set3_9_0019/", "/Set4_9_0006/", "/Set4_9_0010/", "/Set5_9_0005/", "/Set5_9_0006/"}};

int class_ia = 0;
int video_ia = 2;
int frame_ta = 2;
void pixelrnn();

void scamp5_main(){

    // Update Frame Rate
	// vs_frame_trigger_set(1,16);
	vs_on_gui_update(VS_GUI_FRAME_RATE,[&](int32_t new_value){
	        uint32_t framerate = 16;
	        if(framerate > 0){
	            vs_frame_trigger_set(1,framerate);
	            vs_enable_frame_trigger();
	            vs_post_text("frame trigger: 1/%d\n",(int)framerate);
	        }else{
	            vs_disable_frame_trigger();
	            vs_post_text("frame trigger disabled\n");
	        }
    });
    vs_frame_trigger_set(1,16);

    // Update Frame Gain
	vs_on_gui_update(VS_GUI_FRAME_GAIN,[&](int32_t new_value){
		 img_gain_level = new_value;
	});

	// On Connect
    vs_on_host_connect([&](){
    	vs_post_text("PixelRNN Load in Videos \n");
    	vs_post_text("loop_counter: %d\n",(int)vs_loop_counter_get());
        scamp5_kernel::print_debug_info();
        vs_led_on(VS_LED_2);
    });
    vs_on_host_disconnect([&](){
        vs_led_off(VS_LED_2);
    });

// Voltage powering it...
//	auto slider_vxc = vs_gui_add_slider("vxc: ",0,4095,2900);
//	vs_on_gui_update(slider_vxc,[&](int32_t new_value){
//		vs_scamp5_configure_voltage(3,new_value);
//	});

    vs_stopwatch timer;
	pixelrnn();

}


int main(){
	// initialize M0 system
	vs_init();

	// make default output to be USB
	vs_post_bind_io_agent(vs_usb);

	scamp5_bind_io_agent(vs_usb);

    vs_on_shutdown([&](){
    	vs_post_text("M0 shutdown\n");
    });

    // run the vision algorithm
	scamp5_main();
	return 0;

}




