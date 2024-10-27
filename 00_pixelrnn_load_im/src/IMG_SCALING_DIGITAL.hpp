#include <scamp5.hpp>
#include <math.h>

#include "IMG_TF_HELPER.hpp"

using namespace SCAMP5_PE;

#ifndef IMG_SCALING_DIGITAL
#define IMG_SCALING_DIGITAL

namespace IMGTF
{
	namespace SCALING
	{
		namespace DIGITAL
		{
			void STEP_SCALE_UP_S6(int step_number);
			void STEP_SCALE_DOWN_S6(int step_number);

			void STEP_SCALE_UPY_S6(int step_number);
			void STEP_SCALE_DOWNY_S6(int step_number);

			void STEP_SCALE_UPX_S6(int step_number);
			void STEP_SCALE_DOWNX_S6(int step_number);

			///////////////////////////////////////////

			int STEP_SCALE(dreg_t reg,int current_scaling_value, bool scale_down);
			void SCALE_Y(dreg_t reg,int scaling_mag, bool scale_down);
			void SCALE_X(dreg_t reg,int scaling_mag, bool scale_down);
			void SCALE(dreg_t reg,int scaling_mag, bool scale_down);

			///////////////////////////////////////////

			void HALF_SCALE(dreg_t reg);
			void QUARTER_SCALE(dreg_t reg);
		}
	}
}
#endif
