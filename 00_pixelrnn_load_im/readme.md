# Pixelrnn SCAMP5 Code

## Setup:
### Dataset
Access the test set to upload into SCAMP5 at [gdrive link](https://drive.google.com/drive/folders/1QRdkHLVSc2mXBeM0uxlsRQLQKjTXXy9x?usp=sharing).  
This is the test dataset for hand gesture. There are 10 examples of each of the 9 gestures. Each is 16 frames long. In the scamp5 code, replace the path in `pixelrnn.cpp` line 38 with your path. Line 36 is where the outputs will be saved. They will be saved in the same folder structure. For ease,  I included the empty folders here (`test_output_empty_folder`). Replace path in line 39 if needed.

For the test videos, I saved out what the images _should_ look like in the folder called `test_simulated_intermediates`. The path is as so: `class/gesture_name/frame_i/image` where image is different outputs along the PixelRNN to compare intermediate outputs. Note that the input white for the conv is 5, and for the rnn, it is 2. This is due to the 255 limit of levels available before overflow. 

## Run PixelRNN:
### Pipeline Overview
When using the host app, click `load_the_weights`, turn on `save_outputs` to save the outputs if desired, and click "load image/stop" to start loading in images. Click again to pause. If you want to go back to the beginning of the dataset, aka reset, click `reset`. This will run the PixelRNN and save out the outputs at the end of every gesture. Then, you can crop the outputs and run the fully connected layer off-sensor. Ideally, we'd run it all on SCAMP. The fully connected weights are not binary though, so microcontroller is maybe the best option. I included the pytorch code to run the fully connected layer. See more later.

### Host App details 
- **img threshold:** threshold input image by value
- **conv1 threshold:** we binarize the signal after the conv. This is the threshold value
- **rnn threshold:** we binarize the hidden state. This is the threshold for that.
- **input_white:** input white for the CNN convolution.
- **input rnn white:** input white for the rnn convolution. These values should technically be smaller to avoid overflow, but it seems noisy if we make it much smaller.
- **load_weights_button:** load weights of cnn and rnn
- **print timings:** print timings
- **load image/stop:** this loads in the images sequentially. if running, click again to pause.
- **reset:** this takes us back to the first video of the first class
- **save outputs:** saves the outputs if ON

### Fully Connected Layer
`[To Do]`
The fully connected layer is currently run off-sensor. Ideally, we can have it on the microcontroller to make a good demo. It would be a simple matrix operation. See pytorch code to run linear layer only.

## Weights
The weights are saved in `WEIGHTS_CAMBRIDGE.hpp`. I've also included the text files where I extracted the weights. The kernels are 5x5. They are loaded in this order. They are also read in this order during the convolution.

|   |   |   |   |   |
|---|---|---|---|---|
21 | 22 | 23 | 24 | 25 
16 | 17 | 18 | 19 | 20 
11 | 12 | 13 | 14 | 15 
6 | 7 | 8 | 9 | 10 
1 | 2 | 3 | 4 | 5 
