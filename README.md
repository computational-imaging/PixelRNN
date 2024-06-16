## PixelRNN: In-pixel Recurrent Neural Networks for End-to-end-optimized Perception with Neural Sensors
_Haley So, Laurie Bose, Piotr Dudek, Gordon Wetzstein_<br>
(CVPR 2024)

[[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/So_PixelRNN_In-pixel_Recurrent_Neural_Networks_for_End-to-end-optimized_Perception_with_Neural_CVPR_2024_paper.html)] [[Website](https://www.computationalimaging.org/publications/pixelrnn/)] [[Video](https://youtu.be/grlIwYMcmG0?si=wt59EqZCRwNoYA0E)]

## SIMULATIONS: PixelRNN PyTorch code

### Download Datasets
1. Cambridge Hand Gesture Dataset 
    * Download the dataset here: [link](https://labicvl.github.io/ges_db.htm)
2. Tulips1 Lip Reading Dataset
    * Download the dataset here: [link](https://inc.ucsd.edu/mplab/36/)

Make sure you change the paths in data_utils/utils.py. You can find the paths around line 180 and 240.

### Train:
Specify the config options you would like in the config file. The CNNs weights and activations are trained to be binary. The 1 layer CNN option is `CNN_k5_single_layer_relu_mp_thresh_quantized` and the 2 layer CNN option is 2 layer CNN: `CNN_k5_two_layer_relu_mp_thresh_quantized`.

The RNNs are in model_utils/rnn.py. We include a number of convolutional RNNs we tested out. Our PixelRNN is `rnn_1_qt`. When training with noise for the prototype, you can use the `rnn_1_mul_noise_fixed` model which will add noise during the training to emulate the noise that accumulates in the sensor--processor. 

In addition, you can try different differentiable quantization techniques found in training_utils/quantization.py. We used tanhmx where m=2. You can specify these near the bottom of the config file. For ease, here are a few examples of training the 1CNN+PixelRNN models with and without noise we used to develop our model. 

#### Training the Hand Gesture Model 
Train without noise: <br>
```CUDA_VISIBLE_DEVICES=0 python train.py -c configs/Cambridge/1cnnPixelRNN/cam.json```

Train with noise (needed to combat noise in the prototype): <br>
```CUDA_VISIBLE_DEVICES=0 python train.py -c configs/Cambridge/1cnnPixelRNN/cam_noise.json```

#### Training the Lip Reading Model
Similarly, you can train the lip reading model by changing a few parameters in the config file.<br>
```CUDA_VISIBLE_DEVICES=0 python train.py -c configs/Lips/1cnnPixelRNN/lip_noise.json```

While training, you will be able to see the train, val, and corresponding test accuracy. The model is chosen based on validation. We just show the corresponding performance on the test set for convenience.

#### Evaluate and Print Intermediate values.
You can use the print_intermediates.py and print_intermediates_lip.py scripts. This will print what each of the stages in SCAMP5 should look like. This is useful for debugging the hardware implementation. With the trained model you have, update the path in the test_lips.json config for example. Specify the args such as (-sd) where to save, (-iw) the input white, which we use the value 10 and (-si) for if you want to save images. At the end of the script, you can also see the performance of the model once again and where you saved the images to. 
`CUDA_VISIBLE_DEVICES=6 python print_intermediates.py -c configs/Cambridge/1cnnPixelRNN/test_cam.json -sd /home/haleyso/PixelRNN/intermediates/cam10 -si true -iw 10`

`CUDA_VISIBLE_DEVICES=0 python print_intermediates_lips.py -c configs/Lips/1cnnPixelRNN/test_lips.json -sd /home/haleyso/PixelRNN/intermediates/lip10 -iw 10 -si true`

#### Evaluate on simulated scamp results with Off-sensor Linear Layer
Now with the saved intermediates, you can use these as simulated scamp results. This will just run the off-sensor decoder, which is just a linear layer in our case. If you add noise or have saturation due to the iw value in SCAMP, you can see how this will affect performance for example.
`CUDA_VISIBLE_DEVICES=1 python test_simulated_scamp.py -c configs/Cambridge/1cnnPixelRNN/test_cam.json -sdd /home/haleyso/PixelRNN/intermediates/cam10/ -iw 10`

`CUDA_VISIBLE_DEVICES=1 python test_simulated_scamp.py -c configs/Lips/1cnnPixelRNN/test_lips.json -sdd /home/haleyso/PixelRNN/intermediates/lip10 -iw 10`


### Prepping for On-Sensor implementation:
#### Extracting weights for on-sensor implementation
To extract the weights, use the extract_weights.py and extract_weights_lips.py scripts. In the config file, replace the path2weights with the full path of the model you want to extract the weights for.

`CUDA_VISIBLE_DEVICES=0 python extract_weights.py -c configs/Cambridge/1cnnPixelRNN/test_cam.json -sc true -stf /home/haleyso/PixelRNN/cam_iw10.txt`

* -sc true will format for SCAMP so it will be easy to load into the SCAMP files.
* copy and paste these into the .cpp files in the prototype code. The CNN weights can be inserted in `CONV1_WEIGHTS_5x5` and the pixelrnn weights can be inserted in `HIDDEN_WEIGHTS_5x5`.

Here is another example. Make sure to specify where to save the file to around line 159. The output will also contain the threshold values. Remember to convert to the -127 to 128 regime and also multiply by your iw value.

`CUDA_VISIBLE_DEVICES=1 python extract_weights_lips.py -c configs/Lips/1cnnPixelRNN/test_lips.json -sc true`
    

#### Finetuning:
Once you implement the prototype on SCAMP5, you can send the whole train set and validation set through and finetune the decoder on these outputs. Then, with this finetuned model, you can test on the test set outputs from SCAMP. We provide out finetuned model for SCAMP in pretrained/cam_noise_finetuned.pt. We also provide our outputs from SCAMP in the scamp_outputs folder as an example.

`CUDA_VISIBLE_DEVICES=0 python test_scamp_outputs.py -c configs/Cambridge/1cnnPixelRNN/test_cam_finetuned.json -sd scamp_outputs/cambridge_all_oct2023_r4/ -iw 10`

With noise added to training, our hardware implementation resulted in a test performance of 84.44% for hand gesture recognition and 80% on lip reading.

## PROTOTYPE: SCAMP-5 Vision Sensor
In this work, we prototype with the SCAMP-5 Vision Sensor Platform. SCAMP-5 is paving the way for future in-pixel compute platforms. Each pixel integrates the photodiode along with analog and digital registers, an ALU, and more. Each pixel can also communicate with the four north, south, east, west, neighboring pixels. It's programmable as well, letting us prototype our algorithms. For more information about this platform, check out the [SCAMP website](https://personalpages.manchester.ac.uk/staff/p.dudek/scamp/).<br>

**Prototype Codebase**<br>
Our SCAMP5 code can be found under the folder: 00_pixelrnn_load_im. I built off of Laurie Bose's [CNN on SCAMP5](https://github.com/lauriebose/Scamp5-MNIST_AREG_CNN_example/tree/main) as well as [SCAMP transformations](https://arxiv.org/pdf/2403.16994).

### Pipeline Overview
When using the host app, click `load_the_weights`, turn on `save_outputs` to save the outputs if desired, and click "load image/stop" to start loading in images. Click again to pause. If you want to go back to the beginning of the dataset, aka reset, click `reset`. This will run the PixelRNN and save out the outputs at the end of every gesture. 

### Host App details 
- **img threshold:** threshold input image by value
- **conv1 threshold:** we binarize the signal after the conv. This is the threshold value. This corresponds to the threshold in the model. But don't forget to convert it to the -127 to 128 regime.
- **rnn threshold:** we binarize the hidden state. This is the threshold for that.
- **input_white:** input white for the CNN convolution. 
- **input rnn white:** input white for the rnn convolution. These values should technically be smaller to avoid overflow, but it seems noisy if we make it much smaller.
- **load_weights_button:** load weights of cnn and rnn
- **print timings:** print timings
- **load image/stop:** this loads in the images sequentially. if running, click again to pause.
- **reset:** this takes us back to the first video of the first class
- **save outputs:** saves the outputs if ON

### Weights
The weights are saved in `WEIGHTS_CAMBRIDGE.hpp`. The kernels are 5x5. They are loaded in this order. They are also read in this order during the convolution.

|   |   |   |   |   |
|---|---|---|---|---|
21 | 22 | 23 | 24 | 25 
16 | 17 | 18 | 19 | 20 
11 | 12 | 13 | 14 | 15 
6 | 7 | 8 | 9 | 10 
1 | 2 | 3 | 4 | 5 



To run the off-sensor linear layer, you can just load in the decoder from your model and send the SCAMP outputs through it on your desktop.  I provided a script run_linear_scamp.py to do this.



If you have any questions,  feel free to send me an email (haleyso [at] stanford [dot] edu). 






