# Kaggle-dog-breeds-classification

![border_collies](https://user-images.githubusercontent.com/25509152/33898274-fa96ef88-df78-11e7-8f4c-59105584fdce.png)

Scripts in this branch may not be stable.


"small_dataset" is a folder for kaggle dataset from https://www.kaggle.com/c/dog-breed-identification/data

"data" is a well seperated dataset (including labels information) for images from stanford dataset http://vision.stanford.edu/aditya86/ImageNetDogs/


This documents describes how to run the code.

In this project Keras library with Tensorflow Backend is used. Thus, in order to run the code successfully, first, we have to install dependencies.

### Preprint version of the paper is published and you can cite like the below:

#### Ayanzadeh, A.; Vahidnia, S.. Modified Deep Neural Networks for Dog Breeds Identification. Preprints 2018, 2018120232 (doi: 10.20944/preprints201812.0232.v1).

1 - installing dependencies on linux or Mac OS:
	* Anaconda contains most of the library we need to run over code. Please go to the following link and install anaconda which is described based on your Operating system:
		https://www.continuum.io/downloads
	
	* install keras on Ubuntu OS or Mac OS is streightforward:
		1- Press Shift+Alt+T to open the terminal and enter the following command:
			sudo pip install keras
	* Please follow the instruction on the link below for installing dependencies on windows:
			https://goo.gl/GrCZl2

	* Please follow the instruction on the link below to install Tensorflow:

			https://www.tensorflow.org/get_started/os_setup


2- If you would like to run the code on GPU you should follow the steps on the link below to install CUDA 8.0 on your operation system:

			https://developer.nvidia.com/cuda-downloads

3- After installing all the dependencies. Please download the images of dataset from the link below, unzip the downloaded file and copy the images folder in the folder containing this REDME file.


				 tensorboard --logdir=logs


