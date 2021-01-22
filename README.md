# Who's that Pokemon

This project wants to explore the power of Deep Learning in order to create models for face recognition and liveness detection that can be used for tasks like a website log in. 
For this purpose, we propose a webapp that allows a system administrator to register new users using directly their face and gives the ability to log in using face recognition. 

![Project demo gif](static/liveness-correct.gif)

## Environment

This project provides a script that setup the environment of the main repository automatically. <br><br>
To create the virtual environment simply run:

```
python env_builder.py
```

this script will create a virtualenv inside `virt/` folder. (This operation may take few minutes) <br><br>
To activate the virtual environment use:

```
. virt/bin/activate
```

To deactivate the virtual environment use:<br>

```
deactivate
```

## Requirements

If you don't use the automatic generated virtual environment or if you're running the code inside the submodules 
you need to install the requirements manually running:

```
pip install -r requirements.txt
```

## Project Structure

 This project is divided into 2 main blocks
 
####  **Research block:** <br>
A set of jupyter notebook and python modules that are the building block for the research made in this project. All of this belongs to this main repository. Here you'll find:

```bash
.
├── data_visualization.ipynb: 
│			Description:	
│				This notebook has been used to explore the 
│				LFW dataset and visualize some images
├── Train_BCE_Contrastive.ipynb 
│			Description:	
│				[Add description here]
├── Train_Group_Loss.ipynb 
│			Description:	
│				[Add description here]
├── evaluate.ipynb 
│			Description:	
│				[Add description here]
├── liveness.ipynb 
│			Description:	
│				[Add description here]
└── src/ 
	 │		Description:	
	 │			Folder containing all the python code 
	 │			used by the notebooks
	 ├── evaluation/
	 │		Description:	
	 │			this folder contains all the functions 
	 │			used for the evaluation of the models
	 ├── hyper_tune/
	 │		Description:	
	 │			[Add description here]
	 ├── model/
	 │		Description:	
	 │			To this folder belongs all the
	 │			classes representing a NN or a piece of it
	 ├── modules/
	 │		Description:	
	 │			This folder contains all the 
	 │			LightningDataModule extensions used for
	 │			the training, validation and testing
	 └── tools/
	 		Description:	
	 			A set of tools used for 
	 				- data augmentation
	 				- dataset download and preparation
	 				- evaluation
	 				- image processing
	 				- model related operations

```

####  **Application block:**
The application block consists of all the microservices that compose the final web application accordingly to the following scheme
![cloud services scheme](static/cloud-services.png)

These services are part of the following submodules:

```bash
.
├── face-detection-UI 
├── image-registration-service
└── liveness_detection_service

```

In order to use the web application is important to run them all and the access the UI with a browser

* **Local Deploy:** is possible to run these services locally using the `entry.py` script inside them as described in the apposite README file inside the submodule repository. A dockerfile is also provided but further configuration for the networking are necessary. 
* **Deploy with Openshift** Each submodule comes together with the documentation to configure the deployment on Openshift like the following one

![Openshift Deploy Scheme](static/openshift-deploy.png)

Further informations about these services are provided inside the README in each submodule.



## Datasets 
The datasets used to train, evaluate and test the models are 3:

* CelebA
* CFW
* LFW

They are automatically downloaded through the scripts provided inside `src/tools/dataset_tools.py`, and the download link provided respectively inside `.config_celeba.py`, `.config_cfw.py` and `.config_lfw.py`.

The dataloaders for these datasets can be found inside `src/modules` as Pytorch Lightning data modules. 


## Image Transformation
The images used for the training of the models are all quite uniform thanks to the amzing work made from the creators of these datasets. Anyway when dealing with images taken from the real world we should be very lucky to get images of the same size as the ones from the dataset and also with the same prospective of the subject. In particular, the model always expects an image with fixed width W and heigth H where W==H and, the face of the subject positioned exactly in the center, with such a rotation that positions the line that connect the eyes exatly parallel to the ground. 
When feeding images from a real world scenario we need a transformation that normalize the size of the picture and positions the subject in a way similar to the one proposed for the used datasets. 
 
The **FaceAlignTransform** located inside `src/tools/image_preprocessing.py` proposes 2 possible kind of transformation:

* Crop and Rotation
* Simple

The former is the slowest but obtain a result very similar to the images provied by LWD Deep Funneled Datased, the latter is sligthly faster but performs only a crop, assuming that the image is already with th correct rotation. 

### Crop and Rotation Transform

Transform enforced initializing the FaceAlignTransform with the following parameter 

```
transform = FaceAlignTransform(shape=255, kind=FaceAlignTransform.ROTATION)
```

Under the hood this transform performs the following steps:

1. Using MTCNN the algorithm searches for the landmark points and bounding boxes of the face in the picture. If more than one face are detected it takes only into consideration the bounding box with the highest level of confidence.
2. Expand with a factor of 1/3 the bounding box proposed by MTCNN.
3. Create a new black image of size `model_input_shape x model_input_shape`.
4. Fit the extracted face from the expanded bounding box inside the new black picture.
5. Use the landmarks detected by MTCNN to calculate the angle of rotation between the eyes and the X axis and the resulting rotation matrix.
6. Apply the Rotation transformation

### Simple transform

Transform enforced initializing the FaceAlignTransform with the following parameter 

```
transform = FaceAlignTransform(shape=255, kind=FaceAlignTransform.SIMPLE)
```

Very similar to the previous transformation except that:

* Avoid the computation of the landmarks point with MTCNN
* Skip step 5 and 6 

### Example
On the left side the original image captured from a webcam, on the right side the image after that the crop and rotation transform have been applied.

![transform example](static/transform_comparison.jpg)