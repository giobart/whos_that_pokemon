# Who's that Pokemon

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

```
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

```
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

## Neural Network Models
Three Models are supported for face recognition. The first model, uses a small custom Siamese model and trains it using the contrastive loss. This model is mostly used to test our setup. The second model is also a Siamese model but transfer learning is performed on InceptionResnetV1 CNN pre-trained on vggface2 and uses Bineary Cross Entropy loss instead. The third model uses a Bn-Inception CNN pretrained on ImageNet and trains the model using the Group Loss. We also use an extra model to perform liveness detection before the face recognition stage.

### Siamese Network using Binary Cross Entropy and Contrastive Loss
The train_BCE_Contrastive.ipynb notebook is used to train and evaluate both the Binary Cross Entropy and Contrastive Loss. Some flages and variables, in the notebook, can be used to choose which the behaviour required. For example, to re-run the evluation for the Contrastive Loss (current state), the following should be set throughout the network: 
```
cnn_model = CNN_MODEL.InceptionResnetV1
do_train = False
save_checkpoint = False
load_checkpoint = True
```
We use the accuracy metric to evaluate our models. The accuracy is calculated by counting correctly classified images over the incorrect ones and in this case correctly classified means if they are similar or not. For the Contrastive Loss model, in order to know whether two images are similar or not, we compare the output embeddings to each other by computing the L2 norm and if the value is less than a specific threshold then we label them as equal. In order to find the best threshold we run the evaluation several times to get the threshold that achieves the best accuracy. The final accuracy value is achieved by averaging over batches and epochs.

![mycnn_contra](./figures/MyCNN_Contrastive/mycnn_contra.png)

For the Binary Cross Entropy model the accuracy is simpler to compute since the model outputs a probability of how similar the two images are.

![jnception_bce_test](./figures/InceptionResnetv1_BCE/jnception_bce_test.png)

### Group Loss

#### Running the notebook

The Train_Group_Loss.ipynb Notebook is used to train the Group Loss Model. To get the best results, the model was trained on the classification task for 10 epochs before training on the Group Loss which is also the same approach as in the original paper. In addition to that, we tuned the hyper-parameters and used the whole CelebA dataset for training and validation, and LFW for testing.
Similar to the previous model, flags can be used to control the behavior required. For example to evaluate the model on LFW, the following flags throughout the notebook should be set as following:

```python
do_tune = False # we don't want to run hyper-parameter tuning
finetune = False # we don't want to train the model on the classification task.
load_finetune = False # we don't want to load the pre trained model for classification
Load_celeb = False # we don't want to evaluate on the CelebA dataset
do_train = False # we don't want to train the group loss model
load_checkpoint = True # we want to load the pretrained group model checkpoint
do_download = True # to download the dataset
```

In case an older version of Pytorch is available, load_ibm and save_ibm flags can be used to load or save checkpoints across different versions of Pytorch.

Links in the notebook are provided to get all checkpoints used.

#### Algorithm

The Group Loss overcomes the problem of other loss functions such as the contrastive loss and the triplet loss which compare pairs or triplets of images together respectively. That means it is hard to consider all possible combinations. In addition, those loss functions require an extra hyper-parameter (margin) to furthermore separate the embeddings of images corresponding to different persons in the embedding space. On the other hand, the group loss compares all the samples in one batch to each other. It uses a similarity measure as prior information to decide whether to images correspond to the same person or not, and by doing that it learns a clear separation of the embeddings. In other words, the group loss answers the question "given that those two images are x similar to each other, what is the probability of them having the same label?" and it does that for all possible combinations of images in a batch by utilizing the gram matrix.

For the group loss to work, a costume sampler is needed for creating each batch. The sampler chooses n classes with m number of images per class to include in every batch. Choosing n = 24 and m = 2 yields the best results for us.

#### Evaluation



We use the accuracy metric to evaluate the performance of our algorithm 

Below you can find some visualization of our result:

![Group loss visualization on LFW](./figures/Group_loss/group_test_lfw_vis_finetuned_all.png)

### Liveness Detection

The liveness detection is used as an extra check to verify whether the person is real or not. It can detect whether a person's eyes are open or closed and with that we can detect if a person blinks which can be added as a requirement on top of the face recognition system. The liveness.ipynb notebook is used to train and evaluate the model. CFW Dataset was used to train the model but since this dataset doesn't contain a lot of images, a couple of tricks needed to be performed. Since we have already trained such a network with the Group Loss model, we were able to use that network with the same trained weights and apply transfer learning to retrain the last classification layer only which brings as to the first trick. For the second trick, we doubled the numbers of training samples by using image augmentation. The effect of image augmentation can be shown in the following graph comparing the three cases where we increased the size of the training set by 1.0, 1.5, and 2.0 for the orange, red and green curves respectively:



![liveness validation accuracy](./figures/liveness/liveness_val_acc_all.png)

​																											*Orange: x1 augmentation*

​																										    *Red: x1.5 augmentation*

​																								 	   	*Green: x2.0 augmentation*

The liveness.ipynb notebook was used to train the model. Similar to the previous notebooks, it is parametrized by various flags that should be set according the required behavior. For example to run the evaluation, the following flags throughout the network set as following:

```
dataset_gdrive_download(config = config_cfw) # uncomment to download CFW Dataset
do_train = False
save_checkpoint = False
load_checkpoint = True
```

The following are the evaluation results:

![Liveness detection test accuracy](./figures/liveness/liveness_aug_test.png)

![Result Visualization](./figures/liveness/liveness_aug_vis.png)

### Checkpoints

### References
