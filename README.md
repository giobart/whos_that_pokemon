# Who's that Pokemon

## Environment

This project provides a script that setup the environment of the main repository automatically. <br><br>
To create the virtual environment simply run:<br>
```
python env_builder.py
```
this script will create a virtualenv inside `virt/` folder. (This operation may take few minutes) <br><br>
To activate the virtual environment use:<br>
```
. virt/bin/activate
```
To deactivate the virtual environment use:<br>
```
deactivate
```

## Requirements

If you don't use the automatic generated virtual environment or if you're running the code inside the submodules 
you need to install the requirements running:
```
pip install -r requirements.txt
```

## Dataset data module
The Data Module takes in input a dataset and generates training,validation and test set. 
Currently the Data Module uses an image transformation that aligns the faces using an affine transformation. 
This can be changed inside `lfw_lightning_data_module` changing the default `FaceAlignTransform(FaceAlignTransform.ROTATION)` to `FaceAlignTransform(FaceAlignTransform.AFFINE)` in order to have the experimental affine transformation. 

## Image registration service
The image registration service belongs to his own module under image-registration-service and is used to store the images to a Mongo DB.
Before starting the service is mandatory to configure the credentials in the `credentials.py` file. The credentials for the database are shared in private among the team members. 
Is possible to start up the service typing `python entry.py` inside the module folder.

This service exposes a very simple REST API with 3 methods:

- POST http://127.0.0.1:5000/api/store/<int:employee_id>
	- take as input a JSON of the image data as following:
```
{
	"name":"giovanni",
	"surname":"bartolomeo",
	"img_features":[[float],[float]], 
	"img_base64":"base64 encoded image"
}
``` 
if the image features are already available is possible to populate the **img_features** int vector, otherwise sending the base64 encoded image, through the **img_base64** string field, the serivce will extract the features on his own and store them on the db. <br>

- POST http://127.0.0.1:5000/api/find_match
	- take a picture as input and return a matching user if any. Is possible to toggle the image cropping system (crops and align the image to the center) and the fraud tedection system.
	- if liveness is true, frames contains 15 base64 pictures used for the liveness detection test
```
{
	"img_crop":bool,
    	"fraud_detection":bool,
	"liveness":bool,
	"img_base64":"base64 encoded image",
	"frames":[string]
}
``` 

- GET http://127.0.0.1:5000/api/get_all/<page_size>/<page_number>
	- If for example we want to get the first batch of 10 elements from the image list we call `api/get_all/10/1`
	- For the second batch of 10 elements we type `api/get_all/10/2`
- DELETE http://127.0.0.1:5000/api/<employee_id> to delete an entry from the db using the id
 
## Running the UI
Move inside the ` face-detection-UI ` folder and run the UI with `python entry.py`. This will start the web server at localhost:5006. <br>
In order to upload a new image to the database you must also run the `ìmage-registration-service` on the default 5005 port.  <br><br>
Disclaimer: Don't trust the username & pass login, it is just a demonstrative login with default username "admin" and password "admin". Nothing more than a graphical feature to simulate a system administrator that adds new people to the database.

