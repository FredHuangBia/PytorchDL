# Introduction

PytorchDL is a general deep learning code using pytorch. The code can be easy modified to train new models and use new datasets. 

# Code Structure

There are 5 main subfolders in this repo: 
  - models: Define your neural network model sturcture, trainnner, dataloader(optional) and criterion(optional)
  - datasets: Define and preparse datasets
  - criterions: Define the default loss function
  - utils: Data augmentation functions, visualization functions, resume and save functions
  - sup: An example dataset which can be used to run and test the code

And 2 other python scripts:
  - main.py: The code to run
  - opts.py: Define general training and essential dataset and model parameters

# Run the example
First you need to make sure the python version is **python 3.5**
Then install these libraries: **tqdm, numpy, pytorch**
```sh
$ pip3 install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl 
$ pip3 install torchvision
$ pip3 install tqdm
$ pip3 install numpy
```

If any of the above failed during installation, please try again with 'sudo'.

After cloning this repo, create a directory structure like this: **YourProjectName/code/PytorchDL/** where PytorchDL shuold be the repo.

Next, go to the **'sup'** folder and run the **sliceData.py** to prepare the dataset. This will create a **'data'** folder parallel to the **'code'** folder inside **'YourProjectName'** which stores the training data.
```sh
$ cd YourProjectName/code/PytorchDL/sup
$ python3 sliceData.py
```

Now you are done! Run the **main.py** to start training your neural network!
```sh
$ cd ../
$ python3 main.py
```

You will notice that this code will create a series of folders in parallel with the **'code'** folder. That's why it is better to create the directory structure we just said before. The folders are:
  - gen: Serialized dataset labels, for faster dataloading
  - models: Store the trained models' parameters
  - www: Aseries of html files, for visualization purpose
 
Also, at the begining the code will take some time to serialize all the training data and store them as '.pth' in the 'data' folder. Just be a little bit patient and by doing so, we'll be able to load the data faster while training!