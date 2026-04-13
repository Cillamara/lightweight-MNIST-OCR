@author Liam Cui

lightweight-MNIST-Classifier
Logistic regression based Classifier for MNIST dataset

By: Balor Brennan and Liam Cui


This program runs a CUDA based logistic regression classifier on the MNIST database. This program has several customizable features for classification in both training and inferencing. Below are instructions to run the program on the Granger server or any Nvidia based GPU system of your choosing.

--------------------------------- Steps to Run -------------------------------------------------------

--- 1) ssh and load files into granger.cs.rit.edu


--- 2) Set up virtual environment to use pybind11 and numpy (Optional if not ssh'ed into Granger)
	
Run in terminal:
"""
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install pybind11 numpy
"""
		
		
--- 3) Set directory to folder



--- 4) Compile

Run in terminal:
"""
chmod +x compile.sh
./compile.sh
"""
	
		
5) Run the Command Line Interface

Run in terminal:
"""
python 3 python/cli.py
"""


--------------------------------------- Tips -------------------------------

--- CLI: Inputs ---

When user is asked to input a parameter, sometimes the default parameter is given in brackets [like this]

If the user hits enter without inputting anything, the default parameter will be input instead.


--- CLI: Training Models ---

When choosing to train a model based on user input. If the specified model does not exist in the model's save directory, a new model will be trained from 0 weights. However if that specified model already exists, the program will update the weights of the existing model instead.

This is useful for changing the learning rate as you progress.


--- I am very hungry ---

I want so much food right now. It is unreal how hungry I am.







