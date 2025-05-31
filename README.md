# SimpleHarmonicMotionNO
## What it does
Our deep model is designed to predict the position of an oscillating spring across a time interval of 10 seconds. Imagine a spring with some mass attached to it is stretched and released. The spring will oscillate back and forth. Our model will predict the motion of the imaginary oscillating spring by plotting its prediction on a position vs. time graph. After each trial, it will compare its prediction to the true position vs. time graph and calculate the margin of error. Following each iteration, the model will learn from the error and attempt to reduce the error. Eventually, the model will be completed with all of its trials and will display one final prediction which will be the final result. For more information, visit https://github.com/neuraloperator/neuraloperator. The model is based off of the github.
## How we built it
We decided to use Python to code our project since Python has many built in libraries which would be very useful for our AI related project. We used several libraries from PyPI to build our model such as torch, numpy, matplotlib, and sklearn within our project. We first generate around 10,000 data points, 90% of which will be used to train the model, and the remaining 10% will be used to test the model. The model will use the data to train, and after each output, it will compare its own prediction with the actual true result. To learn from its own mistakes, it will go back and self-adjust its parameters so its next prediction would be more accurate and reduce error. After training the model, we will feed it the remaining 10% of the data which is unseen by the model. The model will try to make a prediction based on its training which will be displayed.
## PyPI Libraries
These are the commands to download the necessary PyPI libraries in order to run the code:

pip install pip

pip install numpy

pip install matplotlib

pip install neuraloperator

pip install torch

pip install scikit - learn

pip install scipy
