# ML
Machine learning algorithms

Currently there are three main networks in this work. One is LSTM (Long short term memory), two SOM (self-organizing map) and three Convolutional Neural Network (CNN). The general idea is to develop neural networks in the mathematical language of tensor and escape from the limitations of linear algebra and matrix calculus, in hope of allowing ML algorithms to handle more complex data.

The SOM is somewhat more complete, whereas LSTM lucks for now the training part. Also in SOM the basic building blocks of the Convolutional Neural Network (CNN) can be found.

Also an Elman Recurrent network is included, purely for demonstration purposes, as to how RNNs flow and how real time on-line training might be conducted.

In the future the algorithms will be changed to run on GPUs as well. The structure of all algorithms is such that every parameter can be passed as a python dictionary with strings and numbers. This is so that the networks can be used under an evolutionary algorithm and every parameter of the network, from hyper-parameters (like number of neurons, learning rate etc.) to which activation functions are used (even the activation functions are allowed to have parameters) will be used as the genes of the evolutionary algorithm.
