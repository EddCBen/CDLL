# CDLL

A LightWeight C++ Deep Learning Library
## Describtion

A lightweight C++ Deep Learning Library, a starter for an FPGA-based Neural Network training system.

## Dependencies

[NumCpp: A Templatized Header Only C++ Implementation of the Python NumPy Library](https://github.com/dpilger26/NumCpp)

In root directory:

'''
git clone https://github.com/dpilger26/NumCpp.git
'''
## A Quick Start Guide

CDLL constructs sequential Neural Network models using the Module class. Layers (Supporting Dense or fully-connected for now) can be adding with the
.add_layer class method, which takes in a layer class constructer taking in layer_type and its needed attributes like height and width for Dense layers.
CDLL is entirely built on a central Tensor<dtype> class which inherits all of its attributres from the NdArray<dtype> class in NumCpp, and contains extra features which allow for smooth Deep Learning-related algebraic operations. To perform a forward pass Module.forward(Tensor<dtype>) can be used, it outputs a single Tensor<dtype> object populated with results of the sequential Matrix Multiplication operations. To train a Module object, simply feed the Module object and training/validation data and their corresponding labels to the Train function:

'''
void Train(Module & model, Tensor<double> & data, Tensor<double> & labels,\
 Tensor<double> & val_data, Tensor<double> & val_labels, double learning_rate,\
                                             bool Verbose = false)
'''

## Example
 
In [main.cpp](https://github.com/EddCBen/CDLL/blob/main/main.cpp) we create a small Neural Network ArgNet, formed of sequential Dense layers, and train it 
to identify the index of the biggest element in a Tensor<double> input of shape [1,3].

First, we create our Training and Validation Data using 'init::normal_dist(int num_samples, int num_features)'
