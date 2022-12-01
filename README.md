# CDLL

A LightWeight C++ Deep Learning Library
## Describtion

A lightweight C++ Deep Learning Library, a starter for an FPGA-based Neural Network training system.

## Dependencies

[NumCpp: A Templatized Header Only C++ Implementation of the Python NumPy Library](https://github.com/dpilger26/NumCpp)

In root directory:

```
git clone https://github.com/dpilger26/NumCpp.git
```
## A Quick Start Guide

CDLL constructs sequential Neural Network models using the Module class. Layers (Supporting Dense or fully-connected for now) can be adding with the
`.add_layer` class method, which takes in a layer class constructer taking in layer_type and its needed attributes like height and width for Dense layers, and the activation function intended for that specific layer, like `Tanh` or `ReLU`.
CDLL is entirely built on a central `Tensor<dtype>` class which inherits all of its attributres from the `NdArray<dtype>` class in NumCpp, and contains extra features which allow for smooth Deep Learning-related algebraic operations. To perform a forward pass `Module.forward(Tensor<dtype>)` can be used, it outputs a single `Tensor<dtype>` object populated with results of the sequential Matrix Multiplication operations. To train a Module object, simply feed the Module object and training/validation data and their corresponding labels to the Train function:


```
void Train(Module & model, Tensor<double> & data, Tensor<double> & labels,\
 Tensor<double> & val_data, Tensor<double> & val_labels, double learning_rate, bool Verbose = false)
```
 

## Example
 
In [main.cpp](https://github.com/EddCBen/CDLL/blob/main/main.cpp) we create a small Neural Network ArgNet, formed of sequential Dense layers, and train it 
to identify the index of the biggest element in a Tensor<double> input of shape [1,3].
First, we create our Training and Validation Data using `init::normal_dist(int num_samples, int num_features)`. Lables can be geenrated with a for loop for both splits:
```
 for (int i = 0; i < labels.shape().rows; i++)
    {
      labels(i, data.Row(i).argmax().item()) = 1.0;
    }
  
  for (int i = 0; i < val_labels.shape().rows; i++)
    {
      val_labels(i, val_data.Row(i).argmax().item()) = 1.0;
    }
 ```
Next step is conmstructing our Module object `ArgNet` and adding three Dense layers with specific dimensions:
```
 Module ArgNet;
  ArgNet.add_layer(layer(Dense(3,10), "Tanh"));
  ArgNet.add_layer(layer(Dense(10,5), "Tanh"));
  ArgNet.add_layer(layer(Dense(5,3), "Softmax"));
 ``` 
 Then for an `int EPOCHS` we can perform Gradient-Descent optimization on our ArgNet model (backpropagation and nn::Module are contained in [include/backprob.hpp](https://github.com/EddCBen/CDLL/blob/main/include/backprob.hpp) and [include/nn.hpp](https://github.com/EddCBen/CDLL/tree/main/nn.hpp) respectively). 
 ```
 ### Training and Validation
 for (int epoch = 0; epoch < EPOCHS; epoch++)
  { 
    cout << "Epoch : " << epoch << endl;
    Train(ArgNet, data, labels, val_data, val_labels, learning_rate);
  }
 ```
### Testing
Testing can be simply performed by using the `.forward(Tensor<dtype> tens)` on the trained Module object ArgNet.
 
```
 Tensor<double> test_labels = Tensor<double>(zeros<double>(Shape(300,3)));
  for (int i = 0; i < test_data.shape().rows; i++)
  {
    cout << test_data.Row(i) << endl;
    test_labels(i, ArgNet.forward(test_data.Row(i)).argmax().item()) = 1.0;

    cout << test_labels.Row(i) << endl;
    cout << "----------------------------" << endl;
  }
```
### Output and Test Accuracy
                                                  
To test the trained ArgNet model, we pass a Test set of randomly initialized numbers to it, and record the outputs, and the number of true positives
`int tps` in order to calculate the Test Accuracy at the end.
```
double tps = 0;
  for (int i = 0; i < test_labels.shape().rows; i++)
  {
    if (test_labels.Row(i)(test_data.Row(i).argmax().item()) == 1.0)
    {
      ++tps;
    }
  }
    cout << "Test Accuracy is : " << (double)(tps / test_labels.shape().rows) *\
     100 << "%" << endl;
  }                                                  
```
This Loop will produce pairs of [1,3] shaped `Tensor<double>` tensors, representing the input and predicted output for the index of its maximum value.
For `int EPOCHS = 5;`, the Training and Validation losses are recorded as follows:
```
 Epoch : 0
Training Loss : 0.697526
Validation Loss : 0.720253
Epoch : 1
Training Loss : 0.218101
Validation Loss : 0.263989
Epoch : 2
Training Loss : 0.135952
Validation Loss : 0.21052
Epoch : 3
Training Loss : 0.103221
Validation Loss : 0.16885
Epoch : 4
Training Loss : 0.0908546
Validation Loss : 0.148172
 ```
Note: passing `Verbose = true` in the Train function will output realtime Loss values for both splits.
The trained ArgNet Neural Network achieves an accuracy of `93.6667%` on the test set. 
Predicted output VS. Raw input logs:
```
 [[0.425696, -0.644478, -0.135925, ]]

[[1, 0, 0, ]]

----------------------------
[[1.39228, -1.23217, 2.15055, ]]

[[0, 0, 1, ]]

----------------------------
[[1.68678, -0.868659, -1.24229, ]]

[[1, 0, 0, ]]
.
.
. 
Test Accuracy is : 93.6667%
```
