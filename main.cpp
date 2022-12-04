#include <iostream>
#include <vector>
#include "include/CDLL.hpp"

using namespace std;

int main(){
  
  int train_val = 2000;
  int test = 300;
  Tensor<double> data = init::normal_dist(train_val,3).mul_val(5.0);
  Tensor<double> labels = Tensor<double>(zeros<double>(Shape(train_val,3)));
  Tensor<double> val_data = norm::z_norm(init::normal_dist(train_val,3));
  Tensor<double> val_labels = Tensor<double>(zeros<double>(Shape(train_val,3)));
  Tensor<double> test_data = norm::z_norm(init::normal_dist(test,3));
  Tensor<double> test_labels = Tensor<double>(zeros<double>(Shape(test,3)));
  double learning_rate = 0.0035;
  int EPOCHS = 15;

  for (int i = 0; i < labels.shape().rows; i++)
    {
      labels(i, data.Row(i).argmax().item()) = 1.0;
    }
  
  for (int i = 0; i < val_labels.shape().rows; i++)
    {
      val_labels(i, val_data.Row(i).argmax().item()) = 1.0;
    }

  Module ArgNet;
  ArgNet.add_layer(layer(Dense(3,10), "Tanh", 0.7));
  ArgNet.add_layer(layer(Dense(10,10), "Tanh", 0.7));
  ArgNet.add_layer(layer(Dense(10,10), "Tanh", 0.9));  
  ArgNet.add_layer(layer(Dense(10,5), "Tanh", 0.9));
  ArgNet.add_layer(layer(Dense(5,3), "Softmax", 1.0));

  for (int epoch = 0; epoch < EPOCHS; epoch++)
  { 
    cout << "Epoch : " << epoch << endl;
    Train(ArgNet, data, labels, val_data, val_labels, learning_rate);
  }
  
  cout << "----------------------------" << endl;
  cout << "-----------TEST-SET---------" << endl;
  cout << "----------------------------" << endl;

  for (int i = 0; i < test_data.shape().rows; i++)
  {
    cout << test_data.Row(i) << endl;
    test_labels(i, ArgNet.forward(test_data.Row(i)).argmax().item()) = 1.0;

    cout << test_labels.Row(i) << endl;
    cout << "----------------------------" << endl;

  }
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
