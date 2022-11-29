/*
This file stores basic Nueral Nerwork workloads based on the Tensor class.
*/

#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <iterator>
#include "TensorCore.hpp"
//#include "backprob.hpp"
#include <cmath>
#include <complex>

using namespace std;
using namespace nc;
using namespace TensorCore;

namespace activations{

    auto Tanh(Tensor<double> & in_tens){
                NdArray<double>* tmp = &in_tens;
                return Tensor<double>(nc::tanh(*tmp));
            }

    auto ReLU(Tensor<double> & in_tens){
        for (int i = 0; i < (int)in_tens.shape().rows; i++)
        {
            for (int j = 0; j < (int)in_tens.shape().cols; j++)
            {
                if (in_tens(i,j) < (double)0.0){
                    in_tens(i,j) = (double)0.0;
                }
            }            
        }
        return in_tens;
    }

    auto Sigmoid(Tensor<double> & in_tens){
    /*
        exp(in_tens) / exp(int_tens) + ones(shape[0], shape[1])
    */
        NdArray<double>* tmp = &in_tens;
        return Tensor<double>(divide(exp(*tmp),add(exp(*tmp),\
        ones<double>(Shape((int)in_tens.shape().rows, (int)in_tens.shape().cols)))));
    }
    
    auto Softmax(Tensor<double> & in_tens){
        NdArray<double>* tmp = &in_tens;
        return Tensor<double>(special::softmax(*tmp));
    }
}

namespace LossFunctions{
    /*
    Categorical Cross Entropy:
        agrs: tmp1_true --> Labels, One_Hot_Encoded
              tmp2_pred --> predictions, outputs of a Softmax function (Sum up to 1.0)
        returns : Double value --> Loss
    */
    auto CategoricalCrossEntropy(Tensor<double> tmp1_true, Tensor<double> & tmp2_pred){
        NdArray<double>* y_true = &tmp1_true;
        NdArray<double>* y_pred = &tmp2_pred;
        auto CCE = Tensor<double>((-1.0)*(multiply((*y_true),\
                    log((*y_pred).reshape((*y_true).shape())))).sum());
        return CCE;
    }
}

namespace layers{
    
    class trainable_params{
        public: 
            Tensor<double> weights;
            Tensor<double> grads;
            trainable_params(){}

            void init_weights(const int & x,const int & y){
                weights = init::normal_dist(x,y);
                weights = norm::z_norm(weights);
                grads = Tensor<double>(zeros<double>(Shape(x,y)));
            }
        };
    
    class Dense{
        public:
            trainable_params params;   
            const string type_ = "Dense";        
            int height, width;
            Tensor<double> out_tens;
            Dense(const int & rows,const int & cols){
                height = rows;
                width = cols;
                params.init_weights(height, width);
            }

            Tensor<double> pass(Tensor<double> in_tensor){
                /*static_assert((int)params.weights.shape().rows == (int)in_tensor.shape().cols && \
                                    (int)params.weights.shape().cols == (int)in_tensor.shape().rows,\
                                     "Weights and Input Matrices are of incompatible shapes");
                */
                out_tens = in_tensor.matmul(params.weights);
                return out_tens;
            }
        friend ostream& operator<<(ostream& stream, const Dense& dense);
    };

    ostream& operator<<(ostream& stream, const Dense& dense){
        stream << "Dense" << dense.height << dense.width << endl;
        return stream;
    }
}

using namespace layers;

namespace Core{
    //Class layer is object to multiple inheritance ... Dense, CNN, LSTM, Trans-Enc,Dec

    class layer : public Dense{
        public:
            string act_func = "None";
            Tensor<double> z; //Raw layer output
            Tensor<double> act; //Activation
            Tensor<double> layer_loss; //layer_wise loss.
            
            layer(Dense obj, string act_name) : Dense(obj.height, obj.width){
                    act_func = act_name;
            }  
            layer(Dense obj) : Dense(obj.height, obj.width){}  

            Tensor<double> activate(){
                if(act_func != "None"){
                    if(act_func == "Tanh"){
                        out_tens = activations::Tanh(out_tens);
                    }
                    else if (act_func == "Relu")
                    {
                        out_tens = activations::ReLU(out_tens);
                    }
                    else if (act_func == "Sigmoid")
                    {
                        out_tens = activations::Sigmoid(out_tens);
                    }
                    else if (act_func == "Softmax")
                    {
                        out_tens = activations::Softmax(out_tens);
                    }           
                }
                else
                {
                    out_tens = out_tens;
                }
                
                return out_tens;
            }
        friend ostream& operator<<(ostream& stream, const layer& lay);
        };

    ostream& operator<<(ostream& stream, const layer& lay){
        stream << "layer:\n" << "       " <<lay.type_ << "(" << lay.height << "," \
            << lay.width << ")" << endl;
        return stream;
    }

    class Module{
    public:
        vector<layer> layers;
    private:
        int num_layers = layers.size();
    public:
        Module(){}

        void add_layer(layer lay){
        layers.push_back(lay);
        }
        
        Tensor<double> forward(Tensor<double> in_data){
            Tensor<double> output;
            output = this->layers.front().pass(in_data);
            this->layers.front().z = output;
            output = this->layers.front().activate();
            this->layers.front().act = output;

            for (vector<int>::size_type i = 1; i != this->layers.size(); i++)
            {
                output = this->layers[i].pass(output);
                this->layers[i].z = output;                 
                output = this->layers[i].activate();
                this->layers[i].act =  output;
            }
            return output;
        }        
        
    public:
        layer& operator[](int);
  
    friend ostream& operator<<(ostream& stream, const Module& net);
    };

    ostream& operator<<(ostream& stream, const Module& net){
        
        stream << "Model Layers: " << endl;
        for (vector<int>::size_type i = 0; i != net.layers.size(); i++)
        {
            stream << "     " << net.layers[i].type_ << "(" << net.layers[i].height << "," << \
                    net.layers[i].width << ")\n";
        }
        return stream;
    }
    layer& Module::operator[](int index){
        return layers[index];
    }
}

