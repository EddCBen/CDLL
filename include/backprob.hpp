/*
    This is an implemtation of AutoDifferentiation in Reverse (Adjoint) Mode
    for obtaining gradients for Neural Network parameters, and applying
    backprob (Gradient-based optimization)
*/
/* 
    To do:
        - Construction of Computation Graph With initialization of Module obj.
        - Implement Dynamic Reverse-mode autodifferentiation. 
*/

#pragma once

#include "TensorCore.hpp"
#include "nn.hpp"
#include <vector>
#include <iostream>

using namespace TensorCore;
using namespace Core;
using namespace activations;
using namespace layers;
using namespace std;

namespace diff_act_funcs{

    Tensor<double> Tanh_d(Tensor<double> in_tens){
        return Tensor<double>(ones<double>(Shape((uint32)in_tens.shape().rows,\
        (uint32)in_tens.shape().cols))).sub(Tanh(in_tens).element_mul(Tanh(in_tens)));
    }
     double Tanh_d_val(double x){
        return 1.0 - (nc::tanh(x) * nc::tanh(x));
    }


    Tensor<double> ReLU_d(Tensor<double> in_tens){
        for (int i = 0; i < (int)in_tens.shape().rows; i++)
        {
            for (int j = 0; j < (int)in_tens.shape().cols; j++)
            {
                if (in_tens(i,j) <= (double)0.0){
                    in_tens(i,j) = (double)0.0;
                }
                else if (in_tens(i,j) > 0)
                {
                    in_tens(i,j) = (double)(1.0);
                }
            }            
        }
        return in_tens;
    }

    Tensor<double> Sigmoid_d(Tensor<double> & in_tens){
        Tensor<double> sig = Sigmoid(in_tens);
        NdArray<double>* tmp = &sig;
        return Tensor<double>(multiply((*tmp), subtract(ones<double>((uint32)((*tmp).shape().rows,\
            (uint32)((*tmp).shape().cols))),(*tmp))));
    }

    /*
        Dloss/Du_i = y_i - a_i
            Where u(z) = z * W 
        Softmax_backprob: Takes in Tensor<double> C, Tensor<double> softmax Output a.
                          returns: Tensor<double> dC/Du  
    */

    int layer_num(Module model){
        int num_layers = 0;
        for(layer lay: model.layers){
            num_layers++;
        }
        return num_layers;
    }

}



using namespace diff_act_funcs;

namespace backprob{
    
    void grad_bp(Module & model, Tensor<double> input_, Tensor<double> label){
        model.layers.back().layer_loss = label.sub(model.layers.back().act);
        int nbr = layer_num(model);
        for (int i = 1; i < layer_num(model) + 1; i++)
        {
        if(i == 1){
            for (int j = 0; j < model[nbr-i].params.grads.shape().rows; j++)
            {
                for (int k = 0; k < model[nbr-i].params.grads.shape().cols; k++)
                {
                    model[nbr-i].params.grads(j,k) = model[nbr-i-1].act.Trans()(j) * model[nbr - i].layer_loss(k); 
                }
            }
        }
        else if (nbr - i > 0)
        {      
            model[nbr - i].layer_loss = (model[nbr-i+1].layer_loss.matmul(model[nbr-i+1].params.weights.Trans())).\
                element_mul(Tanh_d(model[nbr-i].z));
            //cout << "Loss gotten" << endl;
            for (int j = 0; j < model[nbr-i].params.grads.shape().rows; j++)
            {
                for (int k = 0; k < model[nbr-i].params.grads.shape().cols; k++)
                {
                    model[nbr-i].params.grads(j,k) = model[nbr-i-1].act.Trans()(j) * model[nbr - i].layer_loss(k); 
                }
            }   
        }
        else if (nbr - i == 0)
        {
            model[nbr - i].layer_loss = (model[nbr-i+1].layer_loss.matmul(model[nbr-i+1].params.weights.Trans())).\
                element_mul(Tanh_d(model[nbr-i].z));
            for (int j = 0; j < model[nbr-i].params.grads.shape().rows; j++)
            {
                for (int k = 0; k < model[nbr-i].params.grads.shape().cols; k++)
                {
                    model[nbr-i].params.grads(j,k) = input_.Trans()(j) * model[nbr - i].layer_loss(k); 
                }
            }   
        }
    }
    }    

   void optimize(Module & model, double  lr){
    for (int l = 0; l < layer_num(model); l++)
    {
        for (int i = 0; i < model[l].params.weights.shape().rows; i++)
        {
            for (int j = 0; j < model[l].params.weights.shape().cols; j++)
            {   
                double weight_ = model[l].params.weights(i,j);
                model[l].params.weights(i,j) = weight_ - (lr * model[l].params.grads(i,j));
            }    
        }
    } 
    
    }


}