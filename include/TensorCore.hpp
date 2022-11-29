/*
This file contains the main work horse class of CDLL Library : "Tensor"
    And the Algebraic and mathematical operations needed to construct and train
    Deep Neural Networks on a Distributed-FPGA based system.
*/


/*
    Add indexing [] and double indexing [][] for Tensor<T> class:
*/

#pragma once

#include <iostream>
#include "NumCpp.hpp"

using namespace std;
using namespace nc;

namespace TensorCore{
    template <typename T>
    class Tensor : public NdArray<T>
    {
            // Constructors
        public:
            Tensor<T>(initializer_list<T> InList) : NdArray<T>(InList) {}
            Tensor<T>(initializer_list<initializer_list<T>> InList) : NdArray<T>(InList) {} // 2D Tensor
            Tensor<T>(NdArray<T> InArray) : NdArray<T>(InArray) {} // Constructer with NdArray paramter (Conversion: base --> Tensor)
            Tensor<T>(T x) : NdArray<T>(x){}
            Tensor<T>(){}
            // Conversion from NumCpp.NdArray to Tensor using form_cpp..
        public:
            Tensor<T> from_numcpp(NdArray<T> in_arr){
                return Tensor<T>(in_arr);
                }
            // Example Reshape Function for Tensor objects
        public:
            auto Reshape(const int & x,const int & y)
            {
                return from_numcpp((*this).reshape(x, y));
            }
            auto Trans(){
                return Tensor<double>((*this).transpose());
            }
            auto matmul(Tensor<double> other){
                return Tensor<double>(dot(*this, other));
            }
            auto sub(Tensor<T> other){
                return Tensor<double>(subtract(*this, other));
            }
            auto mul_val(double val){
                return Tensor<double>(multiply(*this, multiply(ones<double>(Shape((*this).shape().rows,\
                 (*this).shape().cols)),val)));
            }
            auto element_mul(Tensor<double> other){
                return Tensor<double>(multiply(*this, other));
            }
            auto Mean(){
                return from_numcpp(mean(*this));        
            }    
            auto std(){
                return from_numcpp(stdev(*this));
            }
            auto Row(int r){
                return from_numcpp((*this).row(r));
            }
            auto Append(Tensor<double> other){
                return Tensor<double>(append(*this, other));
            }
            auto avg(){
                return Tensor<double>(average(*this));
            }
           
            ostream operator<<(Tensor<T> arg2); // Operator overloading for Tensor Object
            T& operator()(int);
            T& operator()(int, int); 
    };
        template<typename T>
        T& Tensor<T>::operator()(int x){
            return (T&)(*this)[x];            
        }
        
        template<typename T>
        T& Tensor<T>::operator()(int x, int y){
            return (T&)(*this)[((*this).shape().cols)*x + y];
        }

namespace init{
    /*
    Random initialization for Tensor elements (Trainable Prameters in a Neural Network Class)
    TODO: Xavier_init for CNN's
    */
    
    // normal_dist creates a Tensor of shape [x,y] with elements distributed within a normal distribution
        auto normal_dist(const int & x, const int & y){
            return Tensor<double>(random::normal<double>(Shape(x,y)));
            }
    // uniform_dist creates a Tensor of shape [x,y] with elements distributed within a uniform distribution
        auto uniform_dist(const int & x, const int & y){
            return Tensor<double>(random::uniform<double>(Shape(x,y), -0.5, 0.5));
            }
    }
    
namespace norm{
    /*
    Normalization flows for Data and Activations (Forward Pass)
    TODO: Layer_norm + batch_norm
    */
        auto z_norm(Tensor<double> in_tens){
            in_tens = Tensor<double>(divide(subtract(in_tens, multiply(ones<double>(Shape((int)in_tens.shape().rows, \
                (int)in_tens.shape().cols)), in_tens.Mean().item())), in_tens.std().item())); 
            return in_tens;
            }
    }
}