#pragma once

#include "nn.hpp"
#include "backprob.hpp"
#include <numeric>
using namespace TensorCore;
using namespace layers;
using namespace activations;
using namespace LossFunctions;
using namespace diff_act_funcs;
using namespace backprob;

/*
    Todo:
        - Create Data Class, and Data Loader.
        - Push to GitHub.
*/

double VecAVG(vector<double> & v) {
    if (v.empty()) {
        return 0;
    }
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

Tensor<double> Eval(Module & model, Tensor<double> val_data_tens, Tensor<double> val_labels_tens){
    Tensor<double> val_loss;
    Tensor<double> output;
    output = model.forward(val_data_tens);
    val_loss = CategoricalCrossEntropy(val_labels_tens, output);
    //cout << "Validation Loss : " << val_loss.item() << endl;
    return val_loss;
}

void Train(Module & model, Tensor<double> & data, Tensor<double> & labels,\
 Tensor<double> & val_data, Tensor<double> & val_labels, double learning_rate,\
                                             bool Verbose = false){
    
    Tensor<double> loss;
    Tensor<double> val_loss;
    Tensor<double> output;
    vector<double> train_losses;
    vector<double> val_losses; 

    for (int i = 0; i < data.shape().rows; i++)
    {
        output = model.forward(data.Row(i));
        loss = CategoricalCrossEntropy(labels.Row(i), output);
        train_losses.push_back(loss.item());
        grad_bp(model, data.Row(i),labels.Row(i));
        optimize(model, - learning_rate);
        val_loss = Eval(model, val_data.Row(i), val_labels.Row(i));
        val_losses.push_back(val_loss.item());
        if (Verbose)
        {
            cout << "Training Loss : " << VecAVG(train_losses) << endl;
            cout << "Validation Loss : " << VecAVG(val_losses) << endl;
        }
    }
        cout << "Training Loss : " << VecAVG(train_losses) << endl;
        cout << "Validation Loss : " << VecAVG(val_losses) << endl;
} 



