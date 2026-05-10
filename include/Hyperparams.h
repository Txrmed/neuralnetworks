//
// Created by termed on 10.05.2026.
//

#ifndef DIGITRECOGNITION_HYPERPARAMETERS_H
#define DIGITRECOGNITION_HYPERPARAMETERS_H

// Struct for storing hyperparameters of the network.
struct Hyperparams{
    unsigned int mini_batch_size;
    float learning_rate;

    int num_learning_img;
    int num_testing_img;

    unsigned int epochs;
};

#endif //DIGITRECOGNITION_HYPERPARAMETERS_H