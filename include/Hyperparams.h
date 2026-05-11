//
// Created by termed on 10.05.2026.
//

#ifndef DIGITRECOGNITION_HYPERPARAMETERS_H
#define DIGITRECOGNITION_HYPERPARAMETERS_H

// Struct for storing hyperparameters of the network.
struct Hyperparams{
    // Mini batch size for SGD
    unsigned int mini_batch_size;

    // Eta - learning rate
    float learning_rate;

    // Amount of learning images in the dataset.
    int num_learning_img;

    // Amount of testing images in the provided dateset.
    int num_testing_img;

    // Epochs
    unsigned int epochs;
};

#endif //DIGITRECOGNITION_HYPERPARAMETERS_H