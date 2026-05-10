//
// Created by termed on 8.05.2026.
//

#ifndef DIGITRECOGNITION_DATALOADER_H
#define DIGITRECOGNITION_DATALOADER_H

#include "../include/Hyperparams.h"

#include <string>
#include<Eigen/Dense>

#define PIXEL_COUNT 784

// Handle loading the MNIST Dataset into a cpp readable format.
class Dataset {

    std::string path_;

    unsigned int num_learning_images_;
    unsigned int num_testing_images_;

    Eigen::MatrixXf learning_data_;
    Eigen::MatrixXf testing_data_;

    unsigned int current_index;

    // Private function to reverse the endiannes of an int.
    int reverse();

public:
    Dataset(const std::string& path, const Hyperparams& parameters);

    // Load entire MNIST set into learning_data_ and test_data_.
    void load_data();

    // Randomy shuffle learning data.
    void shuffle();

    // Get a mini batch of images.
    Eigen::Ref<const Eigen::MatrixXf> get_mini_batch(const unsigned int& amount);

    // Get testing data to test our model on.
    [[nodiscard]] Eigen::Ref<const Eigen::MatrixXf> get_testing_data(unsigned int n) const;

    // Return the index to 0.
    void set_return(){ current_index = 0; }

};


#endif //DIGITRECOGNITION_DATALOADER_H