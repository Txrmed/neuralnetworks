//
// Created by termed on 8.05.2026.
//

#ifndef DIGITRECOGNITION_DATALOADER_H
#define DIGITRECOGNITION_DATALOADER_H

#include<Eigen/Dense>
#include<fstream>
#include<iostream>

#define PIXEL_COUNT 784

// struct DatasetParameters {
//     unsigned int num_training_img;
//     unsigned int num_testing_img;
// };

struct DataFiles {

    // Training Data
    std::ifstream& train_images;
    std::ifstream& train_labels;

    // Testing data
    std::ifstream& test_images;
    std::ifstream& test_labels;

};

// Handle loading the MNIST Dataset into a cpp readable format.
class Dataset {

    // Amount of neurons in the input layer.
    const unsigned int input_layer_n = 0;

    // Amount of training images.
    const unsigned int n_training_img = 0;

    // Amount of testing images.
    const unsigned int n_testing_img = 0;

    Eigen::MatrixXf learning_data_;
    Eigen::MatrixXf testing_data_;

    unsigned int current_index = 0;

    //void validate_file_headers(std::ifstream& train_images, std::ifstream& train_labels, std::ifstream& test_images, ifstream& test_labels);

    // Some function to find files in /data/mnist/ dir
    DataFiles find_files();

public:

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