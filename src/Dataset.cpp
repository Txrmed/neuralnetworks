//
// Created by termed on 8.05.2026.
//

#include "../include/Dataset.h"


#include<Eigen/Dense>
#include<fstream>
#include<iostream>
#include <algorithm> // for std::swap
#include <random>

#define PIXEL_COUNT 784

using namespace std;
Dataset::Dataset(const string& path, const Hyperparams& params) : path_(path),
                                                                  num_learning_images_(params.num_learning_img),
                                                                  num_testing_images_(params.num_testing_img),
                                                                  current_index(0) {};

void Dataset::load_data() {
    // Open training data.
    ifstream train_images("../data/train_images.idx3-ubyte", std::ios::binary);
    ifstream train_labels("../data/train_labels.idx1-ubyte", std::ios::binary);

    // Open testing data.
    ifstream test_images("../data/train_labels.idx1-ubyte", std::ios::binary);
    ifstream test_labels("../data/test_labels.idx1-ubyte", std::ios::binary);

    if (!train_images.is_open() || !train_labels.is_open() || !test_images.is_open() || !test_labels.is_open()) {
        throw std::runtime_error("Could not open image or label file!");
    }

    // Skip 16 byte MNIST header.
    train_images.seekg(16);
    test_images.seekg(16);

    // Skip 8 byte headers.
    train_labels.seekg(8);
    test_labels.seekg(8);

    learning_data_ = Eigen::MatrixXf(PIXEL_COUNT + 1, num_learning_images_);

    unsigned char buffer[PIXEL_COUNT + 1];
    unsigned char label[1];

    for (int col = 0; col < num_learning_images_; col++) {
        train_images.read(reinterpret_cast<char*>(buffer), PIXEL_COUNT);
        train_labels.read(reinterpret_cast<char*>(label), 1);

        if (col % 5000 == 0) std::cout << col << " training images loaded." << endl;

        // Add pixels into learning_data_ matrix.
        learning_data_(0, col) = label[0];
        for (int row = 0; row < PIXEL_COUNT; row++) {
            learning_data_(row + 1, col) = static_cast<float>(buffer[row]) / 255;
        }
    }

    testing_data_ = Eigen::MatrixXf(PIXEL_COUNT + 1, num_testing_images_);

    for (int col = 0; col < num_testing_images_; col++) {
        test_images.read(reinterpret_cast<char*>(buffer), PIXEL_COUNT);
        test_labels.read(reinterpret_cast<char*>(label), 1);

        if (col % 5000 == 0) std::cout << col << " testing images loaded." << endl;

        testing_data_(0, col) = label[0];
        for (int row = 0; row < PIXEL_COUNT; row++) {
            testing_data_(row + 1, col) = static_cast<float>(buffer[row]) / 255;
        }

    }
}

void Dataset::shuffle() {
    std::random_device rd;
    std::mt19937 g(rd());

    int n = static_cast<int>(learning_data_.cols());

    // 2. Fisher-Yates Shuffle
    for (int i = n - 1; i > 0; --i) {
        // Pick a random index from 0 to i
        std::uniform_int_distribution dist(0, i);
        int j = dist(g);

        // 3. Swap column i with column j
        // Eigen's .swap() is highly optimized for this
        learning_data_.col(i).swap(learning_data_.col(j));
    }

    n = static_cast<int>(testing_data_.cols());
    for (int i = n - 1; i > 0; --i) {
        // Pick a random index from 0 to i
        std::uniform_int_distribution dist(0, i);
        int j = dist(g);

        // 3. Swap column i with column j
        // Eigen's .swap() is highly optimized for this
        testing_data_.col(i).swap(testing_data_.col(j));
    }
}

Eigen::Ref<const Eigen::MatrixXf> Dataset::get_mini_batch(const unsigned int& amount) {
    current_index += amount;
    return learning_data_.block(0, current_index - amount, PIXEL_COUNT + 1, amount);
}

[[nodiscard]] Eigen::Ref<const Eigen::MatrixXf> Dataset::get_testing_data(unsigned int n) const{
    return testing_data_.block(0, 0, PIXEL_COUNT + 1, n);
}
