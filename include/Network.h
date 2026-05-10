//
// Created by termed on 2.05.2026.
//

#ifndef DIGITRECOGNITION_NETWORK_H
#define DIGITRECOGNITION_NETWORK_H

#include "../include/Dataset.h"
#include "../include/Hyperparams.h"
#include "../include/NetworkState.h"
#include "../include/Deltas.h"

#include<vector>
#include<Eigen/Dense>


class Network {
    float e_ = 2.71828;

    Dataset dataset_;

    Hyperparams parameters_;

    // Amount of layers in the network.
    const unsigned int layers_;

    // Amount of neurons at each layer.
    std::vector<unsigned int> neurons_;

    // weights and biases vectors.
    std::vector<Eigen::MatrixXf> weights_;
    std::vector<Eigen::VectorXf> biases_;

    // Apply the sigmoid function to x.
    [[nodiscard]] float sigmoid(const float x) const {
        return static_cast<float>(1 / (1 + pow(e_, x)));
    }

    // Sigmoid derivative
    [[nodiscard]] float sigmoid_prime(const float x) const {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    // Return the cost function for output 'a' and label vector.
    [[nodiscard]] float cost_function(const Eigen::VectorXf& a, const Eigen::VectorXf& label) const {
        return 0.5f * (a - label).array().square().sum();

    }

    // Returns an array with all layer activations stored in a vector.
    [[nodiscard]] NetworkState feedforward(const Eigen::VectorXf& input, unsigned int label) const;

    [[nodiscard]] Deltas backpropagation(const NetworkState& state);

    // Return number of correctly classified images.
    unsigned int test_model(unsigned int n) const ;


public:
    explicit Network(const std::vector<unsigned int>& neurons, const Hyperparams& parameters, const Dataset& dataset);

    // Trains the neural network on the loaded dataset using backpropagation.
    void train();

};

#endif //DIGITRECOGNITION_NETWORK_H