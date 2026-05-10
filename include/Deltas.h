//
// Created by termed on 10.05.2026.
//

#ifndef DIGITRECOGNITION_COSTGRADIENT_H
#define DIGITRECOGNITION_COSTGRADIENT_H

#include<vector>
#include<Eigen/Dense>

struct Deltas {
    // Changes for biases at i-th layer;
    std::vector<Eigen::VectorXf> d_biases;

    // Changes for weights.
    std::vector<Eigen::MatrixXf> d_weights;
};

#endif //DIGITRECOGNITION_COSTGRADIENT_H