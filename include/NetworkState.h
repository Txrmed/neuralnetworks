//
// Created by termed on 10.05.2026.
//

#ifndef DIGITRECOGNITION_NETWORKSTATE_H
#define DIGITRECOGNITION_NETWORKSTATE_H

struct NetworkState {

    std::vector<Eigen::VectorXf> zs;
    std::vector<Eigen::VectorXf> activations;

    Eigen::VectorXf output;
    Eigen::VectorXf label;
};

#endif //DIGITRECOGNITION_NETWORKSTATE_H