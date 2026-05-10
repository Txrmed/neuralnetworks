#include "../include/Network.h"
#include "../include/Hyperparams.h"

#include<Eigen/Dense>
#include<iostream>

using namespace std;

int main() {
    const string path = "~/trelemorele/sidehoes/digitrecognition/data/";

    const vector<unsigned int> neurons = {784, 30, 10};

    constexpr Hyperparams parameters{10, 2, 60000, 10000, 10};

    Dataset dataset(path, parameters);


    // Eigen::MatrixXf mini_batch = dataset.get_mini_batch(parameters.mini_batch_size);
    //
    // for (int image = 0; image < 10; image++) {
    //     cout << mini_batch(0, image) << endl;
    //     for (int i = 0; i < 28; i++) {
    //         for (int j = 1; j < 28; j++) {
    //             if (mini_batch(j + i * 28, image) > 0.5) {
    //                 cout << "# ";
    //             }
    //             else {
    //                 cout << ". ";
    //             }
    //         }
    //         cout << endl;
    //     }
    //     cout << endl << endl;
    // }


    Network network(neurons, parameters, dataset);
    network.train();

}