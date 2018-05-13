#include "NeuralNetwork.h"
#include <iostream>

using namespace std;

int main(int argv, char **argc) {
  int *networkInformation = new int[4]{3, 25, 25, 1};
  float input[8][3] = {
      {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
      {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
  };
  float expectedValue[8][1] = {{0.7}, {0.6}, {0.5}, {0.4}, {0.3}, {0.2}, {0.1}, {0}};

  cout << "NEURAL NETWORK IN CPP" << endl;

  NeuralNetwork *network = new NeuralNetwork(networkInformation);

  for (int i = 0; i < 5000; i++) {
    for (int j = 0; j < 8; j++) {
      network->feedForward(input[j]);
      network->backPropagation(expectedValue[j]);
    }
  }

  for (int j = 0; j < 8; j++) {
    cout << *(network->feedForward(input[j])) << endl;
  }

  return 0;
}