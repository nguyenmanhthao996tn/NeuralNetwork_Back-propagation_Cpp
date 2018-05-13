#include <iostream>
#include "layer.h"

class NeuralNetwork {
private:
  int numberOfLayer;
  int *layerSizes;
  Layer **layers;

public:
  NeuralNetwork(int *layerSizes);
  float *feedForward(float *inputs);
  void backPropagation(float *expectedValue);
};