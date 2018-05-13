#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int *layerSizes) {
  this->numberOfLayer = sizeof(layerSizes) / sizeof(layerSizes[0]);
  layers = new Layer *[numberOfLayer - 1];
  for (int i = 0; i < numberOfLayer - 1; i++) {
    layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
  }
}

float *NeuralNetwork::feedForward(float *inputs) {
  layers[0]->feedForward(inputs);

  for (int i = 1; i < numberOfLayer - 1; i++) {
    layers[i]->feedForward(layers[i - 1]->getOutput());
  }

  return (layers[numberOfLayer - 2])->getOutput();
}

void NeuralNetwork::backPropagation(float *expectedValue) {
  for (int i = (numberOfLayer - 2); i >= 0; i--) {
    if (i == (numberOfLayer - 2)) {
      layers[i]->backPropagationOutput(expectedValue);
    } else {
      layers[i]->backPropagationHidden(layers[i + 1]->getGamma(),
                                       layers[i + 1]->getWeights());
    }
  }

  for (int i = 1; i < numberOfLayer - 1; i++) {
    layers[i]->updateWeights();
  }
}
