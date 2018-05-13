#include "layer.h"

Layer::Layer(int numberOfInputs, int numberOfOuputs, float learningRate) {
  this->numberOfInputs = numberOfInputs;
  this->numberOfOuputs = numberOfOuputs;
  this->learningRate = learningRate;

  inputs = new float[numberOfInputs];
  outputs = new float[numberOfOuputs];
  gamma = new float[numberOfOuputs];
  error = new float[numberOfOuputs];
  weights = new float *[numberOfOuputs];
  weightsDelta = new float *[numberOfOuputs];
  for (int i = 0; i < numberOfOuputs; i++) {
    weights[i] = new float[numberOfInputs];
    weightsDelta[i] = new float[numberOfInputs];
  }

  initializeWeights();
}

void Layer::initializeWeights(void) {
  srand(time(NULL));

  for (int i = 0; i < numberOfOuputs; i++) {
    for (int j = 0; j < numberOfInputs; j++) {
      weights[i][j] = (rand() % 100 - 50) / 100.0; // Random from -0.5 to 0.5
    }
  }
}

float *Layer::feedForward(float *inputs) {
  // keep input for back-propagation use
  for (int i = 0; i < numberOfInputs; i++) {
    this->inputs[i] = inputs[i];
  }

  // feed forward
  for (int i = 0; i < numberOfOuputs; i++) {
    outputs[i] = 0;
    for (int j = 0; j < numberOfInputs; j++) {
      outputs[i] += inputs[j] * weights[i][j];
    }

    outputs[i] = (float)tanh(outputs[i]);
  }

  return outputs;
}

float Layer::tanHDer(float value) { return 1 - (value * value); }

void Layer::backPropagationOutput(float *expectedValue) {
  // Calculate error
  for (int i = 0; i < numberOfOuputs; i++) {
    error[i] = outputs[i] - expectedValue[i];
  }

  // Calculate gamma
  for (int i = 0; i < numberOfOuputs; i++) {
    gamma[i] = error[i] * tanHDer(outputs[i]);
  }

  // Calculate weight deltas
  for (int i = 0; i < numberOfOuputs; i++) {
    for (int j = 0; j < numberOfInputs; j++) {
      weightsDelta[i][j] = gamma[i] * inputs[j];
    }
  }
}

void Layer::backPropagationHidden(float *gammaForward, float **weightsForward) {
  int sizeOfForwardLayer = sizeof(weightsForward) / sizeof(weightsForward[0]);

  // Calculate new gamma value using sum of forward layer gamma values
  for (int i = 0; i < numberOfOuputs; i++) {
    gamma[i] = 0;

    for (int j = 0; j < sizeOfForwardLayer; j++) {
      gamma[i] += gammaForward[i] * weightsForward[j][i];
    }
  }

  // Calculate weight deltas
  for (int i = 0; i < numberOfOuputs; i++) {
    for (int j = 0; j < numberOfInputs; j++) {
      weightsDelta[i][j] = gamma[i] * inputs[j];
    }
  }
}

void Layer::updateWeights(void) {
  for (int i = 0; i < numberOfOuputs; i++) {
    for (int j = 0; j < numberOfInputs; j++) {
      weights[i][j] -= weightsDelta[i][j] * learningRate;
    }
  }
}

float *Layer::getOutput(void) { return outputs; }

float *Layer::getGamma(void) { return gamma; }

float **Layer::getWeights(void) { return weights; }
