#include <cmath>
#include <ctime>
#include <iostream>

#ifndef DEFAULT_LEARNING_RATE
#define DEFAULT_LEARNING_RATE 0.33
#endif

class Layer {
private:
  float learningRate;

  int numberOfInputs; // of neurons in the previous layer
  int numberOfOuputs; // of neurons in the current layer

  float *outputs;       // outputs array of this layer
  float *inputs;        // inputs array in into this layer
  float **weights;      // weights array of this layer
  float **weightsDelta; // deltas array of this layer
  float *gamma;         // gamma array of this layer
  float *error;         // error array of the output layer

  float tanHDer(float value);

public:
  Layer(int numberOfInputs, int numberOfOuputs,
        float learningRate = DEFAULT_LEARNING_RATE);

  void initializeWeights(void);
  float *feedForward(float *inputs);
  void backPropagationOutput(float *expectedValue);
  void backPropagationHidden(float *gammaForward, float **weightsForward);
  void updateWeights(void);

  float *getOutput(void);
  float *getGamma(void);
  float **getWeights(void);
};
