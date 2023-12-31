// MathLibrary.cpp : Defines the exported functions for the DLL.
#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
#include <utility>
#include <limits.h>
#include "MLLibrary.h"
#include "Perceptron.h"

// DLL internal state variables:
static unsigned long long previous_;  // Previous value, if any
static unsigned long long current_;   // Current sequence value
static unsigned index_;               // Current seq. position

void createPerceptron(Perceptron * *perceptron) {
    *perceptron = new Perceptron();
}

void destroyPerceptron(Perceptron * perceptron) {
    delete perceptron;
}

void initializePerceptron(Perceptron * perceptron, int inputSize, int* hiddenLayerSizes, int hiddenLayerCount, int outputSize) {
    std::vector<int> hiddenSizes(hiddenLayerSizes, hiddenLayerSizes + hiddenLayerCount);
    perceptron->initialize(inputSize, hiddenSizes, outputSize);
}

void trainPerceptron(Perceptron * perceptron, const double* input, const double* targetOutput, double learningRate, int epochs) {
    perceptron->train(input, targetOutput, learningRate, epochs);
}

void predictPerceptron(Perceptron * perceptron, const double* input, double* output) {
    perceptron->predict(input, output);
}
