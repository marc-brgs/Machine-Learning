// MLLibrary.h - Contains declarations of machine learning functions
#pragma once

#include "Perceptron.h"

#ifdef MLLIBRARY_API_EXPORTS
#define MLLIBRARY_API __declspec(dllexport)
#else
#define MLLIBRARY_API __declspec(dllimport)
#endif

extern "C" MLLIBRARY_API void createPerceptron(Perceptron * *perceptron);
extern "C" MLLIBRARY_API void destroyPerceptron(Perceptron * perceptron);

extern "C" MLLIBRARY_API void initializePerceptron(Perceptron * perceptron, int inputSize, int* hiddenLayerSizes, int hiddenLayerCount, int outputSize);
extern "C" MLLIBRARY_API void trainPerceptron(Perceptron * perceptron, const double* input, const double* targetOutput, double learningRate, int epochs);
extern "C" MLLIBRARY_API void predictPerceptron(Perceptron* perceptron, const double* input, double* output);