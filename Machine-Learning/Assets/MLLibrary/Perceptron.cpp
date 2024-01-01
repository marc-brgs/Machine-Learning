#include "pch.h"
#include "Perceptron.h"
#include <random>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath> // exp()

// Utiliser un g�n�rateur de nombres al�atoires
std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0, 1.0);

// La fonction d'activation sigmoid
double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

// La d�riv�e de la fonction sigmoid
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

Perceptron::Perceptron() {
    
}

Perceptron::~Perceptron() {
    
}

void Perceptron::initialize(int inputSize, const std::vector<int>& hiddenLayerSizes, int outputSize) {
    this->inputLayerSize = inputSize;
    this->hiddenLayerSizes = hiddenLayerSizes;
    this->outputLayerSize = outputSize;

    initializeWeights();
    initializeBiases();
}

void Perceptron::initializeWeights() {
    // Initialiser les poids pour chaque couche, y compris la couche de sortie
    int layerCount = hiddenLayerSizes.size() + 1; // couche cach�e + sortie

    weights.resize(layerCount);

    // Taille de la premi�re couche de poids
    int previousLayerSize = inputLayerSize;

    for (int i = 0; i < layerCount; ++i) {
        int currentLayerSize = (i == hiddenLayerSizes.size()) ? outputLayerSize : hiddenLayerSizes[i];

        weights[i].resize(currentLayerSize, std::vector<double>(previousLayerSize));

        for (int j = 0; j < currentLayerSize; ++j) {
            for (int k = 0; k < previousLayerSize; ++k) {
                weights[i][j][k] = distribution(generator); // initialise avec une distribution normale
            }
        }

        previousLayerSize = currentLayerSize;
    }
}

void Perceptron::initializeBiases() {
    // Initialiser les biais pour chaque couche, y compris la couche de sortie
    int layerCount = hiddenLayerSizes.size() + 1; // couche cach�e + sortie

    biases.resize(layerCount);

    for (int i = 0; i < layerCount; ++i) {
        int currentLayerSize = (i == hiddenLayerSizes.size()) ? outputLayerSize : hiddenLayerSizes[i];

        biases[i].resize(currentLayerSize);

        for (int j = 0; j < currentLayerSize; ++j) {
            biases[i][j] = distribution(generator); // initialise avec une distribution normale
        }
    }
}

void Perceptron::train(const double* input, const double* targetOutput, double learningRate, int epochs) {
    // Impl�mentez l'entra�nement de votre perceptron ici.
}

void Perceptron::predict(const double* input, double* output) {
    std::vector<double> layerInput(input, input + inputLayerSize);
    std::vector<double> layerOutput;

    // Propagation � travers chaque couche
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        layerOutput.clear();
        for (size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
            double activation = biases[layer][neuron];
            for (size_t weightIndex = 0; weightIndex < weights[layer][neuron].size(); ++weightIndex) {
                // Accumuler les entr�es pond�r�es
                activation += layerInput[weightIndex] * weights[layer][neuron][weightIndex];
            }
            // Appliquer la fonction d'activation
            layerOutput.push_back(sigmoid(activation));
        }

        // La sortie de cette couche devient l'entr�e de la couche suivante
        layerInput = layerOutput;
    }

    // Copier la sortie de la derni�re couche dans le tableau de sortie
    for (size_t i = 0; i < layerOutput.size(); ++i) {
        output[i] = layerOutput[i];
    }
}