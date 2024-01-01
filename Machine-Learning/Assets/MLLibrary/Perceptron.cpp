#include "pch.h"
#include "Perceptron.h"
#include <random>
#include <iostream>
#include <cstdlib>
#include <vector>

// Utiliser un générateur de nombres aléatoires
std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0, 1.0);

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
    int layerCount = hiddenLayerSizes.size() + 1; // couche cachée + sortie

    weights.resize(layerCount);

    // Taille de la première couche de poids
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
    int layerCount = hiddenLayerSizes.size() + 1; // couche cachée + sortie

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
    // Implémentez l'entraînement de votre perceptron ici.
}

void Perceptron::predict(const double* input, double* output) {
    // Implémentez la prédiction de votre perceptron ici.
}