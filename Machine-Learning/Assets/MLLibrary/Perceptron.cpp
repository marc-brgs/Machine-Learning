#include "pch.h"
#include "Perceptron.h"
#include <random>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <cmath> // exp()

// G�n�rateur de nombres al�atoires entre 0 et 1
std::default_random_engine generator;
std::normal_distribution<double> distribution(-1.0, 1.0);

// La fonction d'activation sigmoid
double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

// La d�riv�e de la fonction sigmoid
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

Perceptron::Perceptron() {
    logClear();
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
    int layerCount = hiddenLayerSizes.size() + 1; // Couche cach�e + sortie

    weights.resize(layerCount);

    // Taille de la premi�re couche de poids
    int previousLayerSize = inputLayerSize;

    for (int i = 0; i < layerCount; ++i) {
        int currentLayerSize = (i == hiddenLayerSizes.size()) ? outputLayerSize : hiddenLayerSizes[i];

        weights[i].resize(currentLayerSize, std::vector<double>(previousLayerSize));

        for (int j = 0; j < currentLayerSize; ++j) {
            for (int k = 0; k < previousLayerSize; ++k) {
                weights[i][j][k] = distribution(generator); // Initialise avec une valeur entre 0 et 1
            }
        }

        previousLayerSize = currentLayerSize;
    }
}

void Perceptron::initializeBiases() {
    // Initialise les biais pour chaque couche, y compris la couche de sortie
    int layerCount = hiddenLayerSizes.size() + 1; // Couche cach�e + sortie

    biases.resize(layerCount);

    for (int i = 0; i < layerCount; ++i) {
        int currentLayerSize = (i == hiddenLayerSizes.size()) ? outputLayerSize : hiddenLayerSizes[i];

        biases[i].resize(currentLayerSize);

        for (int j = 0; j < currentLayerSize; ++j) {
            biases[i][j] = distribution(generator); // Initialise avec une valeur entre 0 et 1
        }
    }
}

void Perceptron::train(const double* input, const double* targetOutput, double learningRate) {
    std::vector<double> inputs(input, input + inputLayerSize);
    std::vector<double> outputs(outputLayerSize);

    // Structures pour stocker les activations et les deltas pour chaque couche
    std::vector<std::vector<double>> deltas(weights.size());

    // Feed forward
    predict(input, outputs.data());

    // Calcul de l'erreur de sortie (diff�rence entre la sortie attendue et la sortie actuelle)
    std::vector<double> outputError = activations.back();
    for (size_t i = 0; i < outputLayerSize; ++i) {
        outputError[i] = activations.back()[i] - targetOutput[i];
    }

    // Back propagation
    for (int layer = weights.size() - 1; layer >= 0; --layer) {
        std::vector<double> layerDelta(weights[layer].size(), 0.0);
        for (size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
            double delta;
            if (layer == weights.size() - 1) {
                // Pour la derni�re couche, on utilise l'erreur de sortie directement
                delta = outputError[neuron] * sigmoid_derivative(activations[layer][neuron]);
            }
            else {
                // Pour les couches cach�es, on calcule l'erreur en fonction des deltas de la couche suivante
                double errorSum = 0.0;
                for (size_t nextNeuron = 0; nextNeuron < weights[layer + 1].size(); ++nextNeuron) {
                    errorSum += weights[layer + 1][nextNeuron][neuron] * deltas[layer + 1][nextNeuron];
                }
                delta = errorSum * sigmoid_derivative(activations[layer][neuron]);
            }
            layerDelta[neuron] = delta;

            // Mise � jour des poids
            for (size_t w = 0; w < weights[layer][neuron].size(); ++w) {
                double inputVal = layer > 0 ? activations[layer - 1][w] : inputs[w];
                weights[layer][neuron][w] -= learningRate * delta * inputVal;
            }

            // Mise � jour des biais
            biases[layer][neuron] -= learningRate * delta;
        }

        deltas[layer] = layerDelta; // Enregistre les deltas pour cette couche
    }
}



void Perceptron::predict(const double* input, double* output) {
    std::vector<double> layerInput(input, input + inputLayerSize);
    std::vector<double> layerOutput;
    activations.clear(); // Reset les activations � chaque nouvelle pr�diction

    // Propagation � travers chaque couche
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        layerOutput.clear();
        for (size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
            double activation = biases[layer][neuron];
            for (size_t weightIndex = 0; weightIndex < weights[layer][neuron].size(); ++weightIndex) {
                // Accumulation des entr�es pond�r�es
                activation += layerInput[weightIndex] * weights[layer][neuron][weightIndex];
            }
            // Application de la fonction d'activation
            layerOutput.push_back(sigmoid(activation));
        }

        // Utilis� pour la m�thode train (back propagation)
        activations.push_back(layerOutput);

        // La sortie de cette couche devient l'entr�e de la couche suivante
        layerInput = layerOutput;
    }

    // Copie de la sortie de la derni�re couche dans le tableau de sortie
    for (size_t i = 0; i < layerOutput.size(); ++i) {
        output[i] = layerOutput[i];
    }
}

void Perceptron::logClear() {
    std::ofstream logFile("path_to_log_file.txt", std::ofstream::out | std::ofstream::trunc);
    logFile.close();
}

void Perceptron::logPrint(std::string str) {
    std::ofstream logFile("path_to_log_file.txt", std::ios::out | std::ios::app);
    logFile << str << std::endl;
    logFile.close();
}