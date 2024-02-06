#include "pch.h"
#include "Perceptron.h"
#include <random>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <cmath> // exp()

// Générateur de nombres aléatoires entre 0 et 1
std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0, 1.0);

// La fonction d'activation sigmoid
double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

// La dérivée de la fonction sigmoid
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
    int layerCount = hiddenLayerSizes.size() + 1; // Couche cachée + sortie

    weights.resize(layerCount);

    // Taille de la première couche de poids
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
    int layerCount = hiddenLayerSizes.size() + 1; // Couche cachée + sortie

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

    // Calcul de l'erreur de sortie (différence entre la sortie attendue et la sortie actuelle)
    outputError = activations.back();
    for (size_t i = 0; i < outputLayerSize; ++i) {
        outputError[i] = activations.back()[i] - targetOutput[i];
    }

    // Back propagation
    for (int layer = weights.size() - 1; layer >= 0; --layer) {
        std::vector<double> layerDelta(weights[layer].size(), 0.0);
        for (size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
            double delta;
            if (layer == weights.size() - 1) {
                // Pour la dernière couche, on utilise l'erreur de sortie directement
                delta = outputError[neuron] * sigmoid_derivative(activations[layer][neuron]);
            }
            else {
                // Pour les couches cachées, on calcule l'erreur en fonction des deltas de la couche suivante
                double errorSum = 0.0;
                for (size_t nextNeuron = 0; nextNeuron < weights[layer + 1].size(); ++nextNeuron) {
                    errorSum += weights[layer + 1][nextNeuron][neuron] * deltas[layer + 1][nextNeuron];
                }
                delta = errorSum * sigmoid_derivative(activations[layer][neuron]);
            }
            layerDelta[neuron] = delta;

            // Mise à jour des poids
            for (size_t w = 0; w < weights[layer][neuron].size(); ++w) {
                double inputVal = layer > 0 ? activations[layer - 1][w] : inputs[w];
                weights[layer][neuron][w] -= learningRate * delta * inputVal;
            }

            // Mise à jour des biais
            biases[layer][neuron] -= learningRate * delta;
        }

        deltas[layer] = layerDelta; // Enregistre les deltas pour cette couche
    }
}



void Perceptron::predict(const double* input, double* output) {
    std::vector<double> layerInput(input, input + inputLayerSize);
    std::vector<double> layerOutput;
    activations.clear(); // Reset les activations à chaque nouvelle prédiction

    // Propagation à travers chaque couche
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        layerOutput.clear();
        for (size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
            double activation = biases[layer][neuron];
            for (size_t weightIndex = 0; weightIndex < weights[layer][neuron].size(); ++weightIndex) {
                // Accumulation des entrées pondérées
                activation += layerInput[weightIndex] * weights[layer][neuron][weightIndex];
            }
            // Application de la fonction d'activation
            layerOutput.push_back(sigmoid(activation));
        }

        // Utilisé pour la méthode train (back propagation)
        activations.push_back(layerOutput);

        // La sortie de cette couche devient l'entrée de la couche suivante
        layerInput = layerOutput;
    }

    // Copie de la sortie de la dernière couche dans le tableau de sortie
    for (size_t i = 0; i < layerOutput.size(); ++i) {
        output[i] = layerOutput[i];
    }
}

void Perceptron::saveToFile(const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Sérialisation des tailles des couches
    outFile.write(reinterpret_cast<const char*>(&inputLayerSize), sizeof(inputLayerSize));
    int hiddenLayersCount = hiddenLayerSizes.size();
    outFile.write(reinterpret_cast<const char*>(&hiddenLayersCount), sizeof(hiddenLayersCount));
    for (int size : hiddenLayerSizes) {
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }
    outFile.write(reinterpret_cast<const char*>(&outputLayerSize), sizeof(outputLayerSize));

    // Sérialisation des poids et biais
    for (const auto& layerWeights : weights) {
        for (const auto& neuronWeights : layerWeights) {
            for (double weight : neuronWeights) {
                outFile.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
            }
        }
    }
    for (const auto& layerBiases : biases) {
        for (double bias : layerBiases) {
            outFile.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
        }
    }

    outFile.close();
}

void Perceptron::loadFromFile(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    // Désérialisation des tailles des couches
    inFile.read(reinterpret_cast<char*>(&inputLayerSize), sizeof(inputLayerSize));
    int hiddenLayersCount;
    inFile.read(reinterpret_cast<char*>(&hiddenLayersCount), sizeof(hiddenLayersCount));
    hiddenLayerSizes.resize(hiddenLayersCount);
    for (int& size : hiddenLayerSizes) {
        inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    }
    inFile.read(reinterpret_cast<char*>(&outputLayerSize), sizeof(outputLayerSize));

    initializeWeights();
    initializeBiases();

    // Désérialisation des poids et biais
    for (auto& layerWeights : weights) {
        for (auto& neuronWeights : layerWeights) {
            for (double& weight : neuronWeights) {
                inFile.read(reinterpret_cast<char*>(&weight), sizeof(weight));
            }
        }
    }
    for (auto& layerBiases : biases) {
        for (double& bias : layerBiases) {
            inFile.read(reinterpret_cast<char*>(&bias), sizeof(bias));
        }
    }

    inFile.close();
}

void Perceptron::getOutputError(double* error) {
    for (size_t i = 0; i < outputLayerSize; i++) {
        error[i] = outputError[i];
    }
    //return outputError.data();
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