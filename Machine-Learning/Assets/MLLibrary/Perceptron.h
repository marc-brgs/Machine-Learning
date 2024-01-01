#pragma once
#include <vector>

class Perceptron {
public:
    Perceptron();
    ~Perceptron();

    // Initialiser le perceptron avec le nombre d'entrées, de neurones cachés et de sorties
    void initialize(int inputSize, const std::vector<int>& hiddenLayerSizes, int outputSize);

    // Entraîner le perceptron avec les données d'entrée et de sortie attendues
    void train(const double* input, const double* targetOutput, double learningRate, int epochs);

    // Effectuer une prédiction avec le perceptron
    void predict(const double* input, double* output);

private:
    int inputLayerSize; // Tailles des différentes couches
    std::vector<int> hiddenLayerSizes;
    int outputLayerSize;

    // Poids et biais
    std::vector<std::vector<std::vector<double>>> weights; // poids [couche][neurone][poids]
    std::vector<std::vector<double>> biases; // biais [couche][neurone]

    // Initialisation des poids et biais
    void initializeWeights();
    void initializeBiases();
};