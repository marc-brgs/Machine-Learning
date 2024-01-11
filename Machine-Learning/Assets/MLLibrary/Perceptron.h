#pragma once
#include <vector>
#include <string>

class Perceptron {
public:
    Perceptron();
    ~Perceptron();

    // Init (input, [hidden], output)
    void initialize(int inputSize, const std::vector<int>& hiddenLayerSizes, int outputSize);

    // Back propagation
    void train(const double* input, const double* targetOutput, double learningRate);

    // Feed forward
    void predict(const double* input, double* output);

private:
    // Tailles des couches
    int inputLayerSize;
    std::vector<int> hiddenLayerSizes;
    int outputLayerSize;

    // Poids et biais
    std::vector<std::vector<std::vector<double>>> weights; // poids [couche][neurone][poids]
    std::vector<std::vector<double>> biases; // biais [couche][neurone]
    std::vector<std::vector<double>> activations; // activations [couche][neurone]

    // Initialisation des poids et biais
    void initializeWeights();
    void initializeBiases();

    // Log file
    void logClear();
    void logPrint(std::string str);
};