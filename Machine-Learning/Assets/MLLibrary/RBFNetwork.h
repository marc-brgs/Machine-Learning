#pragma once
#include <vector>
#include <string>

class RBFNetwork {
public:
    RBFNetwork(int inputSize, int hiddenSize, int outputSize);
    ~RBFNetwork();

    void initialize();
    void train(const std::vector<double>& input, const std::vector<double>& target, double learningRate);
    std::vector<double> predict(const std::vector<double>& input);

private:
    int inputLayerSize;
    int hiddenLayerSize;
    int outputLayerSize;

    std::vector<std::vector<double>> centers; // Centres des neurones RBF
    std::vector<double> sigmas; // Écarts-types des neurones RBF
    std::vector<std::vector<double>> outputWeights; // Poids de la couche de sortie
    std::vector<double> outputBiases; // Biais de la couche de sortie

    double radialBasisFunction(double distance, double sigma);
    void calculateCentersAndSigmas(); // À implémenter selon votre algorithme de choix
};