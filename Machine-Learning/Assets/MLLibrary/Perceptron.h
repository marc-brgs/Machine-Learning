#pragma once

class Perceptron {
public:
    // Initialiser le perceptron avec le nombre d'entrées, de neurones cachés et de sorties.
    void initialize(int inputSize, int hiddenLayerSize, int outputSize);

    // Entraîner le perceptron avec les données d'entrée et de sortie attendues.
    void train(const double* input, const double* targetOutput, double learningRate, int epochs);

    // Effectuer une prédiction avec le perceptron.
    void predict(const double* input, double* output);

private:
    // Ajouter vos membres privés pour les poids, les biais, etc.
};