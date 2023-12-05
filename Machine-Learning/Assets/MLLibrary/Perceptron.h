#pragma once

class Perceptron {
public:
    // Initialiser le perceptron avec le nombre d'entr�es, de neurones cach�s et de sorties.
    void initialize(int inputSize, int hiddenLayerSize, int outputSize);

    // Entra�ner le perceptron avec les donn�es d'entr�e et de sortie attendues.
    void train(const double* input, const double* targetOutput, double learningRate, int epochs);

    // Effectuer une pr�diction avec le perceptron.
    void predict(const double* input, double* output);

private:
    // Ajouter vos membres priv�s pour les poids, les biais, etc.
};