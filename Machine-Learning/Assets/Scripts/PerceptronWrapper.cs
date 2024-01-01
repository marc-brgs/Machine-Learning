using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Runtime.InteropServices;

public class PerceptronWrapper
{
    [DllImport("MLLibrary")]
    public static extern void createPerceptron(out IntPtr perceptron);

    [DllImport("MLLibrary")]
    public static extern void destroyPerceptron(IntPtr perceptron);

    [DllImport("MLLibrary")]
    public static extern void initializePerceptron(IntPtr perceptron, int inputSize, int[] hiddenLayerSize, int hiddenLayerCount, int outputSize);
    
    [DllImport("MLLibrary")]
    public static extern void trainPerceptron(IntPtr perceptron, double[] input, double[] targetOutput, double learningRate, int epochs);
    
    [DllImport("MLLibrary")]
    public static extern void predictPerceptron(IntPtr perceptron, double[] input, double[] output);

    public IntPtr perceptron;
    public double learningRate;

    public PerceptronWrapper(int inputSize, int[] hiddenLayerSize, int outputSize)
    {
        IntPtr perceptronPtr;
        createPerceptron(out perceptronPtr);
        Debug.Log("Perceptron created");

        initializePerceptron(perceptronPtr, inputSize, hiddenLayerSize, hiddenLayerSize.Length, outputSize); // par référence?
        Debug.Log("Perceptron created");

        this.perceptron = perceptronPtr;
    }

    ~PerceptronWrapper()
    {
        destroyPerceptron(this.perceptron);
        Debug.Log("Destroy perceptron");
    }

    public void train(double[] input, double[] targetOutput, int epochs)
    {
        trainPerceptron(this.perceptron, input, targetOutput, this.learningRate, epochs);
        Debug.Log("Perceptron train");
    }

    public void predict(double[] input, ref double[] output)
    {
        predictPerceptron(this.perceptron, input, output);
        Debug.Log("Perceptron predict");
    }
}
