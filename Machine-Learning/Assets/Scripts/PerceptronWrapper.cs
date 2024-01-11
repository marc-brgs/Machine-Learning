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
    public static extern void trainPerceptron(IntPtr perceptron, double[] input, double[] targetOutput, double learningRate);
    
    [DllImport("MLLibrary")]
    public static extern void predictPerceptron(IntPtr perceptron, double[] input, double[] output);

    [DllImport("MLLibrary")]
    public static extern double[] getOutputError(IntPtr perceptron);

    public IntPtr perceptron;
    public double learningRate;

    private int outputSize;

    public PerceptronWrapper(int inputSize, int[] hiddenLayerSize, int outputSize)
    {
        IntPtr perceptronPtr;
        createPerceptron(out perceptronPtr);
        Debug.Log("Perceptron created");

        this.outputSize = outputSize;
        initializePerceptron(perceptronPtr, inputSize, hiddenLayerSize, hiddenLayerSize.Length, outputSize); // par référence?
        Debug.Log("Perceptron initialized");

        this.perceptron = perceptronPtr;
        this.learningRate = 0.02;
    }

    ~PerceptronWrapper()
    {
        destroyPerceptron(this.perceptron);
        Debug.Log("Destroy perceptron");
    }

    public void train(double[] input, double[] targetOutput)
    {
        trainPerceptron(this.perceptron, input, targetOutput, this.learningRate);
        //Debug.Log("Perceptron train");
    }

    public double[] predict(double[] input, bool print=false)
    {
        double[] output = new double[this.outputSize];

        predictPerceptron(this.perceptron, input, output);

        string outputString = "";
        for(int i = 0; i < output.Length; i++)
        {
            outputString += output[i].ToString();
            if (i != output.Length-1)
            {
                outputString += ", ";
            }
        }

        string inputString = "";
        for (int i = 0; i < input.Length; i++)
        {
            inputString += input[i].ToString();
            if (i != input.Length - 1)
            {
                inputString += ", ";
            }
        }

        if (print)
        {
            Debug.Log("Feed forward : [ " + inputString + " ] => [ " + outputString + " ]");
        }

        return output;
    }

    public void getOutputError()
    {
        double[] error = getOutputError(this.perceptron);
    }
}
