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
    public static extern void initializePerceptron(IntPtr perceptron, int inputSize, int[] hiddenLayerSize, int hiddenLayerCount, int outputSize, bool isLinearModel);
    
    [DllImport("MLLibrary")]
    public static extern void trainPerceptron(IntPtr perceptron, double[] input, double[] targetOutput, double learningRate);
    
    [DllImport("MLLibrary")]
    public static extern void predictPerceptron(IntPtr perceptron, double[] input, double[] output);

    [DllImport("MLLibrary")]
    public static extern void evaluatePerceptron(IntPtr perceptron, double[] error);

    [DllImport("MLLibrary")]
    public static extern void savePerceptronToFile(IntPtr perceptron, string filename);

    [DllImport("MLLibrary")]
    public static extern void loadPerceptronFromFile(IntPtr perceptron, string filename);

    public IntPtr perceptron;
    public double learningRate;
    public bool isLinearModel;

    private int outputSize;

    public PerceptronWrapper(int inputSize, int[] hiddenLayerSize, int outputSize, bool isLinearModel=false)
    {
        IntPtr perceptronPtr;
        createPerceptron(out perceptronPtr);
        Debug.Log("Perceptron created");

        this.outputSize = outputSize;
        initializePerceptron(perceptronPtr, inputSize, hiddenLayerSize, hiddenLayerSize.Length, outputSize, isLinearModel); // par référence?
        Debug.Log("Perceptron initialized (linear : " + isLinearModel + ")");

        this.perceptron = perceptronPtr;
        this.isLinearModel = isLinearModel;
        this.learningRate = 0.02; // Default value
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

    /**
     * Serialize perceptron to file
     */
    public void saveToFile(string filename)
    {
        savePerceptronToFile(perceptron, filename);
        Debug.Log("Perceptron saved to \"" + filename + "\"");
    }

    /**
     * Unserialize perceptron from file
     */
    public void loadFromFile(string filename)
    {
        if(System.IO.File.Exists(filename))
        {
            loadPerceptronFromFile(perceptron, filename);
            Debug.Log("Perceptron loaded from \"" + filename + "\"");
        }
        else
        {
            Debug.Log("Failed to load perceptron from \"" + filename + "\"");
        }
        
    }
}
