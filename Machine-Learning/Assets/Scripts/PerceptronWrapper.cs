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
    public static extern void evaluatePerceptron(IntPtr perceptron, double[] error);

    public IntPtr perceptron;
    public double learningRate;

    private int outputSize;
    private List<double> errorOverTime;

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
        this.errorOverTime = new List<double>();
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

    public void getOutputError(bool print=false)
    {
        double[] error = new double[this.outputSize];

        evaluatePerceptron(this.perceptron, error);

        double avgError = 0;
        string str = "";
        for (int i = 0; i < error.Length; i++)
        {
            if (print)
            {
                str += "ERROR[" + i + "] : " + error[i] + "\n";
            }

            avgError += Math.Abs(error[i]);
        }
        avgError /= error.Length;

        if (print)
        {
            Debug.Log(str);
        }
        errorOverTime.Add(avgError);
    }

    public void printErrorOverTime()
    {
        string str = "";
        for(int i = 0; i < this.errorOverTime.Count; i++)
        {
            str += i + " " + this.errorOverTime[i] + "\n";
        }
        Debug.Log(str);
    }
}
