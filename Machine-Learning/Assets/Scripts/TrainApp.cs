using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Unity.PlasticSCM.Editor.WebApi;
using Unity.VisualScripting;
using UnityEngine;

public class TrainApp : MonoBehaviour
{
    private PerceptronWrapper p;
    public int epochs = 100000;
    public int epochCluster = 10; // Higher = faster processing but less fps
    public int printInterval = 1000;
    public double learningRate = 0.05f;
    public int inputLayerSize = 3888;
    public List<int> hiddenLayerSizes = new List<int> { 100, 20, 10 };
    public int outputLayerSize = 3;
    public bool matchImagesCountPerGameWithEpoch = true;
    public bool matchImagesIndexAsRandom = false;
    public int maxImageIndexOfTrainingData = 13000;
    public int imageIndexToTest = 13001;
    public int imagesCountToTest = 10;
    public bool measureSample = false;
    public int measureEpochInterval = 10;
    

    public string perceptronFilePath = "./Assets/Serialized Perceptrons/qty_images_unknown_100_20_10.dat";
    public bool loadPerceptron = false;
    public bool savePerceptron = true;
    public bool savePerceptronOnInterrupt = false;


    private const string DATASET_TRAIN_PATH = "./Assets/Dataset/Train";
    private double[] targetOutputRL = new double[] { 1, 0, 0 };
    private double[] targetOutputCS = new double[] { 0, 1, 0 };
    private double[] targetOutputDS = new double[] { 0, 0, 1 };

    private bool isTraining = false;
    private DateTime startTime;
    private int epoch = 0;
    private List<List<double>> errorMargins = new List<List<double>>();


    private enum Game
    {
        RocketLeague,
        CounterStrike,
        DarkSouls
    }

    // Update is called once per frame
    void Update()
    {
        InterruptTraining();
        
        if (isTraining)
        {
            if (epoch < epochs)
            {
                for(int i = 0; i < epochCluster; i++) {
                    if (epoch >= epochs) break;

                    if(matchImagesIndexAsRandom)
                    {
                        trainTestSample(UnityEngine.Random.Range(1, maxImageIndexOfTrainingData+1));
                    }
                    else if(matchImagesCountPerGameWithEpoch)
                    {
                        trainTestSample((epoch % maxImageIndexOfTrainingData) + 1);
                    }
                    else
                    {
                        Debug.Log("Index definition method of training images is not defined (check matchImagesIndexAsRandom or matchImagesCountPerGameWithEpoch)");
                    }

                    // Costly (increase measureEpochInterval to speed up training)
                    if (measureSample && (epoch % measureEpochInterval == 0))
                    {
                        measureTestSample();
                    }
                    epoch++;

                    // Debug current step
                    if (epoch % printInterval == 0)
                    {
                        Debug.Log(epoch + " / " + epochs);
                    }
                }
            }
            else
            {
                
                isTraining = false;
                TimeSpan trainingDuration = DateTime.UtcNow - startTime;
                Debug.Log("Training finished (duration : "+ trainingDuration + ")");

                if(savePerceptron)
                {
                    p.saveToFile(perceptronFilePath);
                }
                
                predictTestSample();
                measureTestSample();
                printPythonPlotScript();
            }
        }
    }

    public void StartTraining()
    {
        p = new PerceptronWrapper(inputLayerSize, hiddenLayerSizes.ToArray(), outputLayerSize);
        p.learningRate = learningRate;
        if (loadPerceptron)
        {
            p.loadFromFile(perceptronFilePath);
        }

        predictTestSample();
        startTime = DateTime.UtcNow;

        // Init error margins
        for (int i = 0; i < 3; i++)
        {
            errorMargins.Add(new List<double>());
        }

        isTraining = true;
    }

    private void InterruptTraining()
    {
        if (Input.GetKeyUp(KeyCode.Space))
        {
            if (isTraining)
            {
                isTraining = false;
                Debug.Log("Training interrupted");

                if (savePerceptronOnInterrupt)
                {
                    p.saveToFile(perceptronFilePath);
                }

                predictTestSample();
                printPythonPlotScript();
            }
            else
            {
                isTraining = true;
                Debug.Log("Training resumed");
            }
        }
    }

    public void predictTestSample()
    {
        double[] averageRL = new double[3] { 0, 0, 0 };
        double[] averageCS = new double[3] { 0, 0, 0 };
        double[] averageDS = new double[3] { 0, 0, 0 };

        for (int i = imageIndexToTest; i < imageIndexToTest + imagesCountToTest; i++)
        {
            double[] outputRL = p.predict(ImageLoader.LoadImagePixels(DATASET_TRAIN_PATH + "/Rocket League/RL-image-" + i + ".jpg"));
            double[] outputCS = p.predict(ImageLoader.LoadImagePixels(DATASET_TRAIN_PATH + "/Counter Strike/CS-image-" + i + ".jpg"));
            double[] outputDS = p.predict(ImageLoader.LoadImagePixels(DATASET_TRAIN_PATH + "/Dark Souls/DS-image-" + i + ".jpg"));

            for(int j = 0; j < 3; j++)
            {
                averageRL[j] += outputRL[j];
                averageCS[j] += outputCS[j];
                averageDS[j] += outputDS[j];
            }
            
        }

        for(int i = 0; i < 3; i++)
        {
            averageRL[i] /= imagesCountToTest;
            averageCS[i] /= imagesCountToTest;
            averageDS[i] /= imagesCountToTest;
        }
        

        Debug.Log("Test d'image Rocket League (proba.) : (RL : " + averageRL[0] + ", CS: " + averageRL[1] + ", DS: " + averageRL[2] + ")");
        Debug.Log("Test d'image Counter Strike (proba.) : (RL : " + averageCS[0] + ", CS: " + averageCS[1] + ", DS: " + averageCS[2] + ")");
        Debug.Log("Test d'image Dark Souls (proba.) : (RL : " + averageDS[0] + ", CS: " + averageDS[1] + ", DS: " + averageDS[2] + ")");
    }

    private void trainTestSample(int index)
    {
        double[] inputsRL = ImageLoader.LoadImagePixels(DATASET_TRAIN_PATH + "/Rocket League/RL-image-" + index + ".jpg");
        double[] inputsCS = ImageLoader.LoadImagePixels(DATASET_TRAIN_PATH + "/Counter Strike/CS-image-" + index + ".jpg");
        double[] inputsDS = ImageLoader.LoadImagePixels(DATASET_TRAIN_PATH + "/Dark Souls/DS-image-" + index + ".jpg");

        p.train(inputsRL, targetOutputRL);
        p.train(inputsCS, targetOutputCS);
        p.train(inputsDS, targetOutputDS);
    }

    public void measureTestSample()
    {
        double[] averageOutputRL = new double[3] { 0, 0, 0 };
        double[] averageOutputCS = new double[3] { 0, 0, 0 };
        double[] averageOutputDS = new double[3] { 0, 0, 0 };

        for (int i = imageIndexToTest; i < imageIndexToTest + imagesCountToTest; i++)
        {
            double[] outputRL = p.predict(ImageLoader.LoadImagePixels(DATASET_TRAIN_PATH + "/Rocket League/RL-image-" + i + ".jpg"));
            double[] outputCS = p.predict(ImageLoader.LoadImagePixels(DATASET_TRAIN_PATH + "/Counter Strike/CS-image-" + i + ".jpg"));
            double[] outputDS = p.predict(ImageLoader.LoadImagePixels(DATASET_TRAIN_PATH + "/Dark Souls/DS-image-" + i + ".jpg"));

            
            for(int j = 0; j < 3; j++)
            {
                averageOutputRL[j] += outputRL[j];
                averageOutputCS[j] += outputCS[j];
                averageOutputDS[j] += outputDS[j];
            }
        }

        // Output average
        for (int i = 0; i < 3; i++)
        {
            averageOutputRL[i] /= imagesCountToTest;
            averageOutputCS[i] /= imagesCountToTest;
            averageOutputDS[i] /= imagesCountToTest;
        }

        // Enregistrement des erreurs moyennes
        registerErrorAverage(Game.RocketLeague, averageOutputRL);
        registerErrorAverage(Game.CounterStrike, averageOutputCS);
        registerErrorAverage(Game.DarkSouls, averageOutputDS);
    }

    private void registerErrorAverage(Game game, double[] currentOutput)
    {
        if (game == Game.RocketLeague)
        {
            errorMargins[0].Add(computeErrorAverage(targetOutputRL, currentOutput));
        }
        else if (game == Game.CounterStrike)
        {
            errorMargins[1].Add(computeErrorAverage(targetOutputCS, currentOutput));
        }
        else if (game == Game.DarkSouls)
        {
            errorMargins[2].Add(computeErrorAverage(targetOutputDS, currentOutput));
        }
    }

    private double computeErrorAverage(double[] target, double[] current)
    {
        double avg = 0;
        for (int i = 0; i < target.Length; i++)
        {
            avg += Math.Abs(target[i] - current[i]);
        }
        avg /= target.Length;

        return avg;
    }

    public void printPythonPlotScript()
    {
        string index = string.Empty;
        string valueRL = string.Empty;
        string valueCS = string.Empty;
        string valueDS = string.Empty;
        for (int i = 0; i < this.errorMargins[0].Count; i++)
        {
            index += i * measureEpochInterval;
            valueRL += this.errorMargins[0][i].ToString("F5").Replace(",", ".");
            valueCS += this.errorMargins[1][i].ToString("F5").Replace(",", ".");
            valueDS += this.errorMargins[2][i].ToString("F5").Replace(",", ".");

            // Separate values
            if (i != this.errorMargins[0].Count - 1)
            {
                index += ",";
                valueRL += ",";
                valueCS += ",";
                valueDS += ",";
            }
        }

        string script =
            "import matplotlib.pyplot as plt\n" +
            "import numpy as np\n" +
            "index = [" + index + "]\n" +
            "valueRL = [" + valueRL + "]\n" +
            "valueCS = [" + valueCS + "]\n" +
            "valueDS = [" + valueDS + "]\n" +
            "plt.plot(index, valueRL, label=\"Rocket League\")\n" +
            "plt.plot(index, valueCS, label=\"Counter Strike\")\n" +
            "plt.plot(index, valueDS, label=\"Dark Souls\")\n" +
            "plt.title(\"Image prediction truth (x"+ imagesCountToTest +")\")\n" +
            "plt.xlabel(\"Epoch\")\n" +
            "plt.ylabel(\"Outputs error margin (average)\")\n" +
            "plt.legend(loc=\"upper right\")\n" +
            "plt.grid()\n" +
            "plt.show()\n";

        Debug.Log(script);
    }
}
