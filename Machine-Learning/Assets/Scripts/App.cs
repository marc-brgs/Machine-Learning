using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Unity.PlasticSCM.Editor.WebApi;
using Unity.VisualScripting;
using UnityEngine;

public class App : MonoBehaviour
{
    private PerceptronWrapper p;
    public int epochs = 100000;
    public int epochCluster = 100; // Higher = faster processing but less fps
    public int printInterval = 1000;
    public double learningRate = 0.05f;
    public int inputLayerSize = 3888;
    public List<int> hiddenLayerSizes = new List<int> { 100, 20, 10 };
    public int outputLayerSize = 3;
    public int imagesCountPerGame = 1;
    public bool matchImagesCountPerGameWithEpoch = true;
    public bool matchImagesIndexAsRandom = false;
    public int imageIndexToTest = 1;
    public bool measureSample = false;
    

    public string perceptronFilePath = "./Assets/Serialized Perceptrons/qty_images_unknown_100_20_10.dat";
    public bool loadPerceptron = false;
    public bool savePerceptron = true;
    public bool savePerceptronOnInterrupt = false;
    

    private const string DATASET_TRAIN_PATH = "./Assets/Dataset/Train";
    private double[] targetOutputRL = new double[] { 1, 0, 0 };
    private double[] targetOutputCS = new double[] { 0, 1, 0 };
    private double[] targetOutputDS = new double[] { 0, 0, 1 };

    private bool isTraining = true;
    private DateTime startTime;
    private int epoch = 0;
    private List<List<double>> errorMargins = new List<List<double>>();


    private enum Game
    {
        RocketLeague,
        CounterStrike,
        DarkSouls
    }

    // Start is called before the first frame update
    void Start()
    {
        p = new PerceptronWrapper(inputLayerSize, hiddenLayerSizes.ToArray(), outputLayerSize);
        p.learningRate = learningRate;
        if(loadPerceptron)
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
                        trainTestSample(UnityEngine.Random.Range(1, 10000));
                    }
                    else if(matchImagesCountPerGameWithEpoch)
                    {
                        trainTestSample(epoch + 1);
                    }
                    else
                    {
                        for(int j = 1; j < imagesCountPerGame + 1; j++)
                        {
                            trainTestSample(j);
                        }
                    }

                    // Set to 1 FPS if activated (looks like predict is costly)
                    if (measureSample)
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
                Debug.Log("Training finished (duration : "+ trainingDuration + "s");

                if(savePerceptron)
                {
                    p.saveToFile(perceptronFilePath);
                }
                
                predictTestSample();
                printPythonPlotScript();
            }
        }
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
        double[] outputRL = p.predict(LoadImagePixels(DATASET_TRAIN_PATH + "/Rocket League/RL-image-" + imageIndexToTest + ".jpg"));
        double[] outputCS = p.predict(LoadImagePixels(DATASET_TRAIN_PATH + "/Counter Strike/CS-image-" + imageIndexToTest + ".jpg"));
        double[] outputDS = p.predict(LoadImagePixels(DATASET_TRAIN_PATH + "/Dark Souls/DS-image-" + imageIndexToTest + ".jpg"));

        Debug.Log("Test d'image Rocket League (proba.) : (RL : " + outputRL[0] + ", CS: " + outputRL[1] + ", DS: " + outputRL[2] + ")");
        Debug.Log("Test d'image Counter Strike (proba.) : (RL : " + outputCS[0] + ", CS: " + outputCS[1] + ", DS: " + outputCS[2] + ")");
        Debug.Log("Test d'image Dark Souls (proba.) : (RL : " + outputDS[0] + ", CS: " + outputDS[1] + ", DS: " + outputDS[2] + ")");
    }

    private void trainTestSample(int index)
    {
        double[] inputsRL = LoadImagePixels(DATASET_TRAIN_PATH + "/Rocket League/RL-image-" + index + ".jpg");
        double[] inputsCS = LoadImagePixels(DATASET_TRAIN_PATH + "/Counter Strike/CS-image-" + index + ".jpg");
        double[] inputsDS = LoadImagePixels(DATASET_TRAIN_PATH + "/Dark Souls/DS-image-" + index + ".jpg");

        p.train(inputsRL, targetOutputRL);
        p.train(inputsCS, targetOutputCS);
        p.train(inputsDS, targetOutputDS);
    }

    public void measureTestSample()
    {
        double[] outputRL = p.predict(LoadImagePixels(DATASET_TRAIN_PATH + "/Rocket League/RL-image-" + imageIndexToTest + ".jpg"));
        registerErrorAverage(Game.RocketLeague, outputRL);

        double[] outputCS = p.predict(LoadImagePixels(DATASET_TRAIN_PATH + "/Counter Strike/CS-image-" + imageIndexToTest + ".jpg"));
        registerErrorAverage(Game.CounterStrike, outputCS);

        double[] outputDS = p.predict(LoadImagePixels(DATASET_TRAIN_PATH + "/Dark Souls/DS-image-" + imageIndexToTest + ".jpg"));
        registerErrorAverage(Game.DarkSouls, outputDS);
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

    public static Texture2D LoadTexture(string filePath)
    {
        Texture2D tex = null;
        byte[] fileData;

        if (File.Exists(filePath))
        {
            fileData = File.ReadAllBytes(filePath);
            tex = new Texture2D(48, 27); // 48x27 pixels (1296 pixels)
            tex.LoadImage(fileData); //..this will auto-resize the texture dimensions
        }
        else
        {
            Debug.Log("Path \""+ filePath +"\" does not exists!");
        }
        return tex;
    }

    public double[] LoadFlatPixels(Texture2D tex)
    {
        List<double> myList = new List<double>();
        for (int i = 0; i < tex.width; i++)
        {
            for (int j = 0; j < tex.height; j++)
            {
                Color pixel = tex.GetPixel(i, j);

                myList.Add(pixel.r);
                myList.Add(pixel.g);
                myList.Add(pixel.b);
            }
        }
        
        return myList.ToArray();
    }

    public double[] LoadImagePixels(string filePath)
    {
        Texture2D tex = LoadTexture(filePath);
        return LoadFlatPixels(tex);
    }

    public void printPythonPlotScript()
    {
        string index = string.Empty;
        string valueRL = string.Empty;
        string valueCS = string.Empty;
        string valueDS = string.Empty;
        for (int i = 0; i < this.errorMargins[0].Count; i++)
        {
            index += i;
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
            "plt.title(\"Image prediction truth (x1)\")\n" +
            "plt.xlabel(\"Epoch\")\n" +
            "plt.ylabel(\"Outputs error margin (average)\")\n" +
            "plt.legend(loc=\"upper right\")\n" +
            "plt.grid()\n" +
            "plt.show()\n";

        Debug.Log(script);
    }

    /**
     * Deprecated
     * List images path inside given directory
     */
    public string[] ReadFolder(string dir)
    {
        if (Directory.Exists(dir))
        {
            string[] files = Directory.GetFiles(dir, "*.jpg", SearchOption.AllDirectories);
            Debug.Log(dir + " : " + files.Length + " images");
            return files;
        }
        else
        {
            Debug.Log("Dossier non trouvé : " + dir);
            return new string[] { };
        }
    }
}
