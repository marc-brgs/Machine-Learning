using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using System;
using TMPro;
using System.Drawing;
using System.IO;
using System.Linq;

public class TestApp : MonoBehaviour
{
    public PredictionVizualizer predictionVizualizer;
    public bool usePreTrainedModel = true;
    public string[] modelPaths;

    // UI objects
    [SerializeField] private GameObject mainCanvas;
    [SerializeField] private GameObject testCanvas;
    [SerializeField] private TextMeshProUGUI timespanText;

    private DateTime startTime;

    public enum TestCase
    {
        AND = 0,
        XOR = 1,
        CROSS = 2,
        ML3C = 3,
        MCROSS = 4
    }

    public void StartTestApp()
    {
        mainCanvas.SetActive(false);
        testCanvas.SetActive(true);
        timespanText.gameObject.SetActive(false);
    }

    public void ExitTestApp()
    {
        mainCanvas.SetActive(true);
        testCanvas.SetActive(false);
        timespanText.gameObject.SetActive(false);
        predictionVizualizer.ClearPoints();
    }

    public void TrainOnTestCase(int testIndex)
    {
        predictionVizualizer.ClearPoints();
        startTime = DateTime.UtcNow;

        if (testIndex == (int)TestCase.AND)
        {
            AND(100000, 0.02);
        }
        else if(testIndex == (int)TestCase.XOR)
        {
            XOR(200000, 0.01);
        }
        else if (testIndex == (int)TestCase.CROSS)
        {
            Cross(2000, 0.05, 500);
        }
        else if (testIndex == (int)TestCase.ML3C)
        {
            MultiLinear3Classes(500, 0.05, 500);
        }
        else if(testIndex == (int)TestCase.MCROSS)
        {
            MultiCross(2000, 0.1, 200);
        }

        // Time measure
        TimeSpan timespan = DateTime.UtcNow - startTime;
        timespanText.text = "DURATION : " + timespan.TotalSeconds.ToString("F3").Replace(",", ".") + "s";
        timespanText.gameObject.SetActive(true);
    }

    void AND(int epochs, double learningRate)
    {
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { }, 1);
        p.learningRate = learningRate;

        if (usePreTrainedModel && DoesModelExists(TestCase.AND))
        {
            string modelPath = modelPaths[(int)TestCase.AND];
            p.loadFromFile(modelPath);
            p.predict(new double[] { 0, 0 }, true);
            p.predict(new double[] { 0, 1 }, true);
            p.predict(new double[] { 1, 0 }, true);
            p.predict(new double[] { 1, 1 }, true);
            return;
        }

        for (int i = 0; i < epochs; i++)
        {
            p.train(new double[] { 0, 0 }, new double[] { 0 });
            p.train(new double[] { 0, 1 }, new double[] { 0 });
            p.train(new double[] { 1, 0 }, new double[] { 0 });
            p.train(new double[] { 1, 1 }, new double[] { 1 });
        }

        // Afficher la sortie
        p.predict(new double[] { 0, 0 }, true);
        p.predict(new double[] { 0, 1 }, true);
        p.predict(new double[] { 1, 0 }, true);
        p.predict(new double[] { 1, 1 }, true);
    }

    void LinearSimple()
    {
        // Linear Model : OK
        // MLP (2, 1) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { }, 1);
    }

    void LinearMultiple()
    {
        // Linear Model : OK
        // MLP (2, 1) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { }, 1);
    }

    void XOR(int epochs, double learningRate)
    {
        // Linear Model : KO
        // MLP (2, 2, 1) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 2 }, 1);
        p.learningRate = learningRate;

        if (usePreTrainedModel && DoesModelExists(TestCase.XOR))
        {
            string modelPath = modelPaths[(int)TestCase.XOR];
            p.loadFromFile(modelPath);
            p.predict(new double[] { 0, 0 }, true);
            p.predict(new double[] { 0, 1 }, true);
            p.predict(new double[] { 1, 0 }, true);
            p.predict(new double[] { 1, 1 }, true);
            return;
        }

            for (int i = 0; i < epochs; i++)
        {
            p.train(new double[] { 0, 0 }, new double[] { 0 });
            p.train(new double[] { 0, 1 }, new double[] { 1 });
            p.train(new double[] { 1, 0 }, new double[] { 1 });
            p.train(new double[] { 1, 1 }, new double[] { 0 });
        }

        p.predict(new double[] { 0, 0 }, true);
        p.predict(new double[] { 0, 1 }, true);
        p.predict(new double[] { 1, 0 }, true);
        p.predict(new double[] { 1, 1 }, true);
    }

    void Cross(int epochs, double learningRate, int sampleSize)
    {
        // Linear Model x3 : KO
        // MLP (2, 4, 1) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 4 }, 1);
        p.learningRate = learningRate;

        if(usePreTrainedModel && DoesModelExists(TestCase.CROSS)) {
            string modelPath = modelPaths[(int)TestCase.CROSS];
            p.loadFromFile(modelPath);
            predictionVizualizer.VisualizePredictions(p);
            return;
        }

        // Génération des points
        MLPoint[] points = MLPoint.GenerateRandomPoints(sampleSize);

        // Target values
        int[] target = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++)
        {
            // Ajustement à [-1, 1]
            double x = points[i].x * 2 - 1;
            double y = points[i].y * 2 - 1;

            if (Math.Abs(x) <= 0.3 || Math.Abs(y) <= 0.3)
            {
                target[i] = 1;
            }
            else
            {
                target[i] = 0;
            }
        }

        // Entrainement
        for (int e = 0; e < epochs; e++)
        {
            for (int i = 0; i < sampleSize; i++)
            {
                p.train(new double[] { points[i].x, points[i].y }, new double[] { target[i] });
            }
        }

        predictionVizualizer.VisualizePredictions(p);
    }

    void MultiLinear3Classes(int epochs, double learningRate, int sampleSize)
    {
        // Linear Model : OK
        // MLP (2, 3) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { }, 3);
        p.learningRate = learningRate;

        if (usePreTrainedModel && DoesModelExists(TestCase.ML3C))
        {
            string modelPath = modelPaths[(int)TestCase.ML3C];
            p.loadFromFile(modelPath);
            predictionVizualizer.VisualizePredictions(p, 3);
            return;
        }

        // Génération des points
        MLPoint[] points = MLPoint.GenerateRandomPoints(sampleSize);
        List<double[]> trainingInputs = new List<double[]>();
        List<double[]> trainingOutputs = new List<double[]>();

        for (int i = 0; i < sampleSize; i++)
        {
            // Ajustement à [-1, 1]
            double x = points[i].x * 2 - 1; 
            double y = points[i].y * 2 - 1;
            
            double[] output;
            if (-x - y - 0.5 > 0 && y < 0 && x - y - 0.5 < 0)
            {
                output = new double[] { 1, 0, 0 }; // Bleu
            }
            else if (-x - y - 0.5 < 0 && y > 0 && x - y - 0.5 < 0)
            {
                output = new double[] { 0, 1, 0 }; // Rouge
            }
            else if (-x - y - 0.5 < 0 && y < 0 && x - y - 0.5 > 0)
            {
                output = new double[] { 0, 0, 1 }; // Vert
            }
            else
            {
                continue;
            }

            trainingInputs.Add(new double[] { points[i].x, points[i].y });
            trainingOutputs.Add(output);
        }

        // Entraînement
        for (int e = 0; e < epochs; e++)
        {
            for (int i = 0; i < trainingInputs.Count; i++)
            {
                p.train(trainingInputs[i], trainingOutputs[i]);
            }
        }

        predictionVizualizer.VisualizePredictions(p, 3);
    }

    void MultiCross(int epochs, double learningRate, int sampleSize)
    {
        // Linear Model x3 : KO
        // MLP (2, ?, ?, 3) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 8, 8 }, 3);
        p.learningRate = learningRate;

        if (usePreTrainedModel && DoesModelExists(TestCase.MCROSS))
        {
            string modelPath = modelPaths[(int)TestCase.MCROSS];
            p.loadFromFile(modelPath);
            predictionVizualizer.VisualizePredictions(p, 3);
            return;
        }

        // Génération des points
        MLPoint[] points = MLPoint.GenerateRandomPoints(sampleSize);
        List<double[]> trainingInputs = new List<double[]>();
        List<double[]> trainingOutputs = new List<double[]>();

        for (int i = 0; i < sampleSize; i++)
        {
            // Ajustement à [-1, 1]
            double x = points[i].x * 2 - 1; 
            double y = points[i].y * 2 - 1;

            double[] output;
            if (Math.Abs(x % 0.5) <= 0.25 && Math.Abs(y % 0.5) > 0.25)
            {
                output = new double[] { 1, 0, 0 }; // Bleu
            }
            else if (Math.Abs(x % 0.5) > 0.25 && Math.Abs(y % 0.5) <= 0.25)
            {
                output = new double[] { 0, 1, 0 }; // Rouge
            }
            else
            {
                output = new double[] { 0, 0, 1 }; // Vert
            }

            trainingInputs.Add(new double[] { points[i].x, points[i].y });
            trainingOutputs.Add(output);
        }
        
        // Entraînement
        for (int e = 0; e < epochs; e++)
        {
            for (int i = 0; i < trainingInputs.Count; i++)
            {
                p.train(trainingInputs[i], trainingOutputs[i]);
            }
        }
        
        predictionVizualizer.VisualizePredictions(p, 3);
        p.saveToFile("./Assets/mcross.dat");
    }

    private bool DoesModelExists(TestCase testCase)
    {
        if (modelPaths.Length > (int)testCase && File.Exists(modelPaths[(int)testCase]))
        {
            return true;
        }
        return false;
    }
}
