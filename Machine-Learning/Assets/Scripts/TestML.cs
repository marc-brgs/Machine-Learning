using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using System;

public class TestML : MonoBehaviour
{
    public PredictionVizualizer predictionVizualizer;

    // Start is called before the first frame update
    void Start()
    {
        //AND(100000, 0.02);
        //XOR(200000, 0.01);
        Cross(1500, 0.1);
    }

    void AND(int epochs, double learningRate)
    {
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { }, 1);
        p.learningRate = learningRate;

        for (int i = 0; i < epochs; i++)
        {
            p.train(new double[] { 0, 0 }, new double[] { 0 });
            p.train(new double[] { 0, 1 }, new double[] { 0 });
            p.train(new double[] { 1, 0 }, new double[] { 0 });
            p.train(new double[] { 1, 1 }, new double[] { 1 });
        }

        // Afficher la sortie
        p.predict(new double[] { 0, 0 });
        p.predict(new double[] { 0, 1 });
        p.predict(new double[] { 1, 0 });
        p.predict(new double[] { 1, 1 });
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

        for (int i = 0; i < epochs; i++)
        {
            p.train(new double[] { 0, 0 }, new double[] { 0 });
            p.train(new double[] { 0, 1 }, new double[] { 1 });
            p.train(new double[] { 1, 0 }, new double[] { 1 });
            p.train(new double[] { 1, 1 }, new double[] { 0 });
        }

        p.predict(new double[] { 0, 0 });
        p.predict(new double[] { 0, 1 });
        p.predict(new double[] { 1, 0 });
        p.predict(new double[] { 1, 1 });
    }

    void Cross(int epochs, double learningRate)
    {
        // Linear Model : KO
        // MLP (2, 4, 1) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 4 }, 1);
        p.learningRate = learningRate;

        // Génération des données X
        int sampleSize = 500; // Précision de la forme (croix)
        double[,] X = new double[sampleSize, 2];

        System.Random rand = new System.Random();
        for (int i = 0; i < sampleSize; i++)
        {
            X[i, 0] = rand.NextDouble() * 2.0 - 1.0; // Random entre -1.0 et 1.0
            X[i, 1] = rand.NextDouble() * 2.0 - 1.0; // Random entre -1.0 et 1.0
        }

        // Définition de Y (résultat souhaité)
        int[] Y = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++)
        {
            Y[i] = (Math.Abs(X[i, 0]) <= 0.3 || Math.Abs(X[i, 1]) <= 0.3) ? 1 : -1;
        }

        // Entrainement
        for (int e = 0; e < epochs; e++)
        {
            for (int i = 0; i < sampleSize; i++)
            {
                p.train(new double[] { X[i, 0], X[i, 1] }, new double[] { Y[i] });
            }
        }

        // Prédictions
        double[] Z = new double[sampleSize];
        for(int i = 0; i < sampleSize; i++)
        {
            Z[i] = p.predict(new double[] { X[i, 0], X[i, 1] })[0];
        }

        predictionVizualizer.VisualizeData(X, Z);
        //predictionVizualizer.VisualizePredictions(p);
    }

    private GameObject Instantiate(object pointPrefab, Vector3 vector3, Quaternion identity)
    {
        throw new NotImplementedException();
    }

    void MultiLinear3Classes()
    {
        // Linear Model : OK
        // MLP (2, 3) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 0 }, 3);
    }

    void MultiCross()
    {
        // Linear Model : OK
        // MLP (2, ?, ?, 3) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 0 }, 3);
    }
}
