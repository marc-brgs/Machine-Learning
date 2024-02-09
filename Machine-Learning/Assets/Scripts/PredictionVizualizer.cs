using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class PredictionVizualizer : MonoBehaviour
{
    public GameObject pointsParent;
    public GameObject pointPrefab;
    public PerceptronWrapper perceptron;
    public int width = 100;
    public int height = 100;

    public void VisualizePredictions(PerceptronWrapper p, int outputCount = 1)
    {
        for (float x = 0f; x <= 1.0f; x += (1f/width))
        {
            for (float y = 0f; y <= 1.0f; y += (1f/height))
            {
                double[] prediction = p.predict(new double[] { x, y });

                // Create point
                GameObject point = Instantiate(pointPrefab, new Vector3(x - 0.5f, y - 0.5f, 0), Quaternion.identity);
                point.transform.parent = pointsParent.transform;

                // Colorize point
                if(outputCount == 1)
                {
                    Color pointColor = prediction[0] > 0.5 ? Color.blue : Color.red;
                    point.GetComponent<Renderer>().material.color = pointColor;
                }
                else if(outputCount == 3)
                {
                    int predictedClass = Array.IndexOf(prediction, prediction.Max());

                    Color pointColor = Color.black;
                    switch (predictedClass)
                    {
                        case 0:
                            pointColor = Color.blue;
                            break;
                        case 1:
                            pointColor = Color.red;
                            break;
                        case 2:
                            pointColor = Color.green;
                            break;
                    }
                    point.GetComponent<Renderer>().material.color = pointColor;
                }
            }
        }
    }

    public void ClearPoints()
    {
        foreach (Transform child in pointsParent.transform)
        {
            Destroy(child.gameObject);
        }
    }

    public void VisualizeData(double[,] X, double[] Y)
    {
        for(int i = 0; i < 500; i++)
        {
            GameObject point = Instantiate(pointPrefab, new Vector3((float)X[i, 0], (float)X[i, 1], 0), Quaternion.identity);
            point.transform.parent = this.transform; // Organiser sous le visualiseur

            // Colorer le point en fonction de la prédiction
            Color pointColor = Y[i] > 0.5 ? Color.blue : Color.red;
            point.GetComponent<Renderer>().material.color = pointColor;
        }
    }
}