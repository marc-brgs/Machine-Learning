using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PredictionVizualizer : MonoBehaviour
{
    public GameObject pointPrefab;
    public PerceptronWrapper perceptron;

    public void VisualizePredictions(PerceptronWrapper p)
    {
        for (float x = -1.0f; x <= 1.0f; x += 0.1f)
        {
            for (float y = -1.0f; y <= 1.0f; y += 0.1f)
            {
                // Prédire le résultat
                double[] prediction = p.predict(new double[] { x, y });

                // Créer et positionner le point
                GameObject point = Instantiate(pointPrefab, new Vector3(x, y, 0), Quaternion.identity);
                point.transform.parent = this.transform; // Organiser sous le visualiseur

                // Colorer le point en fonction de la prédiction
                Color pointColor = prediction[0] > 0.5 ? Color.blue : Color.red;
                point.GetComponent<Renderer>().material.color = pointColor;
            }
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