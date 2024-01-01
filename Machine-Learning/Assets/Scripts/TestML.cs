using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;

public class TestML : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        PerceptronWrapper p = new PerceptronWrapper(1, new int[] { 2 }, 1);

        // p.train(new double[] { 0 }, new double[] { 0 }, 0); // test

        double[] input = new double[] { 0 };
        double[] output = new double[1];  // Assurez-vous que la taille est correcte pour la sortie attendue

        p.predict(input, ref output); // test

        // Afficher la sortie
        Debug.Log("Predicted output: " + output[0]);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
