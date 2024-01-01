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

        p.train(new double[] { 0 }, new double[] { 0 }, 0); // test
        p.predict(new double[] { 0 }, new double[] { 0 }); // test
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
