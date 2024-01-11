using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class App : MonoBehaviour
{
    PerceptronWrapper p;
    public int epochs = 100000;
    public int epochCluster = 100; // Higher = faster processing but less fps
    public int printInterval = 1000;

    private bool isTraining = true;
    private int epoch = 0;

    // Start is called before the first frame update
    void Start()
    {
        p = new PerceptronWrapper(2, new int[] { 2 }, 1);
        p.learningRate = 0.05f;
    }

    // Update is called once per frame
    void Update()
    {
        InterruptTraining();

        if (isTraining)
        {
            if (epoch < epochs)
            {
                for(int i = 0; i < 100; i++) {
                    p.train(new double[] { 0, 0 }, new double[] { 0 });
                    p.train(new double[] { 0, 1 }, new double[] { 1 });
                    p.train(new double[] { 1, 0 }, new double[] { 1 });
                    p.train(new double[] { 1, 1 }, new double[] { 0 });
                    epoch++;
                }

                // Current step
                if (epoch % printInterval == 0)
                {
                    Debug.Log(epoch + " / " + epochs);
                }
            }
            else
            {
                isTraining = false;
                Debug.Log("Training finished");

                p.predict(new double[] { 0, 0 }, true);
                p.predict(new double[] { 0, 1 }, true);
                p.predict(new double[] { 1, 0 }, true);
                p.predict(new double[] { 1, 1 }, true);
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

                p.predict(new double[] { 0, 0 }, true);
                p.predict(new double[] { 0, 1 }, true);
                p.predict(new double[] { 1, 0 }, true);
                p.predict(new double[] { 1, 1 }, true);
            }
            else
            {
                isTraining = true;
                Debug.Log("Training resumed");
            }
        }
    }
}
