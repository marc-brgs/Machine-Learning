using System.Collections;
using System.Collections.Generic;
using System.IO;
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

        Texture2D tex = LoadPNG("./Assets/Dataset/Train/Rocket League/RL1Image-000.jpg");
        PixelLoader(tex);
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
                    //p.train(new double[] { 0, 0 }, new double[] { 0 });
                    //p.train(new double[] { 0, 1 }, new double[] { 1 });
                    //p.train(new double[] { 1, 0 }, new double[] { 1 });
                    //p.train(new double[] { 1, 1 }, new double[] { 0 });
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

    public static Texture2D LoadPNG(string filePath)
    {

        Texture2D tex = null;
        byte[] fileData;

        if (File.Exists(filePath))
        {
            fileData = File.ReadAllBytes(filePath);
            tex = new Texture2D(48, 27); // 48x27 pixels (1296 pixels)
            tex.LoadImage(fileData); //..this will auto-resize the texture dimensions
        }
        return tex;
    }

    public double[] PixelLoader(Texture2D tex)
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
        Debug.Log(myList.Count);
        return myList.ToArray();
    }
}
