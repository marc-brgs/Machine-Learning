using System.Collections;
using System.Collections.Generic;
using System.IO;
using Unity.VisualScripting;
using UnityEngine;

public class App : MonoBehaviour
{
    PerceptronWrapper p;
    public int epochs = 100000;
    public int epochCluster = 100; // Higher = faster processing but less fps
    public int printInterval = 1000;
    public double learningRate = 0.05f;
    public int inputLayerSize = 3888;
    public List<int> hiddenLayerSizes = new List<int> { 100, 20, 10 };
    public int outputLayerSize = 3;
    public int imagesPerGame = 3;
    public int imageIndexToTest = 5;

    private bool isTraining = true;
    private int epoch = 0;

    string[] RLImagesPath;
    string[] CSImagesPath;
    string[] DSImagesPath;

    float startTraining;

    // Start is called before the first frame update
    void Start()
    {
        p = new PerceptronWrapper(inputLayerSize, hiddenLayerSizes.ToArray(), outputLayerSize);
        p.learningRate = learningRate;

        //RLImagesPath = ReadFolder("./Assets/Dataset/Train/Rocket League");
        //CSImagesPath = ReadFolder("./Assets/Dataset/Train/Counter Strike");
        //DSImagesPath = ReadFolder("./Assets/Dataset/Train/Dark Souls");

        testPredictWithSample();
        startTraining = Time.time;
    }

    public void testPredictWithSample(bool measure=false)
    {
        double[] output = p.predict(PixelLoader(LoadPNG("./Assets/Dataset/Train/Rocket League/RL1Image-00"+ imageIndexToTest +".jpg")));

        if (measure)
        {
            p.getOutputError();
        }
        else
        {
            double[] output2 = p.predict(PixelLoader(LoadPNG("./Assets/Dataset/Train/Counter Strike/CS1Image-00"+ imageIndexToTest + ".jpg")));
            double[] output3 = p.predict(PixelLoader(LoadPNG("./Assets/Dataset/Train/Dark Souls/DSP2Image-00"+ imageIndexToTest +".jpg")));
            Debug.Log("Test d'image Rocket League (probabilités) : (RL : " + output[0] + ", CS : " + output[1] + ", DS : " + output[2] + ")");
            Debug.Log("Test d'image Counter Strike (probabilités) : (RL : " + output2[0] + ", CS : " + output2[1] + ", DS : " + output2[2] + ")");
            Debug.Log("Test d'image Dark Souls (probabilités) : (RL : " + output3[0] + ", CS : " + output3[1] + ", DS : " + output3[2] + ")");
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

                    for(int j = 0; j < imagesPerGame; j++)
                    {
                        //Texture2D texRL = LoadPNG(RLImagesPath[i]);
                        Texture2D texRL = LoadPNG("./Assets/Dataset/Train/Rocket League/RL1Image-00"+ j + ".jpg");
                        double[] inputsRL = PixelLoader(texRL);
                        double[] outputsRL = new double[] { 1, 0, 0 };

                        //Texture2D texCS = LoadPNG(CSImagesPath[i]);
                        Texture2D texCS = LoadPNG("./Assets/Dataset/Train/Counter Strike/CS1Image-00"+ j +".jpg");
                        double[] inputsCS = PixelLoader(texCS);
                        double[] outputsCS = new double[] { 0, 1, 0 };

                        //Texture2D texDS = LoadPNG(DSImagesPath[i]);
                        Texture2D texDS = LoadPNG("./Assets/Dataset/Train/Dark Souls/DSP2Image-00"+ j +".jpg");
                        double[] inputsDS = PixelLoader(texDS);
                        double[] outputsDS = new double[] { 0, 0, 1 };

                        p.train(inputsRL, outputsRL);
                        p.train(inputsCS, outputsCS);
                        p.train(inputsDS, outputsDS);
                    }

                    testPredictWithSample(true);
                    epoch++;

                    // Current step
                    if (epoch % printInterval == 0)
                    {
                        Debug.Log(epoch + " / " + epochs);
                    }
                }
            }
            else
            {
                isTraining = false;
                float trainingDuration = Time.time - startTraining;
                Debug.Log("Training finished (duration : "+ trainingDuration + "s");

                testPredictWithSample();
                p.printErrorOverTime();
                p.printPythonPlotScript();
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
        //Debug.Log(myList.Count);
        return myList.ToArray();
    }
    
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
