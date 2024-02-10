using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

public class UseApp : MonoBehaviour
{
    public PerceptronWrapper p;
    public string modelPath = "./Assets/Serialized Perceptrons/<name>.dat";
    public int minRandomImageIndex = 13001;
    public int maxRandomImageIndex = 13400;
    public Color defaultResultColor = new Color(0.95f, 0.95f, 0.95f, 1f);
    public Color bestResultColor = new Color(1f, 1f, 0.55f, 1f);

    public int inputLayerSize = 3888;
    public List<int> hiddenLayerSizes = new List<int> { 100, 20, 10 };
    public int outputLayerSize = 3;
    
    private const string DATASET_PATH = "./Assets/Dataset/Train";
    private bool modelLoaded = false;

    // Scene elements
    [SerializeField] private GameObject mainMenu;
    [SerializeField] private GameObject useMenu;
    [SerializeField] private RawImage rawImage;
    [SerializeField] private GameObject resultsPanel;
    [SerializeField] private TextMeshProUGUI[] probTexts;

    private enum Game
    {
        RocketLeague,
        CounterStrike,
        DarkSouls
    }

    public void StartUseApp()
    {
        p = new PerceptronWrapper(inputLayerSize, hiddenLayerSizes.ToArray(), outputLayerSize);

        if(File.Exists(modelPath)) {
            p.loadFromFile(modelPath);
            modelLoaded = true;

            mainMenu.SetActive(false);
            useMenu.SetActive(true);
        }
        else
        {
            Debug.Log("Failed to load model, path does not exists : \""+ modelPath +"\"");
        }
    }

    public void ExitUseApp()
    {
        mainMenu.SetActive(true);
        useMenu.SetActive(false);
    }

    public void RandomPredictRL()
    {
        RandomPredict(Game.RocketLeague);
    }

    public void RandomPredictCS()
    {
        RandomPredict(Game.CounterStrike);
    }

    public void RandomPredictDS()
    {
        RandomPredict(Game.DarkSouls);
    }

    private void RandomPredict(Game game)
    {
        if (!modelLoaded) return;

        int randomIndex = UnityEngine.Random.Range(minRandomImageIndex, maxRandomImageIndex + 1);
        string imagePath = string.Empty;

        if(game == Game.RocketLeague)
        {
            imagePath = DATASET_PATH + "/Rocket League/RL-image-" + randomIndex + ".jpg";
        }
        else if(game == Game.CounterStrike)
        {
            imagePath = DATASET_PATH + "/Counter Strike/CS-image-" + randomIndex + ".jpg";
        }
        else
        {
            imagePath = DATASET_PATH + "/Dark Souls/DS-image-" + randomIndex + ".jpg";
        }

        Texture2D tex = ImageLoader.LoadTexture(imagePath);
        rawImage.texture = tex;

        // Predict
        double[] outputs = p.predict(ImageLoader.LoadImagePixels(imagePath));
        List<double> outputsList = outputs.ToList();
        int bestIndex = outputsList.IndexOf(outputsList.Max());

        for (int i = 0; i < probTexts.Length; i++)
        {
            probTexts[i].text = (outputs[i] * 100).ToString("F1") + "%";

            if(i == bestIndex)
            {
                probTexts[i].color = bestResultColor;
            }
            else
            {
                probTexts[i].color = defaultResultColor;
            }
        }
        resultsPanel.SetActive(true);
    }
}
