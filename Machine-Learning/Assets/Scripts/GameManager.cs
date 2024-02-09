using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    public TestApp testApp;
    public TrainApp trainApp;
    public UseApp useApp;

    public void clickTest()
    {
        testApp.StartTestApp();
    }

    public void clickUse()
    {
        useApp.StartUseApp();
    }

    public void clickTrain()
    {
        trainApp.StartTraining();
    }
}
