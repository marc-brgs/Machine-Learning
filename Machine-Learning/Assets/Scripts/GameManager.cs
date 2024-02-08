using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    public App trainApp;
    public UseApp useApp;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void clickTest()
    {

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
