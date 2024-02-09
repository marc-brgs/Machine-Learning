using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class MLPoint
{
    public double x;
    public double y;

    MLPoint(double x, double y)
    {
        this.x = x;
        this.y = y;
    }

    static public MLPoint[] GenerateRandomPoints(int count)
    {
        System.Random rand = new System.Random();

        MLPoint[] points = new MLPoint[count];
        for(int i = 0; i < count; i++)
        {
            points[i] = new MLPoint(rand.NextDouble(), rand.NextDouble());
        }
        return points;
    }
}
