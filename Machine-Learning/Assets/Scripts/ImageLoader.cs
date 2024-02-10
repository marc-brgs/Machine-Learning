using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class ImageLoader
{
    public static Texture2D LoadTexture(string filePath)
    {
        Texture2D tex = null;
        byte[] fileData;

        if (File.Exists(filePath))
        {
            fileData = File.ReadAllBytes(filePath);
            tex = new Texture2D(48, 27); // 48x27 pixels (1296 pixels)
            tex.LoadImage(fileData); //..this will auto-resize the texture dimensions
        }
        else
        {
            Debug.Log("Path \"" + filePath + "\" does not exists!");
        }
        return tex;
    }

    public static double[] LoadFlatPixels(Texture2D tex)
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

        return myList.ToArray();
    }

    public static double[] LoadImagePixels(string filePath)
    {
        Texture2D tex = LoadTexture(filePath);
        return LoadFlatPixels(tex);
    }
}
