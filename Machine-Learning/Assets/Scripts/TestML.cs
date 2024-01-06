using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;

public class TestML : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        //AND();
        XOR();
    }

    void AND()
    {
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 1 }, 1);

        for (int i = 0; i < 100000; i++)
        {
            p.train(new double[] { 0, 0 }, new double[] { 0 }, 1);
            p.train(new double[] { 0, 1 }, new double[] { 0 }, 1);
            p.train(new double[] { 1, 0 }, new double[] { 0 }, 1);
            p.train(new double[] { 1, 1 }, new double[] { 1 }, 1);
        }

        // Afficher la sortie
        p.predict(new double[] { 0, 0 });
        p.predict(new double[] { 0, 1 });
        p.predict(new double[] { 1, 0 });
        p.predict(new double[] { 1, 1 });
    }

    void LinearSimple()
    {
        // Linear Model : OK
        // MLP (2, 1) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 0 }, 1);
    }

    void LinearMultiple()
    {
        // Linear Model : OK
        // MLP (2, 1) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 0 }, 1);
    }

    void XOR()
    {
        // Linear Model : KO
        // MLP (2, 2, 1) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 2 }, 1);

        for (int i = 0; i < 100000; i++)
        {
            p.train(new double[] { 0, 0 }, new double[] { 0 }, 1);
            p.train(new double[] { 0, 1 }, new double[] { 1 }, 1);
            p.train(new double[] { 1, 0 }, new double[] { 1 }, 1);
            p.train(new double[] { 1, 1 }, new double[] { 0 }, 1);
        }

        p.predict(new double[] { 0, 0 });
        p.predict(new double[] { 0, 1 });
        p.predict(new double[] { 1, 0 });
        p.predict(new double[] { 1, 1 });
    }

    void Cross()
    {
        // Linear Model : KO
        // MLP (2, 4, 1) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 4 }, 1);
    }

    void MultiLinear3Classes()
    {
        // Linear Model : OK
        // MLP (2, 3) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 0 }, 3);
    }

    void MultiCross()
    {
        // Linear Model : OK
        // MLP (2, ?, ?, 3) : OK
        PerceptronWrapper p = new PerceptronWrapper(2, new int[] { 0 }, 3);
    }
}
