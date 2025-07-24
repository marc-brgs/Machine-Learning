# Machine Learning: neural networks, Multi-Layer Perceptron (MLP)

The goal of this project was to create a machine learning library in C++ that could be used in any client. In this case, I chose Unity. Then, I created an application using my library and trained a neural network to identify which game the image it was given comes from. The results obtained are excellent, and the neural network manages to make good predictions on images it was not aware of during its training.

Simple cases learning tests documentation:
[README_ML.pdf](https://github.com/marc-brgs/Machine-Learning/blob/main/README_ML.pdf)

### Dataset
Folder : [Machine-Learning/Assets/Dataset](https://github.com/marc-brgs/Machine-Learning/tree/main/Machine-Learning/Assets/Dataset)


### Final model details and results

Training on a total of 156000 images including 39000 unique images (13000 for each of the 3 games)br>
1 epoch : 3 images (1 for each game)<br>
Layers of neurons : 3888 > (100 > 20 > 10) > 3<br>
Learning rate : 0.004<br>
Learning duration : 1h04m48s<br>

<p align="center"><i>MLP accuracy during training on 10 unknown images</i></p>

<p align="center">
  <img alt="Training on 156000 images (3x 52000)" src="https://github.com/marc-brgs/Machine-Learning/blob/main/Python%20plots%20images/156000_images_unknown_100_20_10_lr_004.png">
</p>

