# Fashion-Mnist Classifier with PyTorch

## Inference
1- clone this repository:
```
git clone ""
```
2- Install required packages:
```
pip install -r requirements.txt
```
3-Run below command in terminal:
```
python inference.py
```
* by default this command uses test.png images to test classifier. to test other images(grayscale 28*28 pixels image), use below command:
 ```
        python inference.py --image <path_to_image>
 ```
* also 'cpu' with use to run this command, use ``` --device cuda ``` to change this behavior.


## Train
to train again model on Fashion-Mnist dataset, below command is the way:
```
python train.py --device <cpu or cuda> --batch_size <int> --epochs <int> --lr <learning rate: float>
```
this command assumes that you have dataset in a folder in a same directory.