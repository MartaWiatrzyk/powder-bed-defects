# Detection of powder bed defects in additive manufacturing processes
This project aims to classify images of powder bed used in the additive manufacturing process.


## Installation
To install all required dependencies:
* Use [env.yaml](env.yaml) file to create conda environment (Windows and conda users)
```bash
conda env create -vv -f env.yaml
```
or
* Use [requirements.txt](requirements.txt) 

```bash
pip install -r requirements.txt
```

## Usage
1. [Preprocessing](preprocessing.py)
   
   Find printer's table and remove background for further processing.
   
2. [Augmentation](augmentation.py)
   
    Perform data augmentation on preprocessed images and split data into validation and train/test dataset.

3. [Machine Learning Analysis](machine_learning.py)

    Use Machine Learning algorithms to classify images of powder bed. Train different models (Random Forest, SVM, KNN, XGBoost, AdaBoost), test accuracy, save and validate trained model.

4. [Deep Learning Analysis](deep_learning.py)

    Use pre-trained models (ResNet18, ResNet50 or VGG16) to classify images. Adjust model architecture, train, save and validate the best model.