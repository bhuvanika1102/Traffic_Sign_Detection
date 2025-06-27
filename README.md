# 🚦 Traffic Sign Detection

## 📜 Project Description

This project focuses on implementing and comparing two popular convolutional neural network (CNN) architectures — **VGG-16** and **ResNet-101** — for the task of **Traffic Sign Detection**. The models are trained on the **German Traffic Sign Benchmark (GTSRB)** dataset to accurately classify and detect different traffic signs. This work is a part of the **19AD651 – Deep Learning Laboratory** course.

The project leverages **TensorFlow** and **Keras** to build, train, and evaluate models. Data augmentation techniques like rotation, flipping, and scaling are used to enhance the dataset and improve model generalization.


## 🚀 Features

-  Traffic sign image classification
-  Data augmentation (rotation, flip, scale)
-  Performance comparison between VGG-16 and ResNet-101
-  Visualization of accuracy and loss curves
-  Model evaluation and testing with sample predictions
-  Model saving and loading for inference

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Google Colab / Kaggle / VS Code

## 📁 Project Structure

```plaintext
traffic-sign-detection/
├── data/                     # Dataset (GTSRB)
├── models/                   # Saved models (VGG, ResNet)
├── notebooks/                # Jupyter or Colab notebooks
├── src/                      # Source code
│   ├── data_preprocessing.py
│   ├── model_vgg16.py
│   ├── model_resnet101.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── outputs/                  # Accuracy/Loss plots
├── README.md                 # Project documentation
├── requirements.txt          # Required packages
```

## 💻 Installation & Setup
Prerequisites
Python 3.x

pip (Python package installer)

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Clone the Repository
```bash
git clone https://github.com/yourusername/traffic-sign-detection.git
cd traffic-sign-detection
```
## 🚀 How to Run
1. Prepare Dataset
Download the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

Place the dataset inside the /data directory.

2. Train VGG-16 Model
```bash
python src/model_vgg16.py
```
3. Train ResNet-101 Model
```bash
python src/model_resnet101.py
```
4. Evaluate Models
```bash
python src/evaluate.py
```
5. Predict a Sample Image
```bash
python src/predict.py --image "path/to/image.png"
```
6. Model Comparison
Model	Accuracy (%)
VGG-16	94.33%
ResNet-101	96.69%

✅ ResNet-101 outperforms VGG-16 in accuracy due to deeper layers with residual connections that mitigate the vanishing gradient problem.

📦 Deployment
Models can be exported as .h5 files for deployment.

Possible deployment to cloud or edge devices for real-time traffic sign detection.

🏗️ System Requirements
Software: Visual Studio Code, Kaggle, or Google Colab

Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib

---
## 📞 Contact Me
Feel free to reach out to me via email at bhuvani1102@gmail.com or connect with me on LinkedIn at https://www.linkedin.com/in/bhuvani1102
