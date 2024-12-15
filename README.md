
# Flower Classifier 🌸

A deep learning project to classify images of flowers using a convolutional neural network (CNN). This project demonstrates image classification techniques using PyTorch and transfer learning.

## 🚀 Project Overview

This project was developed as part of the **AI Programming with Python Nanodegree**. It focuses on building, training, and deploying an image classifier capable of identifying flower species from images.

## ✨ Features

- **Image Classification**: Classifies images into multiple flower species.
- **Transfer Learning**: Utilizes pre-trained models to improve efficiency and accuracy.
- **Command-Line Application**: Includes a CLI to make predictions on new images.
- **Training & Validation**: Models trained and validated with a robust dataset.

## 🛠️ Tools & Technologies

- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Pre-trained Models**: VGG16, ResNet, etc.
- **Libraries**: NumPy, Matplotlib, PIL, argparse

## 📂 Project Structure

```
flower-classifier/
├── checkpoint.pth         # Saved model weights
├── cat_to_name.json       # Mapping of categories to flower names
├── predict.py             # CLI for making predictions
├── train.py               # Training script
├── ImageClassifier.ipynb  # Jupyter Notebook with the project workflow
└── README.md              # Documentation (you're here!)
```

## 🚀 Usage

### Training the Model
To train the model, run:
```bash
python train.py --data_dir <path_to_data> --save_dir <path_to_save_checkpoint> --arch <model_architecture> --epochs <num_epochs>
```

Example:
```bash
python train.py --data_dir flowers --save_dir checkpoints --arch vgg16 --epochs 10
```

### Making Predictions
To make predictions on an image:
```bash
python predict.py --image_path <path_to_image> --checkpoint <path_to_checkpoint>
```

Example:
```bash
python predict.py --image_path example.jpg --checkpoint checkpoints/vgg16.pth
```

## 📊 Results & Insights

- **Accuracy**: Achieved 85% accuracy on the validation dataset.
- **Key Takeaways**: Transfer learning significantly improved training time and accuracy.

## 📈 Future Improvements

- Optimize the model for mobile devices.
- Increase the number of flower categories.
- Implement a web interface for real-time predictions.

## 📜 License

This project is licensed under the [MIT License](LICENSE).