
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

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kayodeodada/flower-classifier.git
   cd flower-classifier
   ```

2. **Install dependencies:**
   Ensure you have Python 3.x and pip installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   The dataset used for training can be downloaded from [Udacity's Flower Dataset](https://github.com/udacity/aipnd-project).

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

### Command-Line Options
Use the `--help` flag for detailed usage instructions:
```bash
python train.py --help
python predict.py --help
```

## 📊 Results & Insights

- **Accuracy**: Achieved X% accuracy on the validation dataset.
- **Key Takeaways**: Transfer learning significantly improved training time and accuracy.

## 📈 Future Improvements

- Optimize the model for mobile devices.
- Increase the number of flower categories.
- Implement a web interface for real-time predictions.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## 📬 Contact

If you have any questions or suggestions, feel free to reach out:
- **Name**: Kayode Dada
- **Email**: kayode.dada@example.com
- **GitHub**: [kayodeodada](https://github.com/kayodeodada)
