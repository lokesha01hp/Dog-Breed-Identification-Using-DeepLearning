# Dog Breed Identification Using Deep Learning

This project classifies over **120 dog breeds** from a dataset of **10,000+ images** using deep learning. It harnesses the power of **transfer learning** to build an efficient, accurate image classification model with limited compute.

---

## Dataset

- 10,000+ high-quality dog images
- 120 distinct dog breeds
- Images are labeled and split into train/validation sets

---

## Approach

- **Transfer Learning:** Leveraged pre-trained CNN architectures (e.g., ResNet50, EfficientNet, VGG16)
- **Fine-tuning:** Trained top layers on dog images while freezing base model weights (initially)
- **Image Preprocessing:** Resizing, normalization, and optional data augmentation
- **Model Evaluation:** Accuracy, confusion matrix, and visual sample predictions

---

## Tech Stack

- **Python** 
- **TensorFlow / Keras** for model building & training
- **NumPy / Pandas** for data handling
- **Matplotlib / Seaborn** for visualization
- **OpenCV / PIL** for image manipulation

---

## 🛠️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/Dog-Breed-Identification-Using-DeepLearning.git
   cd Dog-Breed-Identification-Using-DeepLearning
