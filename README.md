# Dog Breed Identification Using Deep Learning

This project classifies over **120 dog breeds** from a dataset of **10,000+ images** using deep learning. It harnesses the power of **transfer learning** to build an efficient, accurate image classification model with limited compute.

---

## Dataset

- 10,000+ high-quality dog images
- 120 distinct dog breeds
- Images are labelled and split into train/validation sets

---

## Approach

- **Transfer Learning:** Leveraged pre-trained CNN architectures (Mobilenetv4)
- **Fine-tuning:** Trained top layers on dog images while freezing base model weights (initially)
- **Image Preprocessing:** Resizing, normalisation, and optional data augmentation
- **Model Evaluation:** Accuracy, confusion matrix, and visual sample predictions

---

## Tech Stack

- **Python** 
- **TensorFlow / Keras** for model building & training
- **NumPy / Pandas** for data handling
- **Matplotlib / Seaborn** for visualization

