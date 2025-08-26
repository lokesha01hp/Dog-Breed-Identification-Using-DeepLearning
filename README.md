# Dog Breed Identification Using Deep Learning

This project classifies over **120 dog breeds** from a dataset of more than 10,000 images using deep learning. It harnesses the power of **transfer learning** to build an efficient and accurate image classification model with limited compute resources.

---

## Dataset

- 10,000+ high-quality dog images
- 120 distinct dog breeds
- Images are labelled and split into train/validation sets

---

## Approach

- **Transfer Learning:** Leveraged pre-trained CNN architectures (Mobilenetv4)
- **Fine-tuning:** Trained top layers on dog images while freezing base model weights 
- **Image Preprocessing:** Resizing, normalisation, and optional data augmentation
- **Model Evaluation:** Accuracy, confusion matrix, and visual sample predictions

---

## Tech Stack

- **Python** 
- **TensorFlow and Keras** 
- **NumPy / Pandas** 
- **Matplotlib / Seaborn**

