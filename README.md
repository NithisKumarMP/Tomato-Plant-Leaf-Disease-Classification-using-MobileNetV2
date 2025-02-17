# Tomato-Plant-Leaf-Disease-Classification-using-MobileNetV2

## Dataset
The dataset consists of images of tomato leaves categorized into different disease types. It is sourced from an augmented version of the Pant Village dataset, specifically focusing on tomatoes, an essential crop.

The dataset includes 10 classifications, organized into 10 folders:
- 9 folders contain images of diseased tomato leaves.
- 1 folder contains images of healthy tomato leaves.

The dataset is structured as follows:
- **Training Set:** Contains images of diseased and healthy tomato leaves, categorized into 10 separate folders.
- **Validation Set:** Follows the same classification structure as the training set for model evaluation.
- **Testing Set:** Contains images labeled with their disease name directly in the image filename, instead of being stored in separate folders.

## Abstract
Tomato plant diseases significantly impact crop yield and quality. Early detection and classification of these diseases can enhance crop management and prevent large-scale losses.

Deep learning models, particularly Convolutional Neural Networks (CNNs), provide effective solutions for automated disease detection. By leveraging CNN architectures, we can extract relevant features from leaf images and classify diseases with high accuracy.

In this project, **MobileNetV2** is used due to its lightweight nature and efficiency. MobileNetV2 is designed for mobile and embedded applications, making it ideal for real-time disease detection on edge devices.

The model is trained using a labeled dataset containing diseased and healthy tomato leaf images. Performance evaluation is conducted using standard metrics such as accuracy, precision, recall, and F1-score.

The final system can be integrated into a web or mobile application to assist farmers and agricultural professionals in quickly identifying plant diseases.

## Project Objectives
- Develop an AI-based system to classify tomato plant leaf diseases using **MobileNetV2**.
- Utilize an augmented dataset derived from the Pant Village dataset, specifically focusing on tomato leaves.
- Train and evaluate the model using multiple performance metrics.
- Deploy the model into a real-world application for agricultural usage.

## Folder Structure
```
├── Dataset
│   ├── train
│   │   ├── Tomato_Healthy
│   │   ├── Tomato_Bacterial_Spot
│   │   ├── Tomato_Early_Blight
│   │   ├── Tomato_Late_Blight
│   │   ├── Tomato_Leaf_Mold
│   │   ├── Tomato_Septoria_Leaf_Spot
│   │   ├── Tomato_Spider_Mites
│   │   ├── Tomato_Target_Spot
│   │   ├── Tomato_Tomato_Mosaic_Virus
│   │   ├── Tomato_Tomato_Yellow_Leaf_Curl_Virus
│   ├── validation
│   │   ├── (same structure as train folder)
│   ├── test
│   │   ├── Individual image files labeled with the disease name in their filename
```

## Dataset Download
Due to the large size, the dataset is hosted externally. You can download it from:
🔗 **Kaggle**

After downloading, extract the dataset from the **Tomato** folder.

## Usage Instructions
1. Clone the repository:
   ```bash
   git clone <repository-link>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py --dataset path/to/dataset --model mobilenetv2
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py --model path/to/model
   ```
5. Deploy the model:
   - Integrate into a web or mobile application for real-time classification.

## Performance Metrics
The model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

## Future Work
- Improve dataset augmentation techniques.
- Deploy the model as a lightweight API for mobile and edge devices.
- Expand the dataset with more real-world samples for better generalization.


