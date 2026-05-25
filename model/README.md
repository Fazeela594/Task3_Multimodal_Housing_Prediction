# Task 3 - Multimodal Housing Price Prediction

## Objective
This project predicts house prices using:
- House images
- Tabular data (area and number of rooms)

The project combines image-based deep learning features with structured numerical data.

---

## Technologies Used
- Python
- PyTorch
- Torchvision
- Pandas
- PIL (Python Imaging Library)

---

## Project Structure

Task3_Multimodel_Housing/
│
├── data/
│ └── houses.csv
│
├── images/
│ ├── house1.jpg
│ ├── house2.jpg
│ └── house3.jpg
│
├── model/
│ ├── model.py
│ ├── train.py
│ └── predict.py
│
├── model.pth
└── README.md

---

## Dataset
The dataset contains:
- House images
- Area of house
- Number of rooms
- House prices

Example:

| Image | Area | Rooms | Price |
|------|------|------|------|
| house1.jpg | 1200 | 3 | 50 |
| house2.jpg | 1500 | 4 | 70 |

---

## Model Architecture
The project uses:
- ResNet18 CNN for image feature extraction
- Fully connected layers for tabular data
- Feature fusion for multimodal learning

---

## Training
To train the model:

```bash
python model/train.py
```

---

## Prediction
To predict house price:

```bash
python model/predict.py
```

---

## Output
The model predicts housing prices based on:
- Visual house features
- Area
- Number of rooms

---

## Skills Gained
- Multimodal Machine Learning
- CNN Feature Extraction
- PyTorch Model Training
- Regression Modeling
- Combining Image + Tabular Data

---

## Author
Fazeela Manzoor