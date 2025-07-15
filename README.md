
# Pneumonia Detection from Chest X-rays

This project provides an AI-powered solution for detecting **pneumonia** in chest X-ray images using deep learning. It utilizes a **Convolutional Neural Network (CNN)** trained on medical imaging data to classify input images as either *Normal* or *Pneumonia*. The goal is to assist medical professionals or serve as an educational tool for automatic radiology analysis.

---

## 📦 Project Structure

```
.
├── main.py               # Main entry script to perform predictions
├── model_loader.py       # Singleton class to load and cache the trained model
├── model_predict.py      # Handles image preprocessing and prediction logic
├── trainig_model.py      # (Optional) Code used to train the CNN model
├── pneumonia_model.keras # Saved trained Keras model
├── requirements.txt      # Python dependencies file
├── __pycache__/          # Python cache folder
└── .gitignore            # Git ignore rules
```

---

## 🚀 How to Use

### 1. Clone the repository

```bash
git clone https://github.com/your-username/pneumonia-detector.git
cd pneumonia-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run prediction

```bash
python main.py --image path_to_image.jpg
```

The model will return whether the image shows **pneumonia** or not.

---

## 🧠 About the Model

- **Framework**: TensorFlow / Keras
- **Model Type**: Convolutional Neural Network (CNN)
- **Input**: Chest X-ray image (JPG or PNG)
- **Output**: `Pneumonia` or `Normal`

The model was trained using a binary classification approach and optimized using image preprocessing and model regularization techniques.

---

## 🔍 Detailed File Descriptions

| File               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `main.py`          | Main script to run prediction from terminal                                |
| `model_loader.py`  | Loads the `.keras` model (singleton pattern to avoid redundant loads)       |
| `model_predict.py` | Preprocesses the input image and performs classification                    |
| `trainig_model.py` | (Optional) Contains the training pipeline (model architecture, fit, etc.)   |
| `pneumonia_model.keras` | Trained model file ready for use                                  |
| `requirements.txt` | List of required packages to install using `pip`                           |

---

## 🧪 Dataset

The model was trained on publicly available chest X-ray datasets (e.g., [Kaggle Chest X-ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)) containing labeled images for normal and pneumonia cases.

---

## ⚠️ Disclaimer

> This project is for **educational and research purposes only**. It is not intended to be used for real medical diagnosis or treatment. Please consult a certified radiologist or physician for actual medical interpretation.

---

## 💡 Future Enhancements

- Add web interface (Flask or Streamlit)
- Classify more categories (e.g., viral vs. bacterial pneumonia)
- Improve performance with more data and augmentation
- Deploy as a REST API for remote use

---

## 🙋‍♂️ Author

**Ahmed Khaled**  
Feel free to reach out for questions, feedback, or contributions.

---

## ⭐️ Support

If you like this project, consider giving it a star on GitHub!
