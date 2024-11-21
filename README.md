# **Fake News Detection System Using 7 Types of LSTM**

![cover](https://github.com/user-attachments/assets/568561fb-f33c-4f89-af94-b9f785f097db)

## **Table of Contents**
1. [Introduction](#1-introduction)  
2. [Objective](#2-objective)  
3. [Technologies Used](#3-technologies-used)  
4. [System Architecture](#4-system-architecture)  
5. [Dataset Description](#5-dataset-description)  
6. [Model Design and Training](#6-model-design-and-training)  
7. [Application Workflow](#7-application-workflow)  
8. [Deployment](#8-deployment)  
9. [Performance Metrics](#9-performance-metrics)  
10. [License](#10-license)  
11. [Contact Information](#11-contact-information)  

---

## **1. Introduction**
The **Fake News Detection System** is a machine learning application aimed at addressing the critical issue of misinformation in digital media. Leveraging deep learning techniques, particularly **LSTM (CNN-LSTM hybrid model)** networks, this system is designed to classify news articles or headlines as either real or fake, providing users with a reliable tool for verifying information.

---

## **2. Objective**
The primary objective of this project is to create an automated system capable of:
1. Identifying fake news articles and headlines using NLP (Natural Language Processing) and machine learning models.
2. Providing an easy-to-use web interface for users to input news text and receive a classification result in real-time.
3. Implementing a scalable and efficient solution for deployment in a cloud environment.

---

## **3. Technologies Used**
| **Category**            | **Technology**                  | **Purpose**                          |
|--------------------------|----------------------------------|---------------------------------------|
| **Programming Language** | Python                          | Core implementation                  |
| **Web Framework**        | Flask                           | Backend and API                      |
| **Machine Learning**     | TensorFlow/Keras                | Model development and training       |
| **Data Processing**      | Pandas, NumPy                   | Dataset manipulation and preprocessing|
| **Deployment**           | Render                          | Cloud hosting                        |
| **Frontend**             | HTML, CSS, Bootstrap            | User interface design                |

---

## **4. System Architecture**
The architecture of the Fake News Detection system consists of three main components:

1. **Frontend (User Interface)**:  
   - Built with **HTML**, **CSS**, and **Bootstrap** to provide a responsive, user-friendly interface.  
   - Allows users to input news text for classification.

2. **Backend (Flask Application)**:  
   - Flask is used to serve the model and handle requests.  
   - Preprocessing of text inputs and inference is done via the Flask API.

3. **Model**:  
   - An LSTM-based deep learning model trained on text data to classify news articles as fake or real.  
   - The model and tokenizer are stored as `model.h5` and `tokenizer.pkl`, respectively.

**Architecture Diagram**:  
```
[User Input] -> [Frontend (Flask)] -> [Preprocessing] -> [Model Inference] -> [Result Display]
```

---

## **5. Dataset Description**
The dataset used for training the model contains labeled news articles, each classified as either **fake** (0) or **real** (1). The dataset consists of the following key columns:
- **text**: The content of the news article or headline.
- **label**: A binary label where 0 denotes fake news and 1 denotes real news.

**Dataset Statistics**:  
- Total records: 50,000  
- Fake news: 25,000  
- Real news: 25,000

The dataset was sourced from Kaggle (or specify the actual source here).

---

## **6. Model Design and Training**
### **Model Architecture**:
- **Embedding Layer**: Converts text input into dense vector representations.
- **LSTM Layer**: Processes sequential data to capture context and relationships within the text.
- **Dropout Layer**: Prevents overfitting by randomly setting some weights to zero during training.
- **Dense Layer**: Outputs the classification probability (fake or real).

### **Training Hyperparameters**:
| **Hyperparameter** | **Value**       |
|--------------------|-----------------|
| Optimizer          | Adam            |
| Learning Rate      | 0.001           |
| Epochs             | 5              |
| Batch Size         | 32              |

### **Training Data Split**:
| **Dataset Split** | **Percentage** |
|-------------------|----------------|
| Training          | 70%            |
| Validation        | 15%            |
| Testing           | 15%            |

The model is trained using **TensorFlow** and **Keras**, leveraging GPU acceleration for faster training.

---

## **7. Application Workflow**
1. **User Input**: The user enters a news headline or article into the web form.
2. **Preprocessing**: The backend processes the input by cleaning, tokenizing, and padding the text.
3. **Model Prediction**: The processed text is passed to the trained LSTM model to predict whether the news is real or fake.
4. **Result**: The result is displayed on the web interface, showing whether the news is **real** or **fake**.

---

## **8. Deployment**
The Fake News Detection system is deployed on **Render**, a cloud hosting platform, which ensures seamless scalability and availability.

### **Deployment Steps**:
1. **Requirements**: Install all dependencies listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Application**: Start the Flask application using **Gunicorn**:
   ```bash
   gunicorn app:app
   ```

3. **Render Hosting**: The web application is hosted on **Render**, allowing users to access the interface globally.

---

## **9. Performance Metrics**
The model's performance was evaluated using the following metrics:

| **Metric**   | **Value**  |
|--------------|------------|
| Accuracy     | 98.8%      |
| Precision    | 99%        |
| Recall       | 99%        |
| F1-Score     | 99%        |

The confusion matrix was also computed:

|              | Predicted Fake | Predicted Real |
|--------------|----------------|----------------|
| **Actual Fake** | 4602           | 46             |
| **Actual Real** | 60             | 4230           |

---

## **10. License**
This project is licensed under the **MIT License**

## **11. Contact Information**
- **Email:** [iconicemon01@gmail.com](mailto:iconicemon01@gmail.com)
- **WhatsApp:** [+8801834363533](https://wa.me/8801834363533)
- **GitHub:** [Md-Emon-Hasan](https://github.com/Md-Emon-Hasan)
- **LinkedIn:** [Md Emon Hasan](https://www.linkedin.com/in/md-emon-hasan)
- **Facebook:** [Md Emon Hasan](https://www.facebook.com/mdemon.hasan2001/)

---
