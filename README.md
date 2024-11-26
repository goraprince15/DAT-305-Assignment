{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Face Emotion Detection\
\
## Project Overview\
This project is focused on facial emotion recognition using deep learning techniques. The goal is to classify emotions such as **Happy**, **Sad**, **Angry**, **Surprised**, **Fear**, **Disgust**, and **Neutral** from facial expressions in grayscale images.\
\
Two models were implemented to solve this task:\
1. **Convolutional Neural Network (CNN)**: A custom-designed CNN built from scratch.\
2. **VGG16**: A fine-tuned pre-trained model to compare performance with the custom CNN.\
\
## Dataset\
The dataset used is the **Facial Expression Recognition Challenge Dataset** obtained from Kaggle. The dataset contains **48x48 pixel grayscale images** representing different emotions, and it includes **training**, **validation**, and **test** sets.\
\
Link to dataset: [Kaggle FER Challenge Dataset](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge)\
\
## Project Structure\
The project is structured as follows:\
- **data/**: Contains the dataset (training, validation, test images).\
- **notebooks/**: Jupyter notebooks or Google Colab files with the model training code.\
- **models/**: Saved versions of the trained CNN and VGG16 models.\
- **results/**: Contains results such as confusion matrices, accuracy plots, and classification reports.\
- **README.md**: You are reading this file.\
\
## Dependencies\
The project relies on the following major Python libraries:\
- **TensorFlow**: Used for building and training the models.\
- **Keras**: Higher-level interface for TensorFlow.\
- **OpenCV**: Used for face detection and image preprocessing.\
- **Matplotlib** and **Seaborn**: Used for visualization.\
- **Pandas** and **NumPy**: For data manipulation and handling.\
- **Scikit-learn**: For generating evaluation metrics like classification reports and confusion matrices.\
\
To install the required libraries, you can run:\
```sh\
pip install -r requirements.txt\
```\
\
## How to Run the Project\
1. **Clone the Repository**:\
   ```sh\
   git clone <repository-link>\
   cd face-emotion-detection\
   ```\
\
2. **Prepare the Dataset**:\
   - Download the dataset from Kaggle and extract it into the **data/** directory.\
\
3. **Train the Models**:\
   - You can run the training notebooks using Google Colab or Jupyter Notebook.\
   - Training scripts are available in the **notebooks/** folder.\
\
4. **Evaluate the Models**:\
   - Both models (CNN and VGG16) have evaluation scripts that can be used to see performance metrics such as **accuracy**, **precision**, **recall**, **F1-score**, **confusion matrix**, and **classification report**.\
\
## Code Explanation\
### Data Preprocessing\
- **Normalization**: Images were normalized to scale pixel values between 0 and 1.\
- **Data Augmentation**: Techniques like flipping and rotation were applied to increase data diversity and avoid overfitting.\
- **Face Detection**: OpenCV's Haar Cascade was used to detect faces and crop them if needed.\
- **One-Hot Encoding**: Emotion labels were one-hot encoded to make them suitable for categorical classification.\
\
### Model Architectures\
- **CNN**: A custom model with several convolutional and pooling layers. Dropout was added to avoid overfitting.\
- **VGG16**: A pre-trained VGG16 network was fine-tuned for emotion classification by adding custom fully connected layers at the end.\
\
### Training\
- **Optimizer**: Adam optimizer was used with categorical cross-entropy loss.\
- **Training Duration**: The models were trained for **20-50 epochs** depending on early stopping criteria and computational limits.\
\
### Evaluation\
- Metrics used: **Accuracy**, **Precision**, **Recall**, **F1-score**, and **Confusion Matrix**.\
- **Classification Report**: Generated to evaluate each emotion category's precision, recall, and F1-score.\
- **Visualization**: Plotted training and validation accuracy/loss to observe learning progress, and visualized a confusion matrix to analyze misclassifications.\
- **Prediction Visualization**: Displayed a set of test images along with true and predicted labels to visually inspect model performance.\
\
## Results\
- The **CNN model** outperformed VGG16 in all major metrics, including **Recall** and **F1-score**.\
- The **confusion matrices** indicated that certain emotions, like **Happy**, were classified more accurately compared to others like **Disgust**.\
\
## Limitations\
- **Limited Parameter Tuning**: Due to computational constraints, hyperparameter tuning was minimal.\
- **Class Imbalance**: Some emotion classes had significantly fewer examples, which impacted model performance.\
\
## Future Improvements\
- **Hyperparameter Tuning**: Better tuning to improve performance.\
- **Additional Data**: Incorporating more diverse datasets to reduce class imbalance.\
- **Use of Transformer Models**: Investigate transformer-based models for image classification tasks.\
\
## Acknowledgments\
\
The full code is available on [GitHub](https://github.com/goraprince15/DAT-305-Assignment/blob/main/Face_Emotion_Detection_Final.ipynb).\
- Special thanks to Kaggle for providing the dataset.\
- The open-source community for their amazing tools and resources.\
\
## Contact\
If you have any questions, feel free to contact me:\
- **Name**: Prince Gora\
- **Email**: goraprince15@gmail.com}