# 📱 Neural Network SMS Spam Classifier

A Deep Learning NLP model that classifies SMS messages as either **"ham"** (legitimate) or **"spam"** (unwanted/advertisement). Built with TensorFlow/Keras, this project leverages a **1D Convolutional Neural Network (CNN)** to efficiently identify spam trigger phrases while avoiding the common pitfalls of overfitting on small datasets.

---

## 🚀 Live Demo / Project Link

This project was developed and tested in Google Colaboratory. **[Click here to view the Colab Notebook](#https://colab.research.google.com/drive/1KhcQEdaidaqPXWG7iVQDBFYkz8FvkLca#scrollTo=YMYQPXSiDYP3)** 

![Preview](https://drive.google.com/file/d/1FOL0fL-b82OUV1LxfCx2aeXAMkl3heq9-)

---

## 🎯 Project Overview

The goal of this project was to build a robust text classification model capable of accurately filtering spam SMS messages. Because SMS text is short, heavily abbreviated, and often lacks standard grammar, traditional sequence models (like LSTMs) can overfit by trying to parse the sentence structure.

Instead, this project uses a **1D CNN architecture**, which excels at scanning text for specific "trigger n-grams" (e.g., *"won £1000"*, *"call to claim"*, *"install now"*). If any segment of the message strongly resembles spam, the model flags the entire message.

---

## 🧠 Technical Architecture & Choices

### Why 1D CNN over LSTM?

While LSTMs are powerful for understanding long-term dependencies (like translating a paragraph), they are prone to overfitting on small, short-text datasets like SMS. A message like *"Our new mobile video service is live..."* has perfect grammar and can fool an LSTM. A CNN, however, scans for the phrase **"mobile video service"** and correctly identifies it as a promotional trigger phrase.

---

### Model Pipeline

| Layer | Description |
|---|---|
| `Input(dtype=tf.string)` | Accepts raw string data directly into the model |
| `TextVectorization` | Standardizes (lowercases, strips punctuation), tokenizes, and pads/truncates sequences to a fixed length. Replaces deprecated `one_hot` preprocessing |
| `Embedding` | Converts word tokens into dense 32-dimensional vector representations |
| `SpatialDropout1D` | Drops entire word vectors during training, forcing the model not to rely on a single "magic" word |
| `Conv1D + GlobalMaxPooling1D` | Scans text in 5-word chunks to find patterns, then pulls the most prominent "spammy" feature from the entire text |
| `Dense + Dropout` | Classifies the extracted features with a Sigmoid output (`0` for ham, `1` for spam) |

---

### Key Technical Challenge Solved

During development, a `ValueError: Invalid dtype: strXXX` bug was encountered. This occurs in **Keras 3** when passing raw NumPy string arrays into a model. This was solved by explicitly defining an `Input(shape=(1,), dtype=tf.string)` layer and utilizing `tf.data.Dataset` pipelines to ensure proper tensor conversion.

---

## 📊 Evaluation Metrics

Accuracy alone is insufficient for spam detection due to **class imbalance** (ham significantly outnumbers spam). This project evaluates the model using:

- **Precision & Recall** — via `sklearn.metrics.classification_report`
- **Confusion Matrix** — visualized with Seaborn
- **Early Stopping** — monitors `val_loss` to halt training before overfitting occurs

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Framework | TensorFlow / Keras |
| Data Manipulation | Pandas, NumPy |
| Evaluation & Visualization | Scikit-Learn, Matplotlib, Seaborn |
| Environment | Google Colaboratory |

---

## ⚙️ Installation & Usage

1. Clone the repository (or open the Colab link).
2. Install the required dependencies:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

3. Run the notebook cells sequentially. The dataset is automatically downloaded via `wget`.

### Using the Prediction Function

The model includes a `predict_message()` function that returns the probability and the classification label:

```python
prediction = predict_message("you have won £1000 cash! call to claim your prize.")
print(prediction)
# Output: [0.9874, 'spam']

prediction = predict_message("how are you doing today?")
print(prediction)
# Output: [0.0021, 'ham']
```

---

## 🔮 Future Improvements

- **Deploy as an API:** Wrap the model in a FastAPI or Flask application to classify messages in real-time.
- **Handle Class Imbalance:** Implement class weights during training to improve recall on sparse spam categories.
- **Subword Tokenization:** Use BERT or SentencePiece tokenizers to better handle misspelled words or extreme slang (e.g., *"U w0n a pr1ze!"*).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

© 2025 Kandana Arachchige Don Yasandu Kethmika. All rights reserved.
