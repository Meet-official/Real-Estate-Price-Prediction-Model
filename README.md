# 🏘️ Real Estate Price Prediction Model

This Python project predicts real estate housing prices using the Boston Housing Dataset. The model uses preprocessing pipelines, feature engineering, and Random Forest Regressor from `scikit-learn`.

## 📂 Files Included

- `Real-Estate.py` – Final cleaned model script
- `data.csv` – Dataset (Boston housing)
- `Real-Estate.joblib` – Saved trained model
- `requirements.txt` – Libraries needed to run
- `Real Estate.ipynb` – Step-by-step notebook version

## 🚀 How to Run

```bash
pip install -r requirements.txt
python Real-Estate.py
```

## 🔍 Features

- Preprocessing with pipelines (`SimpleImputer`, `StandardScaler`)
- Model selection: Linear, Decision Tree, Random Forest
- Evaluation with Cross Validation
- Final prediction on test data
- Saved model for future use

## 📈 Example Output

```
Final RMSE on Test Set: 2.69
Predicted Price: [23.56]
```

## 🤖 Libraries Used

- pandas
- numpy
- scikit-learn
- joblib

Made with ❤️ by **Meet Patel**
