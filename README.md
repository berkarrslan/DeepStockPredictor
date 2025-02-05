# Stock Price Prediction using LSTM

## 📌 Project Overview
This project is designed to predict **stock prices** using a **Long Short-Term Memory (LSTM) neural network**. The model processes historical stock market data, extracts key technical indicators, and generates predictions for the next day's closing price.

## 🚀 Features
- Fetches real-time stock market data using **Yahoo Finance API**
- Applies **feature engineering** (RSI, MACD, Bollinger Bands, Moving Averages)
- Normalizes and prepares data for training
- Builds and trains an **LSTM model** for time series forecasting
- Evaluates model performance using **RMSE, MAPE, and R-Squared**
- Provides a prediction for the next day's closing price

## 🛠 Technologies Used
- **Python 3.9+**
- **TensorFlow & Keras** (for deep learning)
- **Scikit-Learn** (for data preprocessing and evaluation)
- **Yahoo Finance API (`yfinance`)** (for stock data)
- **Pydantic** (for configuration validation)
- **Matplotlib** (for visualization)

## 📂 Project Structure
```
├── stock_price_predicton.py     # Core prediction logic (LSTM model)
├── stock_price_pydantic.py      # Configuration validation using Pydantic
├── run.py                       # Main script to execute the pipeline
├── requirements.txt              # Required dependencies
└── README.md                    # Project documentation
```

## 🔧 Installation & Setup
### 1️⃣ Clone the repository
```bash
git clone https://github.com/berkarrslan/Deneme.git
cd Deneme
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the project
```bash
python run.py
```
By default, the model will fetch **NVIDIA (NVDA) stock data** from 2015 onwards, train an LSTM model, and predict the next day's closing price.

## 📊 Model Workflow
1️⃣ **Data Collection** → Fetches historical stock data from Yahoo Finance.
2️⃣ **Feature Engineering** → Adds indicators like **RSI, MACD, Bollinger Bands**.
3️⃣ **Data Preprocessing** → Normalization and sequence creation for LSTM.
4️⃣ **Model Training** → Builds and trains the LSTM network.
5️⃣ **Evaluation** → Computes **RMSE, MAPE, and R² score**.
6️⃣ **Prediction** → Generates the next day's stock price prediction.

## 📈 Example Output
```
Predicted closing price for the next day: 450.25
```

## 🔗 Future Improvements
- Hyperparameter tuning using **Keras Tuner**
- Adding **more technical indicators** for better predictions
- Deploying the model as a **REST API**

## 👨‍💻 Author
- **Berk Arslan**  
For any questions, feel free to reach out via GitHub or email! 📬

