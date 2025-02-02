# primary libraries
from datetime import datetime
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

# pydantic model
from stock_price_pydantic import StockPricePredictorConfig

class StockPricePredictor:
    def __init__(self, config: StockPricePredictorConfig):
        """
        Initialize the StockPricePredictor class.

        :param config: Configuration object containing parameters for the predictor.
                      This includes the ticker symbol, start and end dates, sequence length,
                      batch size, and number of epochs for training.
        """
        self.config = config
        self.data = None
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None
        self.feature_names = []

    def download_data(self) -> None:
        """
        Download historical stock data using the Yahoo Finance API.

        The data is downloaded for the specified ticker symbol and date range.
        If no end date is provided, the current date is used as the end date.
        The downloaded data is stored in the `self.data` attribute.
        """
        end_date = self.config.end_date if self.config.end_date else datetime.now().strftime("%Y-%m-%d")
        self.data = yf.download(
            self.config.ticker, start=self.config.start_date, end=end_date
        )
        
        # MultiIndex sütun isimlerini düzleştir
        self.data.columns = self.data.columns.droplevel(1)  # Ticker sembolünü kaldır

        print(self.data.columns)
        # Save the data to a CSV file for debugging
        self.data.to_csv(f"{self.config.ticker}_data_first.csv")
        print(f"Downloaded data for {self.config.ticker} from {self.config.start_date} to {end_date}.")


    def preprocess_data(self) -> tuple:
        """
        Preprocess the downloaded stock data.

        This method handles missing values, normalizes the data, and splits it into
        training and testing sets using TimeSeriesSplit. It also engineers additional
        features such as technical indicators (RSI, MACD, Bollinger Bands) and temporal
        features (year, month, day, day of the week).

        :return: A tuple containing the training and testing data (X_train, X_test, y_train, y_test).
        """
        # Handle missing values
        self.data.ffill(inplace=True)

        # Split into training and testing sets using TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            # Engineer features for each fold
            train_data = self.engineer_features(train_data)
            test_data = self.engineer_features(test_data)

            X_train, X_test = train_data.drop(columns=['Close']), test_data.drop(columns=['Close'])
            y_train, y_test = train_data['Close'], test_data['Close']

        # Save the feature names for later use
        self.feature_names = X_train.columns.tolist()

        # Normalize the features (X)
        self.scaler_X = MinMaxScaler()
        X_train = self.scaler_X.fit_transform(X_train)
        X_test = self.scaler_X.transform(X_test)

        # Normalize the target (y)
        self.scaler_y = MinMaxScaler()
        y_train = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test = self.scaler_y.transform(y_test.values.reshape(-1, 1))

        return X_train, X_test, y_train, y_test

    def calculate_rsi(self, data, period=14):
        """
        Calculate the Relative Strength Index (RSI).

        RSI is a momentum oscillator that measures the speed and change of price movements.
        It is used to identify overbought or oversold conditions in a stock.

        :param data: DataFrame containing the stock data.
        :param period: The period over which to calculate RSI (default is 14).
        :return: A Series containing the RSI values.
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # İlk değerler için 50 (nötr RSI değeri) atanır
        return rsi

    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate the Moving Average Convergence Divergence (MACD).

        MACD is a trend-following momentum indicator that shows the relationship between
        two moving averages of a stock’s price. It consists of the MACD line, the signal line,
        and the MACD histogram.

        :param data: DataFrame containing the stock data.
        :param fast_period: The period for the fast EMA (default is 12).
        :param slow_period: The period for the slow EMA (default is 26).
        :param signal_period: The period for the signal line (default is 9).
        :return: A DataFrame with additional columns for MACD, MACD signal, and MACD histogram.
        """
        data = data.copy()  # Orijinal DataFrame'i korumak için bir kopya oluştur
        data['EMA_Fast'] = data['Close'].ewm(span=fast_period, adjust=False).mean()
        data['EMA_Slow'] = data['Close'].ewm(span=slow_period, adjust=False).mean()
        data['MACD'] = data['EMA_Fast'] - data['EMA_Slow']
        data['MACD_Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # NaN değerleri doldur (inplace yerine atama yaparak)
        data['EMA_Fast'] = data['EMA_Fast'].fillna(data['Close'].iloc[0])  # İlk değerler için ilk Close değerini kullan
        data['EMA_Slow'] = data['EMA_Slow'].fillna(data['Close'].iloc[0])
        data['MACD'] = data['MACD'].fillna(0)  # İlk değerler için 0 atanır
        data['MACD_Signal'] = data['MACD_Signal'].fillna(0)
        data['MACD_Hist'] = data['MACD_Hist'].fillna(0)
        
        return data

    def calculate_bollinger_bands(self, data, period=20, nbdev=2):
        """
        Calculate Bollinger Bands.

        Bollinger Bands consist of a middle band (simple moving average) and two outer bands
        (standard deviations away from the middle band). They are used to measure volatility
        and identify overbought or oversold conditions.

        :param data: DataFrame containing the stock data.
        :param period: The period over which to calculate the moving average (default is 20).
        :param nbdev: The number of standard deviations to use for the outer bands (default is 2).
        :return: A DataFrame with additional columns for the moving average, upper band, and lower band.
        """
        data = data.copy()  # Orijinal DataFrame'i korumak için bir kopya oluştur
        data['MA'] = data['Close'].rolling(window=period, min_periods=1).mean()
        data['STD'] = data['Close'].rolling(window=period, min_periods=1).std()
        data['Upper_Band'] = data['MA'] + (data['STD'] * nbdev)
        data['Lower_Band'] = data['MA'] - (data['STD'] * nbdev)
        
        # NaN değerleri doldur (inplace yerine atama yaparak)
        data['MA'] = data['MA'].fillna(data['Close'].iloc[0])  # İlk değerler için ilk Close değerini kullan
        data['STD'] = data['STD'].fillna(0)  # İlk değerler için 0 atanır
        data['Upper_Band'] = data['Upper_Band'].fillna(data['Close'].iloc[0])
        data['Lower_Band'] = data['Lower_Band'].fillna(data['Close'].iloc[0])
        
        return data

    def engineer_features(self, data):
        """
        Engineer additional features for the stock data.

        This method adds technical indicators (e.g., MACD, Bollinger Bands) and temporal features
        (e.g., year, month, day, day of the week) to the dataset. These features are used to
        improve the predictive power of the model.

        :param data: Input DataFrame containing the stock data.
        :return: DataFrame with additional features.
        """
        # Calculate RSI
        # data['RSI'] = self.calculate_rsi(data)

        # Copy the DataFrame to avoid modifying the original data
        data = data.copy()

        # Ensure the index is a datetime object
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Extract year, month, and day from the date
        data.loc[:, 'Year'] = data.index.year
        data.loc[:, 'Month'] = data.index.month
        data.loc[:, 'Day'] = data.index.day
        data.loc[:, 'DayOfWeek'] = data.index.dayofweek 

        # one-hot encode the day of the week and month
        data = pd.get_dummies(data, columns=['DayOfWeek', 'Month'])

        # Calculate MACD
        data = self.calculate_macd(data)

        # # Calculate Bollinger Bands
        data = self.calculate_bollinger_bands(data)

        # Add moving averages
        data['MA_7'] = data['Close'].rolling(window=7, min_periods=1).mean()
        data['MA_30'] = data['Close'].rolling(window=30, min_periods=1).mean()

        # Add percentage change
        # data['Pct_Change'] = data['Close'].pct_change().fillna(0)

        # write to csv data
        data.to_csv(f"{self.config.ticker}_data.csv")
        return data

    def create_sequences(self, X: np.array, y: np.array, seq_length: int) -> tuple:
        """
        Create sequences for the LSTM model.

        This method converts the input data into sequences of a specified length, which are
        required for training the LSTM model.

        :param X: Input features.
        :param y: Target values.
        :param seq_length: Length of each sequence.
        :return: A tuple containing the sequences (X_seq) and target values (y_seq).
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    def build_model(self, input_shape: tuple) -> None:
        """
        Build and compile the LSTM model.

        This method constructs the LSTM model architecture, including LSTM layers, dropout layers,
        and dense layers. The model is compiled with the Adam optimizer and mean squared error loss.

        :param input_shape: The shape of the input data (sequence length, number of features).
        """
        print('input_shape',input_shape)
        inputs = Input(shape=input_shape)  # Input katmanı ekleyin
        lstm1, state_h, state_c = LSTM(128, return_sequences=True, return_state=True,  kernel_regularizer=l2(0.005))(inputs)  # Input katmanını LSTM'e bağlayın
        lstm1 = Dropout(0.1)(lstm1)
        lstm2 = LSTM(128, return_sequences=False)(lstm1, initial_state=[state_h, state_c])
        lstm2 = Dropout(0.1)(lstm2)
        x = Dense(32)(lstm2)
        x = Dropout(0.1)(x)
        x = Dense(16)(x)
        outputs = Dense(1)(x)
        
        self.model = Model(inputs, outputs)  # Modeli oluşturun
        optimizer = Adam(learning_rate=self.config.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def train_model(self, X_train: np.array, y_train: np.array) -> None:
        """
        Train the LSTM model.

        This method trains the LSTM model on the provided training data. The training process
        includes validation on a subset of the data to monitor overfitting.

        :param X_train: Training sequences.
        :param y_train: Training target values.
        """
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=0.2
        )

    def plot_training_history(self) -> None:
        """
        Plot the training and validation loss over epochs.

        This method visualizes the training and validation loss to help diagnose overfitting
        or underfitting during the training process.
        """
        if self.history is None:
            print("No training history available.")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_predictions(self, X_test: np.array, y_test: np.array) -> None:
        """
        Plot actual vs predicted stock prices.

        This method visualizes the actual and predicted stock prices to evaluate the model's
        performance on the test data.

        :param X_test: Test sequences.
        :param y_test: Test target values.
        """
        y_pred = self.model.predict(X_test)

        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label=f'{self.config.ticker} Actual Prices')
        plt.plot(y_pred, label=f'{self.config.ticker} Predicted Prices')
        plt.title(f'{self.config.ticker} Actual vs Predicted Stock Prices')
        plt.xlabel('Time')
        plt.ylabel(f'{self.config.ticker} Stock Price (Normalized)')
        plt.legend()
        plt.show()

    def evaluate_model(self, X_test: np.array, y_test: np.array) -> None:
        """
        Evaluate the model on the test data.

        This method calculates and prints the Root Mean Squared Error (RMSE), Mean Absolute
        Percentage Error (MAPE), and R-squared (R²) to evaluate the model's performance.

        :param X_test: Test sequences.
        :param y_test: Test target values.
        """
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{self.config.ticker} - RMSE: {rmse}")
        print(f"{self.config.ticker} - MAPE: {mape}")
        print(f"{self.config.ticker} - R-squared: {r2}")

    def predict_next_day_close(self) -> float:
        """
        Predict the next day's closing price using the most recent data.

        This method uses the most recent data (last `seq_length` days) to predict the next
        day's closing price. The prediction is made using the trained LSTM model.

        :return: Predicted closing price for the next day.
        """
        try:
            # Get the most recent data (last `seq_length` days)
            recent_data = self.data.iloc[-self.config.seq_length:].copy()
            # Engineer features for the recent data
            recent_data = self.engineer_features(recent_data)

            # Drop the 'Close' column (target variable)
            X_recent = recent_data.drop(columns=['Close'])

            # Ensure all feature names are present
            for feature in self.feature_names:
                if feature not in X_recent.columns:
                    X_recent[feature] = False

            # Reorder columns to match the training data
            X_recent = X_recent[self.feature_names]

            # Normalize the features using the pre-fitted scaler
            X_recent_scaled = self.scaler_X.transform(X_recent)

            # Reshape the data to match the LSTM input shape: (1, seq_length, num_features)
            X_recent_reshaped = X_recent_scaled.reshape((1, self.config.seq_length, X_recent.shape[1]))

            # Predict the next day's closing price
            predicted_close_scaled = self.model.predict(X_recent_reshaped)

            # Inverse transform the predicted value to get the actual closing price
            predicted_close = self.scaler_y.inverse_transform(predicted_close_scaled.reshape(-1, 1))
            return predicted_close[0][0]
        except Exception as e:
            print(f"Error in predicting the next day's closing price: {e}")
            return None

    def run_pipeline(self) -> None:
        """
        Run the entire stock price prediction pipeline.

        This method executes the following steps:
        1. Download historical stock data.
        2. Preprocess the data and engineer features.
        3. Build and train the LSTM model.
        4. Evaluate the model's performance.
        5. Plot the training history and predictions.
        """
        try:
            # Step 1: Download data YFinance
            self.download_data()

            # Step 2: Preprocess data and Feature Engineering
            X_train, X_test, y_train, y_test = self.preprocess_data()
            print('after preprocess_data function')
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            # Step 3: Create sequences dynamically based on the data
            seq_length = min(self.config.seq_length, X_train.shape[0] - 1)  # Veri boyutuna göre seq_length ayarla
            print(f"Sequence length: {seq_length}")
            X_train, y_train = self.create_sequences(X_train, y_train, seq_length)
            X_test, y_test = self.create_sequences(X_test, y_test, seq_length)

            # print X_train, y_train, X_test, y_test
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            # Step 4: Build and train the model
            self.build_model((seq_length, X_train.shape[2]))
            self.train_model(X_train, y_train)

            # Step 5: Evaluate the model
            self.evaluate_model(X_test, y_test)

            self.plot_training_history()
            self.plot_predictions(X_test, y_test)

        except Exception as e:
            print(f"An error occurred: {e}")