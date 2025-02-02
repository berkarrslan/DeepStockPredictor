from stock_price_predicton import StockPricePredictor
from stock_price_pydantic import StockPricePredictorConfig

if __name__ == "__main__":
    # Define configuration using Pydantic
    try:
        config = StockPricePredictorConfig(
            ticker="NVDA",
            start_date="2015-01-01",
            end_date=None,  # end_date is optional
            seq_length=30,
            epochs=25,
            batch_size=16
        )

        # Initialize the predictor
        predictor = StockPricePredictor(config)

        # Run the pipeline
        predictor.run_pipeline()

        # Predict the next day's closing price
        next_day_close = predictor.predict_next_day_close()
        if next_day_close:
            print(f"Predicted closing price for the next day: {next_day_close:.2f}")
    except ValueError as e:
        print(f"Configuration error: {e}")