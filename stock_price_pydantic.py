# Pydantic Model for Configuration
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class StockPricePredictorConfig(BaseModel):
    """
    Configuration for the StockPricePredictor class.
    """
    ticker: str = Field(default="AAPL", description="Stock ticker symbol.")
    start_date: str = Field(default="2015-01-01", description="Start date for historical data.")
    end_date: Optional[str] = Field(default=None, description="End date for historical data. If empty, defaults to current date.")
    seq_length: int = Field(default=30, description="Sequence length for LSTM model.")
    epochs: int = Field(default=20, description="Number of epochs for training.")
    batch_size: int = Field(default=32, description="Batch size for training.")

    @field_validator("start_date", "end_date")
    def validate_dates(cls, value):
        """
        Validate that the date strings are in the correct format (YYYY-MM-DD).
        Skip validation if the value is None (for end_date).
        """
        if value is None:
            return value
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {value}. Expected format: YYYY-MM-DD.")
        return value

    @field_validator("end_date")
    def validate_date_range(cls, value, values):
        """
        Validate that start_date is earlier than end_date.
        Skip validation if end_date is None.
        """
        if value is None:
            return value  # Skip validation if end_date is not provided
        if "start_date" in values.data and datetime.strptime(values.data["start_date"], "%Y-%m-%d") >= datetime.strptime(value, "%Y-%m-%d"):
            raise ValueError("start_date must be earlier than end_date.")
        return value