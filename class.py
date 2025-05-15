from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader

from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta

from finbert_utils import estimate_sentiment

from dotenv import load_dotenv
import os
import logging

# --- Load environment variables ---
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = os.getenv("BASE_URL")

# --- Basic checks for missing environment variables ---
if not API_KEY or not API_SECRET or not BASE_URL:
    raise ValueError("API credentials or BASE_URL not set. Check your .env file.")

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)

# --- Alpaca credentials dict ---
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}


class MLTrader(Strategy):
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = 0.5):
        if not 0 < cash_at_risk <= 1:
            raise ValueError("cash_at_risk must be between 0 and 1.")
        self.symbol = symbol.upper()
        self.cash_at_risk = cash_at_risk
        self.sleeptime = "24H"
        self.last_trade = None
        try:
            self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        except Exception as e:
            logging.error(f"Failed to initialize Alpaca REST API: {e}")
            raise

    def position_sizing(self):
        try:
            cash = self.get_cash()
            last_price = self.get_last_price(self.symbol)
            quantity = round(cash * self.cash_at_risk / last_price, 0)
            return cash, last_price, quantity
        except Exception as e:
            logging.error(f"Error in position sizing: {e}")
            return 0, 0, 0

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        try:
            today, three_days_prior = self.get_dates()
            news_items = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
            headlines = [event.__dict__["_raw"]["headline"] for event in news_items]
            probability, sentiment = estimate_sentiment(headlines)
            logging.info(f"Sentiment: {sentiment} (probability: {probability})")
            return probability, sentiment
        except Exception as e:
            logging.warning(f"Sentiment analysis failed: {e}")
            return 0.5, "neutral"

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        if quantity <= 0:
            logging.warning("Skipping trading iteration due to insufficient quantity.")
            return

        probability, sentiment = self.get_sentiment()

        try:
            if cash > last_price:
                if sentiment == "positive" and probability > 0.999:
                    if self.last_trade == "sell":
                        self.sell_all()
                    order = self.create_order(
                        self.symbol,
                        quantity,
                        "buy",
                        type="bracket",
                        take_profit_price=last_price * 1.20,
                        stop_loss_price=last_price * 0.95
                    )
                    self.submit_order(order)
                    logging.info(f"BUY order submitted: {quantity} shares at ${last_price}")
                    self.last_trade = "buy"

                elif sentiment == "negative" and probability > 0.999:
                    if self.last_trade == "buy":
                        self.sell_all()
                    order = self.create_order(
                        self.symbol,
                        quantity,
                        "sell",
                        type="bracket",
                        take_profit_price=last_price * 0.80,
                        stop_loss_price=last_price * 1.05
                    )
                    self.submit_order(order)
                    logging.info(f"SELL order submitted: {quantity} shares at ${last_price}")
                    self.last_trade = "sell"
        except Exception as e:
            logging.error(f"Trade execution error: {e}")


# --- Backtesting Configuration ---
if __name__ == "__main__":
    try:
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        broker = Alpaca(ALPACA_CREDS)

        strategy = MLTrader(
            name="mlstrat",
            broker=broker,
            parameters={"symbol": "SPY", "cash_at_risk": 0.5}
        )

        strategy.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
            parameters={"symbol": "SPY", "cash_at_risk": 0.5}
        )
    except Exception as e:
        logging.critical(f"Failed to start backtest: {e}")
