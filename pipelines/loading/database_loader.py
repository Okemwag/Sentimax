import os
import logging
from typing import Dict, Any, List
import pandas as pd
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pg_engine = self._create_pg_engine()
        self.mongo_client = self._create_mongo_client()

    def _create_pg_engine(self):
        pg_url = f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{self.config['pg_host']}:{self.config['pg_port']}/{self.config['pg_database']}"
        return create_engine(pg_url)

    def _create_mongo_client(self):
        mongo_url = f"mongodb://{os.getenv('MONGO_USER')}:{os.getenv('MONGO_PASSWORD')}@{self.config['mongo_host']}:{self.config['mongo_port']}"
        return MongoClient(mongo_url)

    def load_structured_data(self, data: pd.DataFrame, table_name: str):
        logger.info(f"Loading structured data into PostgreSQL table: {table_name}")
        try:
            metadata = MetaData()
            table = Table(
                table_name,
                metadata,
                Column("id", Integer, primary_key=True),
                Column("text", String),
                Column("sentiment", String),
                Column("user_segment", String),
                Column("confidence_score", Float),
            )
            metadata.create_all(self.pg_engine)

            with self.pg_engine.connect() as conn:
                data.to_sql(table_name, conn, if_exists="append", index=False)
            logger.info(f"Successfully loaded {len(data)} rows into {table_name}")
        except SQLAlchemyError as e:
            logger.error(f"Error loading data into PostgreSQL: {str(e)}")
            raise

    def load_unstructured_data(self, data: List[Dict[str, Any]], collection_name: str):
        logger.info(
            f"Loading unstructured data into MongoDB collection: {collection_name}"
        )
        try:
            db = self.mongo_client[self.config["mongo_database"]]
            collection = db[collection_name]
            result = collection.insert_many(data)
            logger.info(
                f"Successfully inserted {len(result.inserted_ids)} documents into {collection_name}"
            )
        except PyMongoError as e:
            logger.error(f"Error loading data into MongoDB: {str(e)}")
            raise

    def load_sentiment_results(self, results: pd.DataFrame):
        logger.info("Loading sentiment analysis results")
        self.load_structured_data(results, "sentiment_results")

    def load_user_segments(self, segments: pd.DataFrame):
        logger.info("Loading user segmentation results")
        self.load_structured_data(segments, "user_segments")

    def load_raw_feedback(self, feedback: List[Dict[str, Any]]):
        logger.info("Loading raw customer feedback")
        self.load_unstructured_data(feedback, "raw_feedback")

    def load_model_performance(self, performance_data: Dict[str, Any]):
        logger.info("Loading model performance data")
        self.load_unstructured_data([performance_data], "model_performance")

    def execute_custom_query(self, query: str):
        logger.info("Executing custom query")
        try:
            with self.pg_engine.connect() as conn:
                result = conn.execute(query)
                return result.fetchall()
        except SQLAlchemyError as e:
            logger.error(f"Error executing custom query: {str(e)}")
            raise

    def close_connections(self):
        logger.info("Closing database connections")
        self.pg_engine.dispose()
        self.mongo_client.close()


def main():
    config = {
        "pg_host": "localhost",
        "pg_port": 5432,
        "pg_database": "sentimax",
        "mongo_host": "localhost",
        "mongo_port": 27017,
        "mongo_database": "sentimax",
    }

    loader = DatabaseLoader(config)

    try:
        sentiment_results = pd.DataFrame(
            {
                "text": ["Great product!", "Poor service"],
                "sentiment": ["positive", "negative"],
                "user_segment": ["frequent_buyer", "new_customer"],
                "confidence_score": [0.95, 0.87],
            }
        )
        loader.load_sentiment_results(sentiment_results)

        raw_feedback = [
            {
                "user_id": 1,
                "feedback": "The app is user-friendly",
                "timestamp": "2023-07-25T10:00:00Z",
            },
            {
                "user_id": 2,
                "feedback": "I had issues with payment",
                "timestamp": "2023-07-25T11:30:00Z",
            },
        ]
        loader.load_raw_feedback(raw_feedback)

        model_performance = {
            "model_name": "sentiment_analysis_v1",
            "accuracy": 0.85,
            "f1_score": 0.83,
            "timestamp": "2023-07-25T12:00:00Z",
        }
        loader.load_model_performance(model_performance)

    except (SQLAlchemyError, PyMongoError) as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        loader.close_connections()


if __name__ == "__main__":
    main()
