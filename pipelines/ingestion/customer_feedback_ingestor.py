import os
import logging
from typing import Dict, Any, List
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CustomerFeedbackIngestor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_connection = self._create_db_connection()

    def _create_db_connection(self):
        return mysql.connector.connect(
            host=self.config["db_host"],
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=self.config["db_name"],
        )

    def ingest_feedback_data(self, days_ago: int = 7) -> List[Dict[str, Any]]:
        logger.info(f"Ingesting customer feedback data from the last {days_ago} days")
        query = f"""
        SELECT customer_id, feedback_text, rating, submission_date
        FROM customer_feedback
        WHERE submission_date >= DATE_SUB(CURDATE(), INTERVAL {days_ago} DAY)
        """
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(query)
            data = cursor.fetchall()
            logger.info(f"Successfully ingested {len(data)} customer feedback entries")
            return data
        except mysql.connector.Error as e:
            logger.error(f"Error ingesting customer feedback data: {str(e)}")
            return []
        finally:
            cursor.close()

    def ingest_survey_data(self, survey_id: int) -> List[Dict[str, Any]]:
        logger.info(f"Ingesting survey data for survey ID: {survey_id}")
        query = f"""
        SELECT response_id, question_id, answer_text, submission_date
        FROM survey_responses
        WHERE survey_id = {survey_id}
        """
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(query)
            data = cursor.fetchall()
            logger.info(f"Successfully ingested {len(data)} survey responses")
            return data
        except mysql.connector.Error as e:
            logger.error(f"Error ingesting survey data: {str(e)}")
            return []
        finally:
            cursor.close()

    def save_to_csv(self, data: List[Dict[str, Any]], filename: str):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")

    def close_connection(self):
        if self.db_connection.is_connected():
            self.db_connection.close()
            logger.info("Database connection closed")


def main():
    config = {
        "db_host": "localhost",
        "db_name": "customer_feedback_db",
        "survey_id": 123,
    }

    ingestor = CustomerFeedbackIngestor(config)

    try:
        feedback_data = ingestor.ingest_feedback_data(days_ago=30)
        ingestor.save_to_csv(feedback_data, "customer_feedback.csv")

        survey_data = ingestor.ingest_survey_data(config["survey_id"])
        ingestor.save_to_csv(survey_data, "survey_responses.csv")

    finally:
        ingestor.close_connection()


if __name__ == "__main__":
    main()
