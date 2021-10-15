import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from great_expectations.dataset import PandasDataset
from great_expectations.profile import BasicDatasetProfiler
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = self._create_db_engine()

    def _create_db_engine(self):
        db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{self.config['db_host']}:{self.config['db_port']}/{self.config['db_name']}"
        return create_engine(db_url)

    def load_data(self, table_name: str) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name}"
        return pd.read_sql(query, self.engine)

    def run_basic_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Running basic data quality checks")
        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
        }

    def run_statistical_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Running statistical checks")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        return {
            "mean": df[numeric_columns].mean().to_dict(),
            "median": df[numeric_columns].median().to_dict(),
            "std_dev": df[numeric_columns].std().to_dict(),
            "min": df[numeric_columns].min().to_dict(),
            "max": df[numeric_columns].max().to_dict(),
        }

    def run_custom_checks(self, df: pd.DataFrame) -> Dict[str, bool]:
        logger.info("Running custom data quality checks")
        checks = {
            "sentiment_values_valid": df["sentiment"]
            .isin(["positive", "negative", "neutral"])
            .all(),
            "confidence_score_range": (df["confidence_score"] >= 0)
            & (df["confidence_score"] <= 1).all(),
            "text_not_empty": df["text"].str.strip().astype(bool).all(),
        }
        return checks

    def run_great_expectations_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Running Great Expectations checks")
        ge_df = PandasDataset(df)

        results = {}
        results["column_values_unique"] = ge_df.expect_column_values_to_be_unique("id")
        results["sentiment_values"] = ge_df.expect_column_values_to_be_in_set(
            "sentiment", ["positive", "negative", "neutral"]
        )
        results["confidence_score_range"] = ge_df.expect_column_values_to_be_between(
            "confidence_score", 0, 1
        )
        results["text_not_null"] = ge_df.expect_column_values_to_not_be_null("text")

        return {k: v.success for k, v in results.items()}

    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Profiling dataset")
        ge_df = PandasDataset(df)
        profiler = BasicDatasetProfiler()
        profile = profiler.profile(ge_df)
        return profile.to_json_dict()

    def generate_report(self, table_name: str) -> Dict[str, Any]:
        df = self.load_data(table_name)
        report = {
            "table_name": table_name,
            "basic_checks": self.run_basic_checks(df),
            "statistical_checks": self.run_statistical_checks(df),
            "custom_checks": self.run_custom_checks(df),
            "great_expectations_checks": self.run_great_expectations_checks(df),
            "dataset_profile": self.profile_dataset(df),
        }
        return report

    def save_report(self, report: Dict[str, Any], filename: str):
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {filename}")


def main():
    config = {"db_host": "localhost", "db_port": 5432, "db_name": "sentimax"}

    checker = DataQualityChecker(config)
    report = checker.generate_report("sentiment_results")
    checker.save_report(report, "data_quality_report.json")


if __name__ == "__main__":
    main()
