import os
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import mlflow
import mlflow.sklearn
import mlflow.tensorflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = config["data_path"]
        self.models_dir = config["models_dir"]
        self.mlflow_tracking_uri = config["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.data_path}")
        return pd.read_csv(self.data_path)

    def preprocess_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Preprocessing data")
        X = df["text"]
        y_sentiment = df["sentiment"]
        y_user_segment = df["user_segment"]

        (
            X_train,
            X_test,
            y_sentiment_train,
            y_sentiment_test,
            y_segment_train,
            y_segment_test,
        ) = train_test_split(
            X, y_sentiment, y_user_segment, test_size=0.2, random_state=42
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_sentiment_train": y_sentiment_train,
            "y_sentiment_test": y_sentiment_test,
            "y_segment_train": y_segment_train,
            "y_segment_test": y_segment_test,
        }

    def train_sentiment_model(self, X_train, y_train, X_test, y_test):
        logger.info("Training sentiment analysis model")
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42)

        with mlflow.start_run(run_name="sentiment_model"):
            model.fit(X_train_vectorized, y_train)
            y_pred = model.predict(X_test_vectorized)

            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Sentiment Model Accuracy: {accuracy}")

            mlflow.log_param("max_features", 5000)
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", accuracy)

            mlflow.sklearn.log_model(model, "sentiment_model")

        dump(vectorizer, os.path.join(self.models_dir, "tfidf_vectorizer.joblib"))
        return model, vectorizer

    def train_personalization_model(self, X_train, y_train, X_test, y_test):
        logger.info("Training personalization model")
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        max_length = 100
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

        model = Sequential(
            [
                Embedding(5000, 128, input_length=max_length),
                LSTM(64),
                Dense(32, activation="relu"),
                Dense(len(np.unique(y_train)), activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        with mlflow.start_run(run_name="personalization_model"):
            history = model.fit(
                X_train_pad,
                y_train,
                epochs=10,
                validation_data=(X_test_pad, y_test),
                batch_size=32,
            )

            test_loss, test_accuracy = model.evaluate(X_test_pad, y_test)
            logger.info(f"Personalization Model Test Accuracy: {test_accuracy}")

            mlflow.log_param("embedding_dim", 128)
            mlflow.log_param("lstm_units", 64)
            mlflow.log_metric("test_accuracy", test_accuracy)

            mlflow.tensorflow.log_model(model, "personalization_model")

        dump(
            tokenizer, os.path.join(self.models_dir, "personalization_tokenizer.joblib")
        )
        return model, tokenizer

    def run(self):
        df = self.load_data()
        preprocessed_data = self.preprocess_data(df)

        sentiment_model, sentiment_vectorizer = self.train_sentiment_model(
            preprocessed_data["X_train"],
            preprocessed_data["y_sentiment_train"],
            preprocessed_data["X_test"],
            preprocessed_data["y_sentiment_test"],
        )

        personalization_model, personalization_tokenizer = (
            self.train_personalization_model(
                preprocessed_data["X_train"],
                preprocessed_data["y_segment_train"],
                preprocessed_data["X_test"],
                preprocessed_data["y_segment_test"],
            )
        )

        logger.info("Model training completed")


if __name__ == "__main__":
    config = {
        "data_path": "data/processed_data.csv",
        "models_dir": "data/models",
        "mlflow_tracking_uri": "http://localhost:5000",
    }
    trainer = ModelTrainer(config)
    trainer.run()
