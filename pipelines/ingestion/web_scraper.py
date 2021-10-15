import os
import logging
from typing import Dict, Any, List
import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WebScraper:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.driver = self._init_selenium_driver()

    def _init_selenium_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        return webdriver.Chrome(options=chrome_options)

    def scrape_static_page(self, url: str) -> Dict[str, Any]:
        logger.info(f"Scraping static page: {url}")
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            
            product_name = soup.find("h1", class_="product-name").text.strip()
            product_price = soup.find("span", class_="price").text.strip()
            product_description = soup.find("div", class_="description").text.strip()

            return {
                "name": product_name,
                "price": product_price,
                "description": product_description,
            }
        except requests.RequestException as e:
            logger.error(f"Error scraping static page: {str(e)}")
            return {}

    def scrape_dynamic_page(self, url: str) -> List[Dict[str, Any]]:
        logger.info(f"Scraping dynamic page: {url}")
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "review-item"))
            )

            
            review_elements = self.driver.find_elements(By.CLASS_NAME, "review-item")
            reviews = []
            for element in review_elements:
                author = element.find_element(By.CLASS_NAME, "author").text
                rating = element.find_element(By.CLASS_NAME, "rating").get_attribute(
                    "data-rating"
                )
                content = element.find_element(By.CLASS_NAME, "content").text
                reviews.append({"author": author, "rating": rating, "content": content})
            return reviews
        except Exception as e:
            logger.error(f"Error scraping dynamic page: {str(e)}")
            return []

    def save_to_csv(self, data: List[Dict[str, Any]], filename: str):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")

    def close_driver(self):
        if self.driver:
            self.driver.quit()
            logger.info("Selenium WebDriver closed")


def main():
    config = {
        "static_url": "https://share-hub.co/product/123",
        "dynamic_url": "https://share-hub.co/reviews/123",
    }

    scraper = WebScraper(config)

    try:
        static_data = scraper.scrape_static_page(config["static_url"])
        scraper.save_to_csv([static_data], "product_info.csv")

        dynamic_data = scraper.scrape_dynamic_page(config["dynamic_url"])
        scraper.save_to_csv(dynamic_data, "product_reviews.csv")

    finally:
        scraper.close_driver()


if __name__ == "__main__":
    main()
