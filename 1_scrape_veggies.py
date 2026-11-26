import os
from icrawler.builtin import BingImageCrawler

CLASSES = {
    "bitter_gourd": "fresh bitter gourd karela isolated white background",
    "okra": "fresh green okra ladyfinger bhindi isolated white background",
    "brinjal": "fresh purple brinjal eggplant isolated white background",
    "ridge_gourd": "fresh ridge gourd turai isolated white background",
    "bottle_gourd": "fresh bottle gourd lauki isolated white background"
}

RAW_DIR = "raw_dataset"
IMAGES_PER_CLASS = 95

def scrape_images():
    os.makedirs(RAW_DIR, exist_ok=True)
    for class_name, query in CLASSES.items():
        save_dir = os.path.join(RAW_DIR, class_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n--- Scraping ~{IMAGES_PER_CLASS} images for: {class_name} ---")
        
        crawler = BingImageCrawler(
            downloader_threads=4,
            storage={"root_dir": save_dir}
        )
        crawler.crawl(
            keyword=query,
            max_num=IMAGES_PER_CLASS,
            min_size=(300, 300),
            file_idx_offset=0
        )

if __name__ == "__main__":
    scrape_images()
    print("\nScraping complete! Quickly review raw_dataset/ and delete any irrelevant graphics/text banners.")