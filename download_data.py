import os
import requests
import zipfile
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET_URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
BASE_DIR = "dataset"
RAW_DIR = os.path.join(BASE_DIR, "raw")
EXTRACT_DIR = os.path.join(BASE_DIR, "2750") # This matches zip internal structure
OUTPUT_DIR = os.path.join(BASE_DIR, "split")

def download_dataset():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    
    zip_path = os.path.join(BASE_DIR, "EuroSAT.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading EuroSAT dataset from {DATASET_URL}...")
        # Bypassing SSL verification due to certificate issues in the environment
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(DATASET_URL, stream=True, verify=False)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    else:
        print("Dataset zip already exists.")

    if not os.path.exists(EXTRACT_DIR):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
    else:
        print("Dataset already extracted.")

def split_dataset():
    if os.path.exists(OUTPUT_DIR):
        print("Dataset already split.")
        return

    print("Splitting dataset into train, val, and test...")
    # EuroSAT structure: EuroSAT_RGB/<class_name>/<image_name>.jpg
    classes = [d for d in os.listdir(EXTRACT_DIR) if os.path.isdir(os.path.join(EXTRACT_DIR, d))]
    
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    for cls in classes:
        cls_path = os.path.join(EXTRACT_DIR, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith('.jpg')]
        
        # 80% train, 10% val, 10% test
        train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(OUTPUT_DIR, 'train', cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(OUTPUT_DIR, 'val', cls, img))
        for img in test_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(OUTPUT_DIR, 'test', cls, img))

    print("Splitting complete.")

if __name__ == "__main__":
    download_dataset()
    split_dataset()
