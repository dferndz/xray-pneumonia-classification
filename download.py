import urllib.request
import tarfile

DATASET_URL = "https://github.com/dferndz/xray-pneumonia-classification/releases/download/v1.0-dataset/xray-pneumonia.tar.xz"
LOCAL_FILE = "dataset.tar.xz"

if __name__ == "__main__":
    urllib.request.urlretrieve(DATASET_URL, LOCAL_FILE)
    tarfile.open(LOCAL_FILE, "r:xz").extractall()