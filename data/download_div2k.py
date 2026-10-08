import os
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(data_dir, 'DIV2K_valid_HR.zip')
    extract_dir = os.path.join(data_dir, 'DIV2K')
    
    # URL for DIV2K validation set (contains exactly 100 images)
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        
    if not os.path.exists(zip_path):
        print(f"Downloading {url}...")
        download_url(url, zip_path)
    else:
        print(f"{zip_path} already exists. Skipping download.")
        
    extracted_folder = os.path.join(extract_dir, 'DIV2K_valid_HR')
    if not os.path.exists(extracted_folder):
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete.")
    else:
        print("Files already extracted.")

if __name__ == '__main__':
    main()
