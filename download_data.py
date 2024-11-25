import os
import requests
from tqdm import tqdm
import zipfile


def _download_file(url: str, file_name: str):
    """
    Download a file from a URL and save it to the specified location.

    Parameters:
    -----------
    url (str): URL to download the file from
    file_name (str): Name of the file to save
    """

    response = requests.get(url, stream=True)
    response.raise_for_status()

    if os.path.dirname(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

    total_size = int(response.headers.get('content-length', 0))
    pbar = tqdm(total=total_size, desc=f"Downloading {file_name}", unit="B", unit_scale=True, unit_divisor=1024)

    # Write the content to a file
    with open(file_name, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            pbar.update(8192)

    pbar.close()
    print(f"Downloaded {file_name} successfully")


def _extract_zip(data_zip, extract_path):
    """
    Extract a ZIP file to the specified location with progress bar.

    Parameters:
    -----------
    data_zip (str): Path to the ZIP file
    extract_path (str): Path to extract the files to
    """
    os.makedirs(extract_path, exist_ok=True)

    # Open the ZIP file once
    with zipfile.ZipFile(data_zip, "r") as zip_ref:
        # Get list of files to extract
        file_list = zip_ref.namelist()

        pbar = tqdm(total=len(file_list), desc=f"Extracting {data_zip}", unit="file")
        # Create progress bar
        # with tqdm(total=len(file_list), desc=f"Extracting {data_zip}", unit="file") as pbar:
        for file in file_list:
            zip_ref.extract(file, extract_path)
            pbar.update(1)

    print(f"Files extracted to {extract_path}")

    # Clean up
    os.remove(data_zip)


def download_schemaGAN():
    url = "https://zenodo.org/records/13143431/files/schemaGAN.h5"
    _download_file(url, "./schemaGAN/schemaGAN.h5")

def download_data():
    url = "https://zenodo.org/records/13143431/files/data.zip"
    _download_file(url, "data.zip")
    _extract_zip("data.zip", "./data")


if __name__ == "__main__":
    download_schemaGAN()
    download_data()