from CPTSuperLearn.utils import download_file, extract_zip


def download_schemaGAN():
    """
    Download the SchemaGAN model
    """
    url = "https://zenodo.org/records/13143431/files/schemaGAN.h5"
    download_file(url, "./schemaGAN/schemaGAN.h5")

def download_data():
    """
    Download the data for training and validation
    """
    url = "https://zenodo.org/records/13143431/files/data.zip"
    download_file(url, "data.zip")
    extract_zip("data.zip", "./")


if __name__ == "__main__":
    download_schemaGAN()
    #download_data()