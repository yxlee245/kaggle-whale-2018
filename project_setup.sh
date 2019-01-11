# Create folders for data and code base (point to working directory first)
mkdir data
mkdir src
cd data
mkdir raw
mkdir derived

# Download data from Kaggle API (enable virtual environment first and save API token in .kaggle/ first)
chmod 600 ~/.kaggle/kaggle.json # disallow other system users from accessing kaggle API key
cd raw
kaggle competitions download -c humpback-whale-identification
sudo apt-get install unzip
unzip train.zip -d train
unzip test.zip -d test
cd ../..
