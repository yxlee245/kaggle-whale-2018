# Import and activate virtual environment for whale identification project
conda env create -f whale_env.yml
conda activate whale_env

# Create folders for data and code base (point to working directory first)
mkdir data
mkdir src
cd data
mkdir raw
mkdir derived

# Download data from Kaggle API (enable virtual environment first and save API token in .kaggle/ first)
cd raw
kaggle competitions download -c humpback-whale-identification
unzip train.zip -d train
unzip test.zip -d test
cd ../..
