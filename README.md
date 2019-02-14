# Kaggle Competition: Humpback Whale Identification

## Problem
Identify individual whales in images
[Kaggle competition: Humpback Whale Identification](https://www.kaggle.com/c/humpback-whale-identification)

## Reproducing this project
__Note__:
- The terminal commands here will be for Ubuntu, as I'm more familiar with the commands. Please search for the relevant alternatives if you're using Windows or Mac.
- I assume that you have GPU and CUDA set up in your physical machine / VM for deep learning. If you don't, the Tesnsorflow installation for GPU will fail to run properly.

Assuming that you have Anaconda installed:
1. Create a folder named `.kaggle` (hidden folder) in your machine's home directory.
2. Download the `kaggle.json` file containing your Kaggle account info, and move it into the `.kaggle` folder. (You can find out about downloading the `kaggle.json` [here](https://github.com/Kaggle/kaggle-api).)
(First 2 steps are needed to download the data via the terminal using the Kaggle API.)
3. In the terminal, navigate to the directory where you want to clone this repo to, by using the command `cd <your directory>`.
4. Clone this repo by the command `git clone https://github.com/yxlee245/kaggle-whale-2018.git`.
5. Navigate to the local repo by the command `cd kaggle-whale-2018`.
6. Import the environment from the file `whale_env.yml`, using the command `conda env create -f whale_env.yml`.
7. Run the shell script `project_setup.sh` to create the `data` folder, then download and unpack the raw data.
8. Run the Jupyter notebooks as needed.

__Extra__: `drafts` folder created to hold previous attempts of creating the recognition model