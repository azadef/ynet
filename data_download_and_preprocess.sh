#Download Duke dataset
wget http://www.duke.edu/~sf59/Datasets/2015_BOE_Chiu2.zip

#Download UMN dataset
wget http://people.ece.umn.edu/users/parhi/.DATA/OCT/DME/UMNDataset.mat

#Unzip Downloaded File
unzip 2015_BOE_Chiu2.zip

#Prepare and create the required data directories
mkdir -p DukeData
mkdir -p DukeData/train DukeData/val DukeData/test

mkdir -p UMNData
mkdir -p UMNData/train UMNData/val UMNData/test

#Run preprocessing code
python preprocessing.py "./2015_BOE_Chiu" "DukeData"
python preprocessing.py --dataset "UMN" "UMNDataset.mat" "UMNData"
