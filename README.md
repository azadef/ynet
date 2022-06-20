
# $$\Upsilon\text{-Net}$$

## Environment
The requirements.txt file includes the required libraries for this project.

	python -m venv ynet
	source ./ynet/bin/activate
	pip install -r requirements.txt

## Data Preproccesing

Downloads the dataset, creates the required data directories and preprocesses the data:

    sh data_download_and_preprocess.sh

## Model Evaluation
Evaluate the pre-trained models:

    python eval.py
    
This will report the quantitative comparison between UNet and Y-Net + FFC and save the qualitative comparison to "./figs".
To compare the model parameters, set --print_params to True.


## Model Training
Train the Y-net + FFC model:

    python train.py --dataset [Duke | UMN]

Train the plain Y-net model:

    python train.py --model_name y_net_gen --dataset [Duke | UMN]
    
For UMN, set the number of classes to 2, and the correct data path:

    python train.py --dataset UMN --n_classes 2 --image_dir path_to_UMN
    
Train plain U-Net model:

    python train.py --model_name unet --dataset [Duke | UMN]
