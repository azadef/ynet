## Y-Net: A Spatiospectral Network for Retinal OCT Segmentation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/u-net-a-spatiospectral-network-for-retinal/retinal-oct-layer-segmentation-on-duke-sd-oct)](https://paperswithcode.com/sota/retinal-oct-layer-segmentation-on-duke-sd-oct?p=u-net-a-spatiospectral-network-for-retinal)

This is the official source code for the paper "$\Upsilon$-Net: A Spatiospectral Network for Retinal OCT Segmentation" accepted in MICCAI 2022.

<a href="https://arxiv.org/pdf/2204.07613.pdf">Link to arXiv</a>

Authors: <a href="https://www.in.tum.de/campar/members/azade-farshad/">A. Farshad*</a>, <a href="https://campar.in.tum.de/Main/YousefYeganeh">Y. Yeganeh*</a>, <a href="https://www.hopkinsmedicine.org/profiles/details/peter-gehlbach">P. Gehlbach</a>, <a href="https://www.in.tum.de/campar/members/cv-nassir-navab/nassir-navab/">N. Navab</a>


## Citation
If you find this code useful in your research then please cite:
```
@inproceedings{farshad2022_ynet,
	    title={Υ-Net: A Spatiospectral Network for Retinal OCT Segmentation},
	    author={Farshad, Azade and Yeganeh, Yousef and Gehlbach, Peter and Navab, Nassir},
	    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
	    year={2022},
	    organization={Springer}
	  }
```

## Environment
The requirements.txt file includes the required libraries for this project.

	python -m venv ynet
	source ./ynet/bin/activate
	pip install -r requirements.txt

## Datasets Downloading and Preproccesing

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
