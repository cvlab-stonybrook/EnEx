# EnEx

This repository contains datasets, extracted features, and Python implementation of paper *Zekun Zhang, Farrukh M. Koraishy, Minh Hoai, Exemplar-Based Early Event Prediction in Video, BMVC 2021* [[Paper](https://github.com/cvlab-stonybrook/EnEx/blob/main/media/paper.pdf)] [[3min Intro](https://github.com/cvlab-stonybrook/EnEx/blob/main/media/poster.mp4)] [[Supp PDF](https://github.com/cvlab-stonybrook/EnEx/blob/main/media/supp.pdf)] [[Supp Video](https://github.com/cvlab-stonybrook/EnEx/blob/main/media/supp.mp4)].

## Datasets and Extracted Features

After cloning this repository, please download our custom datasets and extracted feature files from [Google Drive](https://drive.google.com/drive/folders/1YiQ1tqAdoRQaZUB_pbH7VOuhD5cHCVnt). Then put the files in the corresponding sub-directories in `datasets`. We do not publish the COVID19-AKI datasets due to concerns about patients' privacy.

## Dependencies

Our implementation depends on several common Python libraries: `numpy matplotlib scikit-learn scikit-learn-extra scipy`. The code has been tested on Python 3.8, but it should work with other versions too.

## EnEx Model Training

We provide dataloader, training, and evaluation code for the *OpenDoor*, *PhonePickup*, and *Epic-Kitchens* datasets. For example, to train and evaluate EnEx model on *OpenDoor* dataset at lead time of 2 seconds for 10 runs with different random data shuffling, please execute
```
python train_eval.py --dataset door --L 5 --T 2 --L2 5 --runs 10
```
The evaluation results will be saved as a JSON file named `log_door_L_5.0_T_2.0_L2_5.0.json`. Similary, to train and evaluation on *Epic-Kitchen* action 18, please run
```
python train_eval.py --dataset epic --action_id 18 --L 5 --T 1 --L2 5 --runs 10
```
The results will be saved in file `log_epic_turn-ontap_L_5_T_1_L2_5.json`. Please note that if you run the experiments with the same configuration, the new results will be appended to the existing JSON file.

## Visualization of Results

You can plot the average precision at different recall thresholds averaged across different runs from the JSON files generated by the training and evaluation process. For example
```
python visualize.py --files log_door_L_5.0_T_2.0_L2_5.0.json
```
This will generate a PDF file named `log_door_L_5.0_T_2.0_L2_5.0_runs_10.pdf` in the same directory as the JSON file.

## Citation

If you find our paper useful, please cite
```
@inproceedings{ZhangBMVC2021, 
  Author = {Zekun Zhang and Farrukh M. Koraishy and Minh Hoai}, 
  Booktitle = {British Machine Vision Conference}, 
  Title = {Exemplar-Based Early Event Prediction in Video}, 
  Year = {2021}
}
```
