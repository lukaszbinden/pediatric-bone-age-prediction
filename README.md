# Pediatric Bone Age Prediction
Based on kaggle RSNA Bone Age Prediction from X-Rays

https://www.kaggle.com/kmader/rsna-bone-age

## Results of transfer learning experiments
Pretrain the model on the much larger chest NIH dataset (~118k images) and finetune on the small bone age dataset (~ 14k). The model is based on the RSNA challenge winner model by 16Bit. The training size of the hand X-Rays dataset is 12'611 images for all experiments.

| Experiment  | Chest training size | Epochs | MAE |  Date |
| ------------- | ------------- | ------------- | ------------- |  ------------- |
| Imagenet								| n/a	| 50	| 76.8	| 181020  |
| Imagenet								| n/a	| 250	| 8.8	| 181029  |
| No TL, random init.					| n/a	| 250	| 10.8	| 181030  |
| Chest 0-20yrs., 30 layers finetuning	| 1560	| 50	| 33.9	| 181022  |
| Chest 0-20yrs., 100 layers finetuning	| 1560	| 50	| 37	| 181022  |
| Chest 0-20yrs., 100 layers finetuning	| 1560	| 250	| 41.8	| 181024  |
| Chest 0-20yrs., 50 layers finetuning	| 1560	| 250	| 34.5	| 181025  |
| Chest 0-20yrs., 20 layers finetuning	| 1560	| 250	| 36.7	| 181026  |
| Chest 0-20yrs., 30 layers finetuning	| 1560	| 250	| 35.8	| 181027  |
| Chest 0-100yrs, 3 layers finetuning	| 89696	| 250	| pending	| pending |
| Chest 0-100yrs, all layers finetuning	| 89696	| 250	| pending	| pending |

## Experiments
* Try transfer learning with other medical datasets (MURA stanford, etc. )
* Try Preprocessing of images (noise, rotation, etc. ) -> possible with keras 
* Try different architectures of the used net
* Make model highlight joints 
* Incorporate gender into model
* Build model on top of model to predict accuracy of age model
* Try combining different architectures/approaches with meta learning (e.g. voting)
* Chest XRays validate against disease and patient age respectively
* Chest XRays take all samples vs. only age in boneage range
* Experiment with different number of freezed layers
* Experiment difference if pretrained on imagenet or not
* Does including gender as input improve the result?
* Experiment with different hyperparameters
* Regression vs. Classification on months range between 0 and 12 * 100
* Pretrain Kevin's baseline with NIH chest dataset (transfer learning)

## Links
* MURA dataset: https://stanfordmlgroup.github.io/projects/mura/
* Chest Xrays dataset on box (full set): https://nihcc.app.box.com/v/ChestXray-NIHCC
* Chest Xrays dataset on kaggle (5% sample): https://www.kaggle.com/nih-chest-xrays/data
* Bone Age dataset: https://stanfordmedicine.app.box.com/s/4r1zwio6z6lrzk7zw3fro7ql5mnoupcv/folder/42459416739
* RSNA Bone Age on kaggle: https://www.kaggle.com/kmader/rsna-bone-age  
* Different architectures to try out: https://keras.io/applications/
* 2017 RSNA pediatric bone age challenge: http://rsnachallenges.cloudapp.net/competitions/4
* 16Bit challenge winner: https://www.16bit.ai/blog/ml-and-future-of-radiology
