# Pediatric Bone Age Prediction
Based on kaggle RSNA Bone Age Prediction from X-Rays

* http://rsnachallenges.cloudapp.net/competitions/4
* https://www.kaggle.com/kmader/rsna-bone-age

## Results of transfer learning experiments
Pretrain the model on the much larger chest NIH dataset (112k images) and finetune on the small bone age dataset (14k). The model is based on the RSNA challenge winner model by 16Bit.

| Experiment  | Chest training size | Epochs | MAE |  Date |
| ------------- | ------------- | ------------- | ------------- |  ------------- |
| 16Bit								| n/a	| 500	| 4.265	| 2017  |
| Radiologist								| n/a	| n/a	| 7.32	| 2017  |
| Imagenet								| n/a	| 50	| 76.8	| 181020  |
| Imagenet								| n/a	| 250	| **8.8**	| 181029  |
| No TL, random init.					| n/a	| 250	| **10.8**	| 181030  |
| Chest 0-20yrs., 30 layers finetuning	| 1560	| 50	| 33.9	| 181022  |
| Chest 0-20yrs., 100 layers finetuning	| 1560	| 50	| 37	| 181022  |
| Chest 0-20yrs., 100 layers finetuning	| 1560	| 250	| 41.8	| 181024  |
| Chest 0-20yrs., 50 layers finetuning	| 1560	| 250	| 34.5	| 181025  |
| Chest 0-20yrs., 20 layers finetuning	| 1560	| 250	| 36.7	| 181026  |
| Chest 0-20yrs., 30 layers finetuning	| 1560	| 250	| 35.8	| 181027  |
| Chest 0-100yrs, 3 layers finetuning	| 89696	| 250	| pending	| pending |
| Chest 0-100yrs, all layers finetuning	| 89696	| 250	| pending	| pending |

The training/validation split is 80/20. The idea to restrict the chest X-rays patient's age to the same range as the hand X-rays patients turned out unsuccessful as the dataset size decreased significantly to 1560 images and transfer learning results were disappointing.

## Experiments
* Try transfer learning with other medical datasets (MURA stanford, etc. )
* Try different architectures of the used net, e.g. use DenseNet instead of InceptionV3
* Build model on top of model to predict accuracy of age model
* Try combining different architectures/approaches with meta learning (e.g. voting)
* Chest XRays validate against disease and patient age respectively
* Experiment with different number of freezed layers
* Do more extensive hyperparameter tuning
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
