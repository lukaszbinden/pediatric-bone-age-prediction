# jmcs-atml-bone-age-prediction
Based on kaggle RSNA Bone Age Prediction from X-Rays

https://www.kaggle.com/kmader/rsna-bone-age

# Cluster
* Pdf: How_to_use_Cluster

Run File: 
* /var/tmp/studi5/boneage/git/jmcs-atml-bone-age-prediction/src/baseline/
* see .bash_profile for environment configuration
* Run the transfer learning program:
* [studi5@node03 baseline]$ p transfer_learning.py 2>&1 | tee ~/tf_testrun.txt &
* -> run the program as a separate (non-child) process from the bash process and copy all output to stdout to file ~/tf_testrun.txt as well

# Tasks
* Port the existing code of one of kevin maders kernels to torch so we can see our starting point (must be done fast!) -> use keras directly
* Try transfer learning with other medical datasets (mura stanford, chestxray nih, etc. )
* Try Preprocessing of images (noise, rotation, etc. ) -> possible with keras 
* Try different architectures of the used net
* Make model highlight joints 
* Incorporate gender into model
* Build model on top of model to predict accuracy of age model
* Try combining different architectures/approaches with meta learning (e.g. voting)

# Experiments
* Chest XRays validate against disease and patient age respectively
* Chest XRays take all samples vs. only age in boneage range
* Experiment with different number of freezed layers
* Experiment difference if pretrained on imagenet or not
* Does including gender as input improve the result?
* Experiment with different hyperparameters
* Regression vs. Classification on months range between 0 and 12 * 100
* Pretrain Kevin's baseline with NIH chest dataset (transfer learning)

# Links
* MURA dataset: https://stanfordmlgroup.github.io/projects/mura/ -> asked for early access but did not get any response
* Chest Xrays dataset on box (full set): https://nihcc.app.box.com/v/ChestXray-NIHCC
* Chest Xrays dataset on kaggle (5% sample): https://www.kaggle.com/nih-chest-xrays/data
* RSNA Bone Age on kaggle: https://www.kaggle.com/kmader/rsna-bone-age  
   Installed 180426 on studi5@node03 in /var/tmp/studi5/boneage/
* Different architectures to try out: https://keras.io/applications/