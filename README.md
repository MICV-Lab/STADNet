<<<<<<< HEAD
### Data Description
Two in-house private datasets: pelvic and brain and one public dataset: fastMRI are utilized in our experiments. For the private data, all studies have been approved by the local Institutional Review Board (IRB). The IRB asked us to protect the privacy of participants and to maintain the confidentiality of data. Since we cannot make the two datasets publicly available, we won't claim them as our contribution.
### Environment and Dependencies
Requirements:
* Python 3.6
* Pytorch 1.4.0 
* scipy
* scikit-image
* opencv-python
* tqdm

Our code has been tested with Python 3.6, Pytorch 1.4.0, torchvision 0.5.0, CUDA 10.0 on Ubuntu 18.04.


### To Run Our Code
- Train the model
```bash
nohup python train_demo.py > sr_4x_new.out 2>&1 &
nohup python train_acdc.py > acdc_sr_4x.out 2>&1 &
```

- Test the model
```bash
 python test_demo.py --resume ' '
```
where
`--resume`  trained model. 

nohup python train_new_mask.py > new_mask_0330.out 2>&1 &

nohup python train_brain_4x.py > train_brain_4x.out 2>&1 &

=======
# STADNet
>>>>>>> dbcbe79ff35ee9d7fdacd65c0dbb12e71dac4463
