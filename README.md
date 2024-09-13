# STADNet: Spatial-Temporal Attention-Guided Dual-Path Network for cardiac cine MRI super-resolution (MIA 2024)
![alt](https://github.com/MICV-Lab/STADNet/blob/main/Framework.png "Framework")


The overall architecture of the proposed STADNet network, which consists of two branches, the location-aware spatial path, and the motion-aware temporal path. The location-aware spatial path employs the information of neighboring frames to enhance the spatial details of the current frame. The motion-aware temporal path utilizes an optical flow-based strategy to exploit the correlation between cine MR frames and extract the motion information of the cardiac. We introduce a sliding-window mechanism with a window size of 3 and a stride of 1. Thus, the network’s input is three consecutive frames, 𝑡 − 1, 𝑡, 𝑡 + 1


### Data Description
Our experiments utilize one in-house private dataset: SAX, and one public dataset: ACDC. All studies have been approved by the local Institutional Review Board (IRB) for the private data. The IRB asked us to protect the privacy of participants and to maintain the confidentiality of data. So we cannot make the private dataset publicly available. 

### Installation
```bash
git clone https://github.com/MICV-Lab/STADNet.git
```
All experiments in our paper were conducted on the NVIDIA Tesla A100 GPU with an identical experimental setting.

### Environment and Dependencies
Requirements:
* Python 3.6
* Pytorch 1.4.0 
* scipy
* scikit-image
* opencv-python
* tqdm

Our code has been tested with Python 3.6, Pytorch 1.4.0, torchvision 0.5.0, CUDA 10.0 on Ubuntu 18.04.


### Usage

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

### Citation
If you used this code in your research, please cite the following works:
```bash
@article{lyu2024stadnet,
  title={STADNet: Spatial-Temporal Attention-Guided Dual-Path Network for cardiac cine MRI super-resolution},
  author={Lyu, Jun and Wang, Shuo and Tian, Yapeng and Zou, Jing and Dong, Shunjie and Wang, Chengyan and Aviles-Rivero, Angelica I and Qin, Jing},
  journal={Medical Image Analysis},
  volume={94},
  pages={103142},
  year={2024},
  publisher={Elsevier}
}
```


