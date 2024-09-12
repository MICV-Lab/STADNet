import csv
import matplotlib.pyplot as plt
import numpy as np

epoch = []
PSNR = []
SSIM = []
with open('metrics.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        PSNR.append(float(row[1]))
        SSIM.append(float(row[2]))
        epoch.append(float(row[0]))

epoch=np.array(epoch)
PSNR=np.array(PSNR)
SSIM=np.array(SSIM)

mean = np.mean(PSNR)
std = np.std(PSNR)
print(mean,std)

mean = np.mean(SSIM)
std = np.std(SSIM)
print(mean,std)

train_loss_lines = plt.plot(epoch, PSNR, 'r', lw=1)#lw为曲线宽度

plt.title("")
plt.xlabel("epoch")
plt.ylabel("PSNR")

plt.show()