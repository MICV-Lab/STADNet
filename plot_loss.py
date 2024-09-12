import csv
import matplotlib.pyplot as plt
import numpy as np

loss = []
with open('train_loss.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        loss.append(float(row[0]))
epcoh = range(0,len(loss))
loss=np.array(loss)


train_loss_lines = plt.plot(epcoh, loss, 'r', lw=1)#lw为曲线宽度

plt.title("")
plt.xlabel("epoch")
plt.ylabel("loss")

plt.show()