import os,glob
path = r"/Users/yymacpro13/Desktop/CINE_SR_DATA/acdc/slide/valid/"
path_list=os.listdir(path)
path_list.sort()
for i in path_list:
    print(i)