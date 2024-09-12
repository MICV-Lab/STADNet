clc
clear

path = '/Users/yymacpro13/Desktop/VRB_cine/mask/ran_dc_mask_10x.npy';

mask = readNPY(path);
mask = fftshift(mask);
for i=1:4
    mask = cat(3,mask,mask);
end
mask = mask(:,:,1:15);
save(['/Users/yymacpro13/Desktop/VRB_cine/mask/dc_mask_10x.mat'],'mask');

