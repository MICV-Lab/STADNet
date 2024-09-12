clc
clear

mask_4 = load('/Users/yymacpro13/Desktop/VRB_cine/mask/dc_mask_4x.mat').mask;
mask_6 = load('/Users/yymacpro13/Desktop/VRB_cine/mask/dc_mask_6x.mat').mask;
mask_8 = load('/Users/yymacpro13/Desktop/VRB_cine/mask/dc_mask_8x.mat').mask;
mask_10 = load('/Users/yymacpro13/Desktop/VRB_cine/mask/dc_mask_10x.mat').mask;

gt_path = '/Users/yymacpro13/Desktop/MRI/cine/crnn_data/new/test/';
save_path = '/Users/yymacpro13/Desktop/VRB_cine/data/test/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train [2,3,6,8,9,12,15,17,20,22,23,24,25,28,31,34,35,36,38,39,40]    %
% valid [44,45,48]                                                     %
% test [1:8]                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for people = 1:8 
for slc = 3:8
    %%%%%%GT%%%%%%
    gt = load(strcat(gt_path,'cine_',num2str(people),'_',num2str(slc),'.mat')).seq;
    gt_k = fft2c(gt);
    gt_i = ifft2c(gt_k);
    %%%%%%ZF4%%%%%%
    zf_4_k = gt_k .* mask_4;
    zf_4 = ifft2c(zf_4_k);
    %%%%%%ZF6%%%%%%
    zf_6_k = gt_k .* mask_6;
    zf_6 = ifft2c(zf_6_k);
    %%%%%%ZF8%%%%%%
    zf_8_k = gt_k .* mask_8;
    zf_8 = ifft2c(zf_8_k);
    %%%%%%ZF10%%%%%%
    zf_10_k = gt_k .* mask_10;
    zf_10 = ifft2c(zf_10_k);
    
    %%%%%% norm %%%%%%
    GT = gt_i/max(abs(gt_i(:)));
    ZF_4 = zf_4/max(abs(zf_4(:)));
    ZF_6 = zf_6/max(abs(zf_6(:)));
    ZF_8 = zf_8/max(abs(zf_8(:)));
    ZF_10 = zf_10/max(abs(zf_10(:)));
    
    %%%%%%% save %%%%%%%%
    save(strcat(save_path,'cine_',num2str(people),'_',num2str(slc),'.mat'),"GT","ZF_4","ZF_6","ZF_8","ZF_10");
    
end
end
