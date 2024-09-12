clc
clear

load('test_eval.mat');

gt_real = squeeze(gt(:,:,1,:,:));
gt_imag = squeeze(gt(:,:,2,:,:));
gt_complex = complex(gt_real,gt_imag);
gt_abs = abs(gt_complex);

lr_real = squeeze(input(:,:,1,:,:));
lr_imag = squeeze(input(:,:,2,:,:));
lr_complex = complex(lr_real,lr_imag);
lr_abs = abs(lr_complex);

recon_real = squeeze(recon(:,:,1,:,:));
recon_imag = squeeze(recon(:,:,2,:,:));
recon_complex = complex(recon_real,recon_imag);
recon_abs = abs(recon_complex);


% for i=10
%     
%     gt = squeeze(gt_abs(i,:,:));
%     gt = gt/max(gt(:));
%     
%     lr = squeeze(lr_abs(i,:,:));
%     lr = lr/max(lr(:));
%     
%     ours = squeeze(recon_abs(i,:,:));
%     ours = ours/max(ours(:));
%     
%     
% 
% %%%%%%%%%%%%% calculate error map %%%%%%%%%%%%%
%     our_error = abs(gt-ours);
%     our_error = our_error/max(our_error(:));
% 
%     figure;
%     subplot(1,4,1);imshow(lr,[]);title('lr');
%     subplot(1,4,2);imshow(ours,[]);title('recon');
%     subplot(1,4,3);imshow(our_error,[]);title('recon-error');
%     subplot(1,4,4);imshow(gt,[]);title('gt');
%        
% 
%     
% end

