clear;
clc;


path = dir('/Users/yymacpro13/Desktop/VRB_cine_sr/data/test');
for frames = 1:5


for j=1:51-3
    data_name = path(j+3).name;

    GT = load(strcat('/Users/yymacpro13/Desktop/VRB_cine_sr/data/test/',num2str(data_name),'')).GT; 

    data_abs_gt = abs(GT);
    HR = uint8(255*abs(data_abs_gt)/max(abs(data_abs_gt(:))));
    %%%%%%%%%%%%%%% preparing LR2
    gt_ks=fft2c(GT);
    lr2_ks = gt_ks(64:191,64:191,:);
    lr2 = ifft2c(lr2_ks);
    lr2 = abs(lr2);
    LR2 = uint8(255*abs(lr2)/max(abs(lr2(:))));
    %%%%%%%%%%%%%%% preparing LR4
    lr4_ks = lr2_ks(32:95,32:95,:);
    lr4 = ifft2c(lr4_ks);
    lr4 = abs(lr4);
    LR4 = uint8(255*abs(lr4)/max(abs(lr4(:))));
    
    %%%%%%%% get 3 frames %%%%%%%%
    empety_hr = ones(256,256,3);
    empety_lr2 = ones(128,128,3);
    empety_lr4 = ones(64,64,3);
    
    if(frames==1)
        for i = 1:3 %4:6 7:9 10:12 13:15
            HR_ = imadjust(squeeze(HR(:,:,i)));
            empety_hr(:,:,i) = HR_;

            LR2_ = imadjust(squeeze(LR2(:,:,i)));
            empety_lr2(:,:,i) = LR2_;

            LR4_ = imadjust(squeeze(LR4(:,:,i)));
            empety_lr4(:,:,i) = LR4_;
        end
    end
    
    if(frames==2)
        for i = 4:6 %7:9 10:12 13:15
            HR_ = imadjust(squeeze(HR(:,:,i)));
            empety_hr(:,:,i-3) = HR_;

            LR2_ = imadjust(squeeze(LR2(:,:,i)));
            empety_lr2(:,:,i-3) = LR2_;

            LR4_ = imadjust(squeeze(LR4(:,:,i)));
            empety_lr4(:,:,i-3) = LR4_;
        end
    end
    
    if(frames==3)
        for i = 7:9 %10:12 13:15
            HR_ = imadjust(squeeze(HR(:,:,i)));
            empety_hr(:,:,i-6) = HR_;

            LR2_ = imadjust(squeeze(LR2(:,:,i)));
            empety_lr2(:,:,i-6) = LR2_;

            LR4_ = imadjust(squeeze(LR4(:,:,i)));
            empety_lr4(:,:,i-6) = LR4_;
        end
    end
    
    if(frames==4)
        for i = 10:12 %13:15
            HR_ = imadjust(squeeze(HR(:,:,i)));
            empety_hr(:,:,i-9) = HR_;

            LR2_ = imadjust(squeeze(LR2(:,:,i)));
            empety_lr2(:,:,i-9) = LR2_;

            LR4_ = imadjust(squeeze(LR4(:,:,i)));
            empety_lr4(:,:,i-9) = LR4_;
        end
    end
    
    if(frames==5)
        for i = 13:15
            HR_ = imadjust(squeeze(HR(:,:,i)));
            empety_hr(:,:,i-12) = HR_;

            LR2_ = imadjust(squeeze(LR2(:,:,i)));
            empety_lr2(:,:,i-12) = LR2_;

            LR4_ = imadjust(squeeze(LR4(:,:,i)));
            empety_lr4(:,:,i-12) = LR4_;
        end
    end
    
    HR = empety_hr;
    LR_2 = empety_lr2;
    LR_4 = empety_lr4;
    
    %%%%%% save %%%%%%
    
    
    mkdir(strcat('/Users/yymacpro13/Desktop/VRB_cine_sr/data_sr/test'));
    
    pattern_g = '.mat';
    str_1 = regexp(data_name, pattern_g, 'split');
    save(strcat('/Users/yymacpro13/Desktop/VRB_cine_sr/data_sr/test/',str_1{1,1},'_0',num2str(frames),'.mat'),"HR","LR_2","LR_4"); 
    


end   
end
