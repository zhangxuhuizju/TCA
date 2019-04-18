clc;
clear all
load('EEG_X.mat', 'X');
load('EEG_Y.mat', 'Y');
feature=[];label=[];
malabeled=[];
cvObj=[];
ACC=[];
for j=1:15
    temp_data = [X{j} Y{j}];
    rowrank = randperm(size(temp_data, 1)); % 随机打乱的数字，从1~行数打乱
    temp_data = temp_data(rowrank, :);

    feature=[feature;temp_data([1:1000],[1:310])];
    label = [label;temp_data([1:1000],[311:311])];
end
%[feature,featureps]=mapminmax(feature,0,1);
% [test_matrix,test_matrixps]=mapminmax(test_matrix,0,1);
for k = 1:15
     malabeled=[];
    for i = 1:15
        if(i ~= k)
            malabeled=[malabeled;true(1000,1)];
        else
            malabeled=[malabeled;false(1000,1)];
        end
    end
    malabeled=logical(malabeled);
    cvObj.training = malabeled;
    cvObj.test = ~cvObj.training;
    % TCA
    param = []; param.kerName = 'lin';param.bSstca = 0;
    param.mu = 1;param.m = 100;param.gamma = .1;param.lambda = 1;
    [Xproj,transMdl] = ftTrans_tca(feature,malabeled,label(malabeled),malabeled,param);
    acc = doPredict(Xproj,label,cvObj);
    ACC=[ACC;acc];
end
ACC
fid=fopen('result.txt','w');

fprintf(fid,'%11.5f\r\n',ACC);

fclose(fid);