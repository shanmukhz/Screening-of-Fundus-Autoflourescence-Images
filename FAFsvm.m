close all
clear all
clc

% ----- parameters ----------
r = 0.9;    %Train-test-ratio
iter=500;
C =  zeros(2,2);
Cum_conf=zeros(2,2,iter);

Abs_count = 0;
Nrm_count = 0;

train_accuracy=zeros(iter,1);
test_accuracy=zeros(iter,1);

% -------- reading data from xlsx-----------
[X0,X1]=data;

Y0=ones(size(X0,1),1);
Y1=zeros(size(X1,1),1);
Y=[Y1;Y0];
X=[X1;X0];

Data = [X,Y];% Normal 1 and diseased 0
Data_N = [X0,Y0];
Data_A = [X1,Y1];

% -----------------------------------

%-- Data splitting -------------
    
for i = 1:iter
    i    
    permdata = randperm(size(Data,1));
    Data1 = Data(permdata,:);
    
    div = round(size(Data1,1)*r);
    
    data_train = Data1(1:div,1:18);
    labels_train = Data1(1:div,19);
    
    %data_test = Data1(1:div,1:3)';
    %labels_test = Data1(1:div,4)'+1;
    
    data_test = Data1(div+1:end,1:18);
    labels_test = Data1(div+1:end,19);
    
%     Data_Train_Iter{temp,i} = data_train;
%     Data_Test_Iter{temp,i} = data_test;
%     
%     Labels_Train_Iter{temp,i} = labels_train;
%     Labels_Test_Iter{temp,i} = labels_test;
    
%-- tain and predictions ----
    svmModel=fitcsvm(data_train,labels_train,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
    A{i}=svmModel;
    [pre_train,~]=predict(svmModel,data_train);
    [pre_test,score]=predict(svmModel,data_test);
    
    train_accuracy(i)=sum(pre_train==labels_train)/size(labels_train,1);
    test_accuracy(i)=sum(pre_test==labels_test)/size(labels_test,1);
    
%---- confusion matrix ---
    
    C(1,1) = size(find(pre_test == 0 & labels_test == 0 ),1);%/ (size(labels_test,1)-sum(labels_test));
    C(1,2) = size(find(pre_test == 0 & labels_test == 1 ),1);%/(size(labels_test,1)-sum(labels_test));
    C(2,1) = size(find(pre_test == 1 & labels_test == 0 ),1);%/sum(labels_test);
    C(2,2) = size(find(pre_test == 1 & labels_test == 1 ),1);%/sum(labels_test);
    
    Cum_conf(:,:,i) = C;
    Abs_count = Abs_count + size(labels_test,1)-sum(labels_test);
    Nrm_count = Nrm_count + sum(labels_test);


end
