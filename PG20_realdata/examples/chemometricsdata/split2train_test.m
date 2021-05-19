function [train_X,test_X,train_Y,test_Y] = split2train_test(X,Y,proportion)
%% 把输入数据随机分为训练和测试样本
 
% 输入参数：
% X,Y : 原始矩阵,默认使用行作为一个样本
% proportion: 训练样本比重
 
% 输出参数：
% train:训练数据
% test：测试数据
 
[n,~]=size(X);
R = randperm(n);
r = R(1:proportion*n);
rc = setdiff(R,r);
train_X = X(r,:);
train_Y = Y(r,:);
test_X = X(rc,:);
test_Y = Y(rc,:);
end