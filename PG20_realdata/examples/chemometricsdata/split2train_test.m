function [train_X,test_X,train_Y,test_Y] = split2train_test(X,Y,proportion)
%% ���������������Ϊѵ���Ͳ�������
 
% ���������
% X,Y : ԭʼ����,Ĭ��ʹ������Ϊһ������
% proportion: ѵ����������
 
% ���������
% train:ѵ������
% test����������
 
[n,~]=size(X);
R = randperm(n);
r = R(1:proportion*n);
rc = setdiff(R,r);
train_X = X(r,:);
train_Y = Y(r,:);
test_X = X(rc,:);
test_Y = Y(rc,:);
end