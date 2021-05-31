% randomly half-split and runing 100 times for chemometrics dataset
clear; clc; close all;
X = load('ChemometricsDatasetX.mat');
X = cell2mat(struct2cell(X)); 
[n,p]=size(X);
Y = load('ChemometricsDatasetY.mat');
Y = cell2mat(struct2cell(Y)); 
Y = log(Y);                            % log-transformed
[~,q]=size(Y);
[X,Y] = normalization(X,Y,1);
K = 100;
s = zeros(K,1);                       
MPSE = zeros(K,1);                    
lam = zeros(K,1);                     
for k=1:K
    [train_X,test_X,train_Y,test_Y]  = split2train_test(X,Y,0.5);
    data.X  = train_X;
    data.Xt = data.X';
    data.Y  = train_Y;
    lam(k)  = CV(data.X,data.Y,p,q,5);    % select by 5-fold cv
    [m,~]   = size(test_Y);
    fun     = str2func('ols_20');
    func    = @(B)fun(B,data);
    pars.tol = 1e-4;
    out      = PG_20(p, q, lam(k), func, pars);
    s(k)     = length(out.T);
    MPSE(k)  = norm(test_Y-test_X*out.B, 'fro')^2/m;
 end
fprintf('\n Sample size:  n=%d, p=%d ,q=%d\n',n,p,q);
fprintf(' Nonzero row: %6.2f\n', mean(s));
fprintf(' std of nonzero row: %6.2f\n', std(s));
fprintf(' MPSE: %6.4f\n',mean(MPSE)); 
fprintf(' std of MPSE: %6.4f\n',std(MPSE)); 