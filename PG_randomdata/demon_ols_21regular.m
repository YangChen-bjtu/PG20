% demon ols_21_regular problems with randomly generated data
n       = 100;
p       = 200; 
s       = ceil(0.05*p); 
q       = 5;
switch 1
    case 1
        sigma   = 1;
        lam     = 100; % var=1
    case 2
        sigma   = 2;
        lam     = 155; % var=4
    case 3
        sigma   = 3;
        lam     = 190; % var=9
     case 4
        sigma   = 4;
        lam     = 250; % var=16
end
K        = 100;
E        = zeros(K,1);
REE     = E; % relative estimation error
ME      = E; % model error
t       = E;   % CPU time
TP      = E;   % ture positive
TN      = E;   % ture negative
FP      = E;   % false positive
FN      = E;   % false negative

% generate ground truth Bopt
I         = randperm(p); 
TB        = I(1:s);
B         = zeros(p,q);  
B(TB,:)     = randn(s,q);
data.Bopt = B;
data.T = TB;
data.F = I(s+1:p);
NBopt  = norm(B,'fro')^2;

for k=1:K 
    data_k  = data_ols(B,n,p,q,sigma);    
    fun     = str2func('ols_20');
    func    = @(B)fun(B,data_k);   
    pars.tol = 1e-6*sqrt(p);
    out_k    = OLS_PG_21(p,q,lam,func,pars); % ols_21_penalty
REE(k)    = norm(out_k.B-data.Bopt,'fro')^2/NBopt;
    ME(k)     = norm((out_k.B-data.Bopt)'*data_k.cov*(out_k.B-data.Bopt),'fro');
    TP(k)     = length(intersect(out_k.T,data.T));
    FN(k)     = length(intersect(out_k.F,data.T));
    FP(k)     = length(intersect(out_k.T,data.F));
    TN(k)     = length(intersect(out_k.F,data.F));
    t(k)=out_k.time;
end
    TPR      = TP./(TP+FN);
    FPR      = FP./(FP+TN);
    TSS      = TPR-FPR;
fprintf('\n Sample size:       n=%d, p=%d ,s=%d\n',n,p,s);
 if isfield(data,'Bopt')
fprintf(' Average CPU time:    %6.4fsec\n',  mean(t));
fprintf(' Average TPR : %6.4f\n',mean(TPR)); 
fprintf(' Average FPR : %6.4f\n',mean(FPR)); 
fprintf(' Average FPR : %6.4f\n',mean(TSS)); 
fprintf(' mean(REE): %6.3f\n',mean(REE));
fprintf(' std(REE): %6.3f\n',std(REE));
fprintf(' mean(ME): %6.3f\n',mean(ME));
fprintf(' std(ME): %6.3f\n',std(ME));
 end