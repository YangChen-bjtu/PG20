function out =data_ols(B,n,p,q,sigma)
SIGMA=zeros(p,p);
for i=1:p
    for j=1:p
        SIGMA(i,j) = 0.5^abs(i-j);
    end
end
MU = zeros(1,p);
X = mvnrnd(MU,SIGMA,n);
X = Normalization(X, 1);
out.X = X;
out.Xt = X';
out.cov  = SIGMA;
Y  = X*B+sigma*randn(n,q); 
out.Y = Y;
end
