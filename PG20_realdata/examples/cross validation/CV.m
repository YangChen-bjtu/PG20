function optlam = CV(X,Y,p,q,K)
[l,~] = size(X);
lam   = 0.5:0.1:1.0; 
J     = length(lam);
MPSE  = zeros(J,1);
S     = zeros(J,1); 
for j=1:J
    s   = zeros(K,1);
    pse = zeros(K,1);
    indices = crossvalind('Kfold',X(1:l,p),K);
    for k = 1:K  
        test  = (indices == k); 
        train = ~test;  
        train_X = X(train,:); 
        data.X  = train_X;
        data.Xt = data.X';
        train_Y = Y(train,:); 
        data.Y  = train_Y;
        test_X  = X(test,:); 
        test_Y  = Y(test,:);
        [m,~]   = size(test_Y);
        fun     = str2func('ols_20');
        func    = @(B)fun(B,data);  
        pars.tol = 1e-4;
        out      = PG_20(p, q, lam(j), func, pars);
        s(k)     = length(out.T);
        pse(k)   = norm(test_Y-test_X*out.B, 'fro')^2/m; 
    end
    MPSE(j) = mean(pse);
    S(j) = mean(s);
end
position = (MPSE==min(MPSE));
optlam = lam(position);
end