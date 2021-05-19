function [out1,out2] = logistic_20(B,data)
[n,q] = size(data.Y);
b     = norm_20(B);
Tb    = find(b);

if ~isempty(Tb)   
    XB  = data.X(:,Tb)*B(Tb,:);
    eXB = exp(XB);
else  
    XB    = zeros(n,q);
    eXB   = ones(n,q);
end

seXB = sum(eXB,2);
out1  = sum(log(1+sum(eXB,2)))-sum(sum(XB.*data.Y));    %objective function    

if  nargout>1 
    eX=1./(1+seXB);
    out2 = data.Xt*(-data.Y+diag(eX)*eXB);  % gradient
end
       
end


