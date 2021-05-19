function [out1,out2] = poisson_20(B,data)
[n,q] = size(data.Y);
b   = norm_20(B);
Tb  = find(b);

if ~isempty(Tb)   
    XB  = data.X(:,Tb)*B(Tb,:);
    eXB = exp(XB);
else
    XB    = zeros(n,q);
    eXB   = ones(n,q);
end   

out1 = sum(sum(eXB-XB.*data.Y));       % objective function

if  nargout>1 
    out2 = data.Xt*(-data.Y+eXB);      % gradien          
end

end

    
    
