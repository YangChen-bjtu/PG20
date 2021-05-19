function [out1,out2] = ols_20(B,data)
    Tb  = find(norm_20(B));
    if ~isempty(Tb)   
    XBY  = data.X(:,Tb)*B(Tb,:)-data.Y;
    else  
    XBY  =  -data.Y;
    end

    out1 = (norm(XBY,'fro'))^2;                %objective function 
    
    if  nargout>1 
    out2 = 2*data.Xt*XBY;                    %gradien          
    end
