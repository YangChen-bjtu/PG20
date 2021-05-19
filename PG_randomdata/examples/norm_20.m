 function [b]=norm_20(B)
       p=size(B,1);
        z=zeros(p,1);
       for i=1:p
           bi=norm(B(i,:),2);
            z(i)=bi;
       end
 b=z;
 end