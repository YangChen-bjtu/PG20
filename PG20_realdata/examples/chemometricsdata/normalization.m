function [X,Y] = normalization(X,Y,type)
[n,~] = size(X);
switch type
    case 1
        for i=1:n
            X(i,:)=(X(i,:)-mean(X(i,:)))/std(X(i,:)); %标准化
            Y(i,:)=(Y(i,:)-mean(Y(i,:)))/std(Y(i,:)); %标准化
        end
    case 2 
       for i=1:n
           X(i,:)=X(i,:)/norm(X(i,:)); %行标准化
           Y(i,:)=Y(i,:)/norm(Y(i,:)); %行标准化
       end
    case 3
        [X,~] = mapminmax(X,0,1); %行归一化
        [Y,~] = mapminmax(Y,0,1); %行归一化
end
end
