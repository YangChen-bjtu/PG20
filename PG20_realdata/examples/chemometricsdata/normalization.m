function [X,Y] = normalization(X,Y,type)
[n,~] = size(X);
switch type
    case 1
        for i=1:n
            X(i,:)=(X(i,:)-mean(X(i,:)))/std(X(i,:)); %��׼��
            Y(i,:)=(Y(i,:)-mean(Y(i,:)))/std(Y(i,:)); %��׼��
        end
    case 2 
       for i=1:n
           X(i,:)=X(i,:)/norm(X(i,:)); %�б�׼��
           Y(i,:)=Y(i,:)/norm(Y(i,:)); %�б�׼��
       end
    case 3
        [X,~] = mapminmax(X,0,1); %�й�һ��
        [Y,~] = mapminmax(Y,0,1); %�й�һ��
end
end
