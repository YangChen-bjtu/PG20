function out = OLS_PG_21(p, q, lam, func, pars)
% A solver for sparsity constraints models:
%
%           min f(B)+lam*||B||_2,1,   
%           f is continuous and differentiable
%           ||B||_2,1 is the sum of all Euclidean norms of rows of matrix B

% Inputs:
%     q       : # of response variables
%     p       : # of predictor variables
%     func    : function handle defines the function f(x) and its gradient             
%     pars:     Parameters are all OPTIONAL
%               pars.iteron --  Results will  be shown for each iteration if pars.iteron=1 (default)
%                               Results won't be shown for each iteration if pars.iteron=0 
%               pars.maxit  --  Maximum nonumber of iteration.  pars.maxit=10000 (default) 
%               pars.tol    --  Tolerance of stopping criteria. pars.maxit=1e-4 (default) 
%
% Outputs:
%     out.x:             The sparse solution x 
%     out.obj:           f(x)
%     out.time           CPU time
%     out.iter:          Number of iterations

%%%%%%%    Warning: Accuracy may not be guaranteed!!!!!              %%%%%%
warning off;

if nargin<4; error('Imputs are not enough!\n'); end
if nargin<5; pars=[]; end
if isfield(pars,'iteron');iteron = pars.iteron; else; iteron = 1;        end
if isfield(pars,'maxit'); maxit  = pars.maxit;  else; maxit  = 1e4;      end
if isfield(pars,'tol');   tol    = pars.tol;    else; tol = 1e-6*sqrt(p);end  

BO     = zeros(p,q);
B_k    = BO;
eta    = 2;
obj    = zeros(maxit,1);

t0     = tic;
% main body
if iteron 
fprintf(' Start to run the sover...\n'); 
fprintf('\n Iter          Error       obj            Time \n'); 
fprintf('------------------------------------------------\n');
end
for iter=1:maxit      
    [f_k,g_k]  = func(B_k);
    obj(iter)  = f_k+lam*sum(norm_2(B_k));
    L0  = 1;
    L_k = L0;
    for iter_L  = 1:10        
        B = soft_threshold(p,q,B_k-g_k/L_k,lam/L_k);
        f = func(B);        
        if f <= f_k+trace(g_k'*(B-B_k))+0.5*L_k*norm(B-B_k,'fro')^2
            break;
        end
        L_k  = eta*L_k;
    end
   
% Stop criteria 
     residual =  L_k*norm(B-B_k,'fro'); 
     
      if iteron 
       fprintf('%4d    %5.2e    %5.2e    %5.2fsec\n',iter,residual,obj(iter),toc(t0)); 
      end
    if residual < tol 
        break;
    end
    B_k=B;
end
      
T      =find(norm_2(B)>=1e-8);
F      =find(norm_2(B)<1e-8);

if iteron
fprintf('------------------------------------------------------------\n');
end

out.B    = B;
out.T    = T;
out.F    = F;
out.iter = iter;
out.obj  = obj(1:iter);
out.time = toc(t0);
out.error= residual;
end
% solve the proximal operator for g=\lam*\ell_{21}
function [S]=soft_threshold(p,q,B,lambda)
B0   = zeros(p,q);
b    = norm_2(B);
for i=1:p
    if b(i)>lambda
        S(i,:)=(b(i)-lambda)*B(i,:)/b(i);
    else
S(i,:)=B0(i,:);
    end
end
end
% solve the \ell_2 norms of rows for a matrix
 function [b]=norm_2(B)
       p=size(B,1);
        z=zeros(p,1);
       for i=1:p
           bi=norm(B(i,:),2);
            z(i)=bi;
       end
 b=z;
 end
