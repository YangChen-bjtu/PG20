function out = OLS_PG_Inf1(p, q, lam, func, pars)
% A solver for sparsity constraints models:
%
%           min f(B)+lam*||B||_\infty,1, 
%           f is continuous and differentiable
%           ||B||_\infty,1 is the sum of all maximum absolute elements of rows of matrix B

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

if nargin<3; error('Imputs are not enough!\n'); end
if nargin<4; pars=[]; end
if isfield(pars,'iteron');iteron = pars.iteron; else; iteron = 1;        end
if isfield(pars,'maxit'); maxit  = pars.maxit;  else; maxit  = 1e4;      end
if isfield(pars,'tol');   tol    = pars.tol;    else; tol = 1e-6*sqrt(p);end  

BO     =zeros(p,q);
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
    obj(iter) = f_k+lam*sum(norm_Inf(B_k));
    L0     = 1;
    L_k    = L0;  
    for iter_L  = 1:20       
         B = proximal_Inf1(p,q,B_k-g_k/L_k,lam/L_k);
         f = func(B);
        if f <= f_k+trace(g_k'*(B-B_k))+0.5*L_k*norm(B-B_k,'fro')^2
            break;
        end
        L_k  = eta*L_k;
    end
    
    % Stop criteria
    residual =  L_k*norm(B-B_k,'fro');

    if iteron  
       fprintf('%4d    %5.2e     %5.2e    %5.2fsec\n',iter,residual,obj(iter),toc(t0)); 
    end
 
	if residual<tol 
       break; 
    end  
    B_k=B;
end
T      =find(norm_Inf(B)>=1e-8);
F      =find(norm_Inf(B)<1e-8);

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
%proximal operator for g=\lam*\ell_{\infty,1}
function S = proximal_Inf1(p,q,B,a)
B0   = zeros(p,q);
b    = norm_1(B);
for i=1:p
    if b(i)>a
        P = diag(sign(B(i,:)));
        bopt = project(P*(B(i,:))'/a);
        S(i,:)=B(i,:)-a*(P*bopt)';
    else
        S(i,:)=B0(i,:);
    end
end
end

function [b]=norm_1(B)
       p=size(B,1);
        z=zeros(p,1);
       for i=1:p
           bi=sum(abs(B(i,:)));
            z(i)=bi;
       end
 b=z;
 end
 function bopt = project(b)
 p = length(b);
normal=ones(p,1);
error=1e3;
threshold=1e-12;
 b0=b; 
while error>threshold
  b1=b0-(1/p)*(sum(b0)-1)*normal;
  b1=max(b1,0);
  error=norm(b1-b0)^2;
  b0=b1;
end
bopt = b0;
 end
 function [b]=norm_Inf(B)
       p=size(B,1);
        z=zeros(p,1);
       for i=1:p
           bi=max(abs(B(i,:)));
            z(i)=bi;
       end
 b=z;
 end