function out = PG_20(p, q, lam, func, pars)
% A solver for sparsity constraints models:
%
%           min f(B)+lam*||B||_2,0,   
%           f is continuous and differentiable
%           ||B||_2,0 is the number of the nonzero rows of matrix B

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
if isfield(pars,'maxit'); maxit  = pars.maxit;  else; maxit  = 1e5;      end
if isfield(pars,'tol');   tol    = pars.tol;    else; tol = 1e-4;        end  

B0     = 0.01*ones(p,q);
B_k    = B0;
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
%%%%%%%%%%%%%%%% Constant stepsize
    L_k = 1000;
    obj(iter) = f_k+lam*length(find(norm_2(B_k)>=1e-8));
    B = hard_threshold(p,q,B_k-g_k/L_k,lam/L_k);
    f = func(B);
%%%%%%%%%%%%%%%% Backtracking procedure
%     eta    = 2;
%     L0     = 1;
%     L_k    = L0;  
%     for iter_L  = 1:10       
%          B = hard_threshold(p,q,B_k-g_k/L_k,lam/L_k);
%         f = func(B);
%         if f <= f_k+trace(g_k'*(B-B_k))+0.5*L_k*norm(B-B_k,'fro')^2
%             break;
%         end
%         L_k  = eta*L_k;
%     end
%     
    % Stop criteria
    residual =  norm(B-B_k,'fro')/norm(B_k,'fro');

    if iteron  && mod(iter,1)==0
       fprintf('%4d    %5.2e     %5.2e    %5.2fsec\n',iter,residual,obj(iter),toc(t0)); 
    end
 
	if residual<tol || abs(f-f_k)<1e-10*(1+abs(f))
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

% solve prox_{\lam*\ell_{2,0}}(x)
function [H]=hard_threshold(p,q,B,lambda)
B0=zeros(p,q);
b =  norm_2(B);
for i=1:p
    if b(i)>sqrt(2*lambda)
        H(i,:)=B(i,:);
    else
        H(i,:)=B0(i,:);
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