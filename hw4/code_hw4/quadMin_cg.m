function [x, hist_res] = quadMin_cg(A,b,x0,tol)

% conjugate gradient method for solving
% min_x 0.5*x'*A*x - b'*x

% get the size of the problem
n = length(b);

x = x0;

% compute vector r, i.e., gradient of the objective
r = A * x - b;

% set the first p vector as negative gradient
p = -r;

% evaluate the norm of gradient
res = norm(r);

% save the value of res
hist_res = res;

while res > tol
    
    % compute A * p once

    Apk = A * p;

    % compute <p, Apk> once
    
    pDotApk = dot(p, Apk);

    % compute alpha
    
    alpha = - dot(p, r) / pDotApk;
    
    % update x 
    
    x = x + alpha * p;
    
    % update r
    
    r = A * x - b;
    
    % compute beta
    
    beta = dot(r, Apk) / pDotApk;
    
    % obtain the new p vector
    
    p = -r + beta * p;
    
    % evaluate the norm of residual vector r
    res = norm(r);
    
    % save the value of res
    hist_res = [hist_res; res];
end

end