function [x, hist_res] = quadMin_pcg(A,C,b,x0,tol)

% conjugate gradient method for solving
% min_x 0.5*x'*A*x - b'*x

% get the size of the problem
n = length(b);

x = x0;

% compute vector r, i.e., gradient of the objective
r = A * x - b;

% compute first y
y = pcg_linsolv(C, r);

% set the first p vector 
p = -y;

% evaluate the norm of gradient
res = norm(r);

% save the value of res
hist_res = res;

while res > tol

    % compute rk dot yk once

    rDoty = dot(r, y);

    % compute A * p once

    Ap = A*p;

    % compute alpha
    
    alpha = rDoty / dot(p, Ap);
    
    % update x 
    
    x = x + alpha*p;
    
    % update r
    
    r = r + alpha * Ap;

    % compute updated y once

    yNext = pcg_linsolv(C, r);
    
    % compute beta
    
    beta = dot(r, yNext) / rDoty;
    
    % obtain the new p vector
    
    p = -yNext + beta * p;

    % update y

    y = yNext;
    
    % evaluate the norm of residual vector r
    res = norm(r);
    
    % save the value of res
    hist_res = [hist_res; res];
end

end
