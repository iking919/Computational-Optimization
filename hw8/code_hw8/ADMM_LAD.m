function [x,hist_obj, out] = ADMM_LAD(A,b,rho,tol,maxit,x0)

% alternating direction method of multipliers for
% min_x ||A*x-b||_1

[m,n] = size(A);

% initialization
x = x0;
z = A*x - b;

hist_obj = norm(z,1);

% initialize the multiplier
y = zeros(m,1);

pr = 0;
dr = 1;

iter = 1;

% Compute transpose to avoid per-loop computations
At = A';
AtA = At*A;

while max(pr,dr) > tol && iter < maxit
    iter = iter + 1;

    % update x
    x = AtA \ (At * (z + b - y/rho)); 

    % update z 
    z0 = z;

    % Compute Ax once
    Ax = A*x;

    % Initialize v and kappa
    v = Ax - b + y/rho;
    kappa = 1/rho;

    % Soft-thresholding operator for the L1 norm
    z = sign(v) .* max(abs(v) - kappa, 0);
    
    % update y
    y = y + rho * (Ax - z - b);

    % compute primal and dual residual
    pr = norm(Ax - z - b);    
    dr = norm(rho * A' * (z0 - z));

    % compute and save objective value ||A*x-b||_1
    obj = norm(Ax - b, 1);
    hist_obj = [hist_obj; obj];
    
end
out.iter = iter;
out.pr = pr;
out.dr = dr;
end