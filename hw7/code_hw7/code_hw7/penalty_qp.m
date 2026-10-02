function [x, hist_obj, hist_res] = penalty_qp(Q,c,A,b,tol,mu0,mu1,x0)
% quadratic penalty method for the quadratic programming
% min_x 0.5*x'*Q*x - c'*x
% s.t.  x >= 0, A*x == b

mu = mu0;
x = x0;


% precompute AtA and Atb outside the loops
AtA = A'*A;
Atb = A'*b;

% compute the residual for the constraint A*x == b
r = A * x - b;

res = norm(r);
grad_err = 1;
hist_res = res;
hist_obj = 0.5*x'*Q*x - c'*x;

while (res > tol || grad_err > tol) && mu < mu1
    % use constant stepsize
    alpha = 1/norm(Q + mu*AtA);
    % compute the gradient
    grad = Q*x - c + mu*(AtA*x - Atb);

    % compute violation of optimality condition
    grad_err = norm(x - max(0, x - grad));
    while grad_err > tol
        % update x
        x = max(0, x - alpha*grad);

        % compute the gradient
        grad = Q*x - c + mu*(AtA*x - Atb);

        % compute violation of optimality condition
        grad_err = norm(x - max(0, x - grad));
    end
    % compute the residual
    r = A*x -  b;
    res = norm(r);
    obj = 0.5*x'*Q*x - c'*x;
    
    % save res and obj
    hist_res = [hist_res; res];
    hist_obj = [hist_obj; obj];
    
    % increase the penalty parameter
    mu = 5*mu;
end
end