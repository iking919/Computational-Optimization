function [x, hist_obj, hist_res] = ALM_qp(Q,c,A,b,tol,beta,x0)
% Augmented Lagrangian method for the quadratic programming
% min_x 0.5*x'*Q*x - c'*x
% s.t.  x >= 0, A*x == b

x = x0;

% compute the residual for the constraint A*x == b
r = A*x - b;

res = norm(r);

v = zeros(size(b)); % initialize the multiplier
grad_err = 1;
hist_res = res;
hist_obj = 0.5 * x' * Q * x - c' * x ;

while res > tol || grad_err > tol
    % use constant stepsize (based on Lipschitz constant)
    alpha = 1/ norm(Q + beta * (A' * A));

    % compute the gradient
    grad = Q * x - c + A' * v + beta * A' * (A * x - b);

    % compute violation of optimality condition
    grad_err = inf;
    while grad_err > tol
        % update x
        x = max(0, x - alpha * grad);
        % compute the gradient        
        grad = Q * x - c + A' * v + beta* A' * (A * x - b);

        % compute violation of optimality condition        
        grad_err = norm(x - max(0, x - grad));
    end
    % compute the residual
    r = A * x - b;
    res = norm(r);
    obj = 0.5 * x' * Q * x - c' * x;
    
    % save res and obj
    hist_res = [hist_res; res];
    hist_obj = [hist_obj; obj];
    
    % update multiplier

    v = v + beta * r;

end
end