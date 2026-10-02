function [x, hist_obj] = quadMin_AltMin(A,b,x0,maxit,lb,ub)

% alternating minimization method for solving
% min_x 0.5*x'*A*x - b'*x
% s.t. lb <= x <= ub

x = x0;

% compute the gradient and maintain it
r = A*x - b;

hist_obj = .5*(x'* (r - b));

n = length(b);

for iter = 1:maxit
    
    % update all coordinates cyclicly
    for i = 1:n 
        
        % store old value

        x_old = x(i);

        % update x(i)
        
        x(i) = max(lb(i), min(ub(i), x(i) - r(i)/A(i,i)));
        
        % update r vector in an efficient way

        r = r + A(:,i) * (x(i) - x_old);
    end
    
    % save objective value after each cycle
    hist_obj = [hist_obj; .5*(x'* (r - b))];
    
end

end

