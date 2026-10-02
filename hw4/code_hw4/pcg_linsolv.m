function [y] = pcg_linsolv(C, r)

    % Initialize C^T, solution vectors z and y, and matrix size n
    CT = transpose(C);
    n = size(CT, 1);
    z = zeros(n, 1);
    y = zeros(n, 1);

    % Forward substitution
    for i = 1:n

        % Add known values up until diagonal
        summation = 0;
        for j = 1:i-1
            summation = summation + CT(i, j) * z(j);
        end

        % Get current diagonal value
        z(i) = (r(i) - summation) / CT(i, i);
    end

    % Backward substitution
    for i = n:-1:1

        % Add known values up until diagonal
        summation = 0;
        for j = i + 1:n
            summation = summation + C(i, j) * y(j);
        end

        % Get current diagonal value
        y(i) = (z(i) - summation) / C(i, i);
    end
end