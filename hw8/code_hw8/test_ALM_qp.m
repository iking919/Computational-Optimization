clear; close all;
m = 100;
n = 500;
A = [eye(m), randn(m,n-m)];
b = rand(m,1) + 0.1;
Q = randn(n);
Q = Q'*Q;
Q = Q/max(abs(Q(:)));
c = randn(n,1);

x0 = max(0,randn(n,1));
tol = 1e-3;
beta = 1;

%% call MATLAB built-in function quadprog to get optimal solution
[xopt, fopt] = quadprog(Q, -c, [],[], A,b,zeros(n,1),[]);

%% call student's solver
t0 = tic;
[x_s, hist_obj_s, hist_res_s] = ALM_qp(Q,c,A,b,tol,beta,x0);
t1 = toc(t0);

fprintf('Student solver: Total running time of student code is %5.4f\n', t1);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(abs(hist_obj_s - fopt), 'b-','linewidth',2);
hold on
semilogy(hist_res_s,'r-','linewidth',2);
legend('Objective error','Feasibility violation')

set(gca,'fontsize',12);

xlabel('outer iteration number','fontsize',12);
ylabel('error','fontsize',12);

title('Student ALM','fontsize',12);

print(fig, '-dpdf','student_ALM_results')

%% call instructor's Qudratic penalty solver
mu0 = 0.1; mu1 = 1e4;
t0 = tic;
[x_p, hist_obj_p, hist_res_p] = penalty_qp_p(Q,c,A,b,tol,mu0,mu1,x0);
t1 = toc(t0);

fprintf('Instructor penalty solver: Total running time of instructor code is %5.4f\n', t1);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(abs(hist_obj_p - fopt), 'b-','linewidth',2);
hold on
semilogy(hist_res_p,'r-','linewidth',2);
legend('Objective error','Feasibility violation')

set(gca,'fontsize',12);

xlabel('outer iteration number','fontsize',12);
ylabel('error','fontsize',12);

title('Instructor penalty','fontsize',12);

print(fig, '-dpdf','instructor_penalty_results')

%% call instructor's Qudratic penalty solver
t0 = tic;
[x_p, hist_obj_p, hist_res_p] = ALM_qp_p(Q,c,A,b,tol,beta,x0);
t1 = toc(t0);

fprintf('Instructor ALM solver: Total running time of instructor code is %5.4f\n', t1);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(abs(hist_obj_p - fopt), 'b-','linewidth',2);
hold on
semilogy(hist_res_p,'r-','linewidth',2);
legend('Objective error','Feasibility violation')

set(gca,'fontsize',12);

xlabel('outer iteration number','fontsize',12);
ylabel('error','fontsize',12);

title('Instructor ALM','fontsize',12);

print(fig, '-dpdf','instructor_ALM_results')


