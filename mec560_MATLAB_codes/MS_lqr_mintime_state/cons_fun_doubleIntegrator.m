function [C,Ceq] = cons_fun_doubleIntegrator(X,type)

% Assuming cosntant U
N_grid = (length(X)-1)/3;
% Cost fun
t_f = X(1);
X1 = X(2:N_grid+1);
X2 = X(2+N_grid:2*N_grid+1);
u = X(2+2*N_grid:3*N_grid+1);


C = [X2-1.5];
Ceq = [X2(1) - 0;
    X2(end) + 0;
    X1(1);
    X1(end)-10;
    ];


time_points = (0:1/(N_grid-1):1)*t_f;
for i = 1:(N_grid-1)
    X_0(1,:) = X1(i);
    X_0(2,:) = X2(i);
    t_start = time_points(i);
    t_end = time_points(i+1);
    
    tt = t_start:(t_end-t_start)/10:t_end;
    [t,sol_int] = ode45(@(t,y)sys_dyn_doubleIntegrator(t,y,X,type),tt,X_0);
    X_end = sol_int(end,:);
    
    Ceq = [Ceq;
        X_end(1) - X1(i+1);
        X_end(2) - X2(i+1)];
end