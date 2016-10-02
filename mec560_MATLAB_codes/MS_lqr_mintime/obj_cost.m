function cost = obj_cost(X,N_grid,type)

N_grid = (length(X)-1)/3;

% Cost fun
t_f = X(1);
X1 = X(2:N_grid+1);
X2 = X(2+N_grid:2*N_grid+1);
u = X(2+2*N_grid:3*N_grid+1);

t = (0:0.01:1)*t_f;
dt = 0.01*t_f;
t_grid = (0:1/(N_grid-1):1)*t_f;


switch type
    case 1
        u_t = interp1(t_grid,u,t,'previous');
    case 2
        u_t = interp1(t_grid,u,t);
    case 3
        u_t = interp1(t_grid,u,t,'spline');
end
cost = t_f;