function dx = sys_dyn_doubleIntegrator(t,states,X,type)
N_grid = (length(X)-1)/3;

t_f = X(1);
X1 = X(2:N_grid+1);
X2 = X(2+N_grid:2*N_grid+1);
u = X(2+2*N_grid:3*N_grid+1);
t_grid = (0:1/(N_grid-1):1)*t_f;
A = [0 1; 0 0];
B = [0; 1];


switch type
    case 1
        u_t = interp1(t_grid,u,t,'previous');
    case 2
        u_t = interp1(t_grid,u,t);
    case 3
        u_t = interp1(t_grid,u,t,'spline');
end
dx = A * states+B*u_t;