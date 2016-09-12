function [t,x] = solve_x(a,x0,dt,t_f)

x(1) = x0;
t(1) = 0
N = round(t_f/dt);
N
for i = 2:N+1
    t(i) = t(i-1) + dt;
    x(i) = (1+a*dt)*x(i-1);
end