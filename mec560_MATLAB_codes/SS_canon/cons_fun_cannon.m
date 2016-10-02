function [C,Ceq] = cons_fun_cannon(X)

% Assuming cosntant U
N_grid = (length(X)-3)/3;
% Cost fun
t_f = X(1);
vx = X(2);
vy = X(3);


d_x = t_f*vx;
d_y = vy*t_f -1/2*9.8*t_f^2;

C = [-t_f+0.001;
    -vx;
    -vy];
Ceq = [d_y-10;
            d_x - 10;
];

