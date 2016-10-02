function cost= cost_fun_cannon(X)


% Assuming cosntant U
N_grid = (length(X)-3)/3;
% Cost fun
t_f = X(1);
vx = X(2);
vy = X(3);

cost = vx^2+vy^2;