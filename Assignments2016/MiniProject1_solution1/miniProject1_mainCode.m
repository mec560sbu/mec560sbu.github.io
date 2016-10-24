clc
close all
clear all

%% Generate Path
grid_x = 0:.1:10;
grid_y = 0:.1:10;

x_start = [1];
y_start = [1];

x_goal = 9;
y_goal = 9;

obstacles = [
    2 2  4 8;
    4 0.25  8 1;
    6 1  8 6;
    6 7 9.5 8;
    2 7 5 9;
    7 8 8 9;
    4 2 5.5 4
    ];
buffer = 0.1;

path = genPath(grid_x,grid_y, x_start, y_start, x_goal, y_goal, obstacles, buffer);

%% Smooth Path
newPath = smoothPath(path);

%% Determine Time between Points
[newTime, x_des, y_des] = pointToTrajectory(newPath);

%% Observer & Controller
% Define system matrices
% q1 = x, q2 = y, q3 = dx, q4 = dy
A = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]; 
B = [0 0; 0 0; 1 0; 0 1];
C = [1 0 0 0; 0 1 0 0];

% Check if system is controllable & observable; rank = 4
display(['The rank of the observability matrix is ' num2str(rank(obsv(A,C)))])
display(['The rank of the controllability matrix is ' num2str(rank(ctrb(A,B)))]) 

% Optimal gain, K
Q = eye(4);
R = .01*eye(2);

optK = LQR_k(A,B,Q,R);
[v_cont, d_cont] = eig(A-B*optK);

% Observer gain,  L (eig(L) must be larger than eig(K))
p_obs = [-18; -19; -20; -21]-10;

save data.mat
load data.mat

% State estimation
[e_est, dt] = fullStateObs(A,B,C, optK, p_obs, newTime, x_des, y_des)

% Controller
x0 = [1.1,1,0,0];
control = @(x)[-optK*x];
sysDyn = @(t,e_est)[A*e_est+B*control(e_est)];

sysControl = control(e_est);
[t,x] = ode45(sysDyn, newTime, x0);

figure;
plot(newTime,sysControl(1,:), newTime, sysControl(2,:));
title(['\lambda_A are ' num2str(d_cont(1,1)) ', ' num2str(d_cont(2,2)) ', ' num2str(d_cont(3,3)) ' and ' num2str(d_cont(4,4))])
axis([0 10 -2 2])
ylabel('Control')
xlabel('Time')
legend('u_x', 'u_y', 'Location', 'southeast') 


