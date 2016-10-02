clc
close all
clear all 

t_f = 10;
dt = 0.001;
P_f= 100*eye(2);
Q = .0001*eye(2);
R = 1;
A = [0 1;0 0];
B = [0 ;1];
P0 =P_f;

% Vectors

N = t_f/dt+1;

% add t_dis array 
t_dis = 0:dt:t_f;
t_res = t_f:-dt:0;

% GET SOLUTION FOR P, 
% USING ODE45 method

[t, P_all] = ode45(@(t,X)P_sys(t,X,A,B,Q,R),t_res,P0); 
P_all_res_ode =   reverse_indices(P_all); % P_all is labeled with time to go. 

X0=[10;0];
X_ode(:,1) = X0;

for i = 2:length(t_dis)
    P_ode = reshape(P_all_res_ode(i-1,:),size(A));
    U_ode(:,i-1) = -inv(R)*B'*P_ode*X_ode(:,i-1);
    X_ode(:,i) = X_ode(:,i-1) + dt* (A*X_ode(:,i-1) + B*U_ode(:,i-1) );  

    
end

figure;
plot(t_dis,X_ode)
xlabel('time')
ylabel('States')
legend('position','velocity')
