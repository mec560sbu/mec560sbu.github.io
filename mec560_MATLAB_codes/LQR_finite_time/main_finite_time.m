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
t_dis = linspace( 0, t_f, N );
t_res = linspace(t_f, 0, N );

t_dis = 0:dt:t_f;
t_res = t_f:-dt:0;





% GET SOLUTION FOR P, 

% USING ODE45 method

[t, P_all] = ode45(@(t,X)P_sys(t,X,A,B,Q,R),t_res,P0); 
P_all_res_ode =   reverse_indices(P_all); % P_all is labeled with time to go. 

P_all_res(:,length(t_res))= P_f(:);
for i = length(t_res):-1:2
    P =reshape(P_all_res(:,i),size(A));
    dPdt = -(A.'*P + P*A - P*B*B.'*P + Q); 
    
    P = P - dt*(dPdt);
    P_all_res(:,i-1)= P(:);
end
P_all_res = P_all_res';
for i = 1:size(P_all_res)
    norm_P(i) = norm(P_all_res(i,:)-P_all_res_ode(i,:))/norm(P_all_res_ode(i,:))*100;    
end

figure;
plot(norm_P);
axis([1 1000 0 .0001]);
ylabel('Percentage error between ODE45 and Numerical integration');
xlabel('Time');

X0=[1;0];
X0=[1;0];
X_ode(:,1) = X0;
X_eul(:,1) = X0;

for i = 2:length(t_dis)
    P_ode = reshape(P_all_res_ode(i-1,:),size(A));
    U_ode(:,i-1) = -inv(R)*B'*P_ode*X_ode(:,i-1);
    X_ode(:,i) = X_ode(:,i-1) + dt* (A*X_ode(:,i-1) + B*U_ode(:,i-1) );  

    P_eul = reshape(P_all_res(i-1,:),size(A));
    U_eul(:,i-1) = -inv(R)*B'*P_eul*X_eul(:,i-1);
    X_eul(:,i) = X_eul(:,i-1) + dt* (A*X_eul(:,i-1) + B*U_eul(:,i-1) );  
end

figure;
plot(t_dis,X_eul,t_dis,X_ode)
