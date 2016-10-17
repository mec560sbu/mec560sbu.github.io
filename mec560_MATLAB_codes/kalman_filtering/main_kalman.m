clc
close all
clear all


A = [0 1 ; 0 0];
B = [0 ; 1];
dt = 0.001;

Ad = eye(2) + dt*[0 1 ; 0 0];
Bd = dt*B;

C = [1 0];
R = 0.1;
Q = diag([.1 .1]);
P_pl = eye(2);
X_hat_0 = [0;0];

X_0 = [1;0];

X(:,1)= X_0;
Y(1) = C*X(:,1);
t(1) = 0;
X_hat(:,1)= X_hat_0;

for i = 1:2000
    u = 1;
    
    t(i+1) = t(i)+dt;
    
    % True process 
    X(:,i+1)=Ad * X(:,i) + Bd*u;
    Y(i+1) = C*X(:,i+1);
    
    P_mi = Ad*P_pl*Ad' + Q;
    X_hat(:,i+1)=Ad * X_hat(:,i) + Bd*u;
    
    
    Y_hat(i+1) = C*X_hat(:,i+1);
    
    
    e_Y  = Y(i+1) - Y_hat(i+1);
    S = C*P_mi*C'+R;
    K = P_mi*C'*inv(S);
    P_pl = (eye(2) - K*C)*P_mi;
    X_hat(:,i+1)=X_hat(:,i+1) + K*e_Y;
end

figure;
subplot(2,1,1)
plot(t,X(1,:),t,X_hat(1,:),'--')
subplot(2,1,2)
plot(t,X(2,:),t,X_hat(2,:),'--')
