clc
close all
clear all


A = [0 1;0 0];
B = [0; 1];
C = [1 0];

R = diag([1]);
Q = diag([1e-2 1e-2]);
M_a = rank(ctrb(A,B));
P = care(A,B,Q,R);
K_u  = inv(R)*B'*P;

t = 0:0.001:20; 
dt = t(2) - t(1);
X(:,1) = [1;0];
y(:,1) = C*X;
R_k = diag([1 ]);
Q_k = diag([1 1]);
P_k = care(A',C',Q_k,R_k);
K_k = P_k*C'*inv(R_k);
L = K_k;

X_hat(:,1) = [0;0];
y_hat(:,1) = C*X_hat;
for i = 2:length(t)
    u = -K_u*X_hat(:,i-1);
    
    X(:,i) = X(:,i-1)  +dt * (A*X(:,i-1) + B*u);
    y(:,i) = C*X(:,i) + sqrt(R_k)*randn(size(C,1),1);

    X_hat(:,i) = X_hat(:,i-1)  +dt * (A*X_hat(:,i-1) + B*u +L *(y(:,i-1)-y_hat(:,i-1)));
    y_hat(:,i) = C*X_hat(:,i) ;
end


figure;
subplot(2,1,1)
plot(t,X(1,:),t,X_hat(1,:))
legend('Actual','Estimate')
xlabel('time')
ylabel('Position')
subplot(2,1,2)
plot(t,X(2,:),t,X_hat(2,:))
xlabel('time')
ylabel('Velocity')