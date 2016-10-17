clc
close all
clear all


A = [0 1 1; 0 0 0;0 0 0];
B = [0 ; 1;0];
dt = 0.001;

Ad = eye(3) + dt*A;
Bd = dt*B;

C = [1 0 0;
    0 1 0];
R = diag([.01 0.1 ]);
Q = diag([.01 .01 1]);
P_pl = eye(3);

X_hat_0 = [1;0;0];
X_0 = [0;0;10];

X(:,1)= X_0;
Y(:,1) = C*X(:,1);
t(1) = 0;
X_hat(:,1)= X_hat_0;

for i_sim = 1:1
    for i = 1:2500
        u = .5;
        
        t(i+1) = t(i)+dt;
        
        % True process
        X(:,i+1)=Ad * X(:,i) + Bd*u;
        Y(:,i+1) = C*X(:,i+1) + sqrt(R)*randn(2,1);
        
        
        % Observer model
        % Prediction based on system dynamics
        P_mi = Ad*P_pl*Ad' + Q;
        X_hat(:,i+1)=Ad * X_hat(:,i) + Bd*u;
        Y_hat(:,i+1) = C*X_hat(:,i+1);
        
        % Update based on measurement
        e_Y  = Y(:,i+1) - Y_hat(:,i+1);
        S = C*P_mi*C'+R;
        K = P_mi*C'*inv(S);
        P_pl = (eye(3) - K*C)*P_mi;
        X_hat(:,i+1)=X_hat(:,i+1) + K*e_Y;
    end
    X_hat(3,1) = X_hat(3,end);
    
end
figure;
subplot(3,1,1)
plot(t,X_hat(1,:),'--',t,X(1,:),'k')
axis([0 2.5 0 40])
subplot(3,1,2)
plot(t,X_hat(2,:),'--',t,X(2,:),'k')
axis([0 2.5 0 6])
subplot(3,1,3)
plot(t,X_hat(3,:),'--',t,X(3,:),'k')
axis([0 2.5 0 11])
