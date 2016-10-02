clc
close all
clear all 

t_f = 2;
dt = 0.001;
t_dis = 0:dt:t_f;
N = length(t_dis);
S{N} = 100*eye(2);
R = dt;
Q = .0001*eye(2)*dt;
A = [1 dt;0 1];
B = [0 ;dt];
K{N} = [0 0];

for i = N-1:-1:1
    %K{i} = inv(R + B'*S{i+1}*B)*B'*S{i+1}*A;
    %S{i} = Q + K{i}'*R*K{i} + (A-B*K{i})'*S{i+1}*(A-B*K{i});
    %S{i} = Q+ A'*S{i+1}*A - A'*S{i+1}*B*inv(R+B'*R*B)*B'*S{i+1}*A;    
    S{i} =  A'*inv(eye(2) + S{i+1}*B*inv(R)*B')*S{i+1}*A + Q; % Third form
    K_norm(i) = norm(K{i});
end

X(:,1) = [1;0];
X_dlqr(:,1) = [1;0];
P_dlqr = dare(A,B,Q,R);
K_dlqr = inv(R)*B'*P_dlqr;

for i = 2:N
    u(i-1) = -inv(R)*B'*S{i-1}*X(:,i-1);
    X(:,i) = A * X(:,i-1) + B*u(i-1);
    X_dlqr(:,i) = A * X_dlqr(:,i-1) - B*K_dlqr*X_dlqr(:,i-1) ;

end


figure;
subplot(2,1,1)
plot(t_dis,X(1,:),t_dis,X_dlqr(1,:))
legend('DyP','LQR')
ylabel('position')
subplot(2,1,2)
plot(t_dis,X(2,:),t_dis,X_dlqr(2,:))
legend('DyP','LQR')
ylabel('velocity')
xlabel('time')