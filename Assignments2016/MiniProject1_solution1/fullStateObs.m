function [e_est, dt] = fullStateObs(A, B, C, K, p, time, x_des, y_des)

L_t = place(A',C',p);
L = L_t';
eig(A-L*C);

% Define error between desired and estimate
dt = time(2) - time(1);

e_des(:,1) = [0;0;0;0];
y(:,1) = C*e_des+[x_des(1,1);y_des(1,1)];

e_est(:,1) = [-1.1;-1;0;0];
y_est(:,1) = C*e_est+[x_des(1,1);y_des(1,1)];

for i =2:length(time)
    u=-K*e_est(:,i-1);
    
    e_des(:,i) = e_des(:,i-1)+dt*(A*e_des(:,i-1) + B*u);
    y(:,i) = C*e_des(:,i)+[x_des(1,i);y_des(1,i)];

    e_est(:,i) = e_est(:,i-1)+dt*(A*e_est(:,i-1)+B*u +L*(y(:,i-1)-C*e_est(:,i-1)-[x_des(1,i);y_des(1,i)]));
    y_est(:,i) = C*e_est(:,i)+[x_des(1,i);y_des(1,i)];
end


plot( y_est(1,:),  y_est(2,:),'r.',y_est(1,:),  y_est(2,:),'r','linewidth',1)
plot(y(1,:),  y(2,:),'k','linewidth',2)


figure
plot(time, y_est(1,:), time, y_est(2,:))
title('Position (Estimate)')
ylabel('Position')
xlabel('Time')
legend('q_1','q_2', 'Location', 'southeast')
grid on
grid minor

figure;
subplot(2,1,1)
plot(time,e_des(1,:),'--',time,e_est(1,:),time,e_des(2,:),'--',time,e_est(1,:))
title('States and observer estimates')
ylabel('Position')
legend('q_1','q_1 (Estimate)','q_2','q_2 (Estimate)', 'Location', 'southeast')
subplot(2,1,2)
plot(time,e_des(3,:),'--',time,e_est(3,:),time,e_des(4,:),'--',time,e_est(4,:))
ylabel('Velocity')
xlabel('Time')
legend('q_3','q_3 (Estimate)','q_4','q_4 (Estimate)')

figure;
subplot(2,1,1)
plot(time,e_des(1,:)-e_est(1,:),time,e_des(2,:)-e_est(2,:))
ylabel('Position')
title('Error: states - observer estimates')
legend('q_1 Error','q_2 Error', 'Location', 'southeast')
subplot(2,1,2)
plot(time,e_des(3,:)-e_est(3,:),time,e_des(4,:)-e_est(4,:))
ylabel('Velocity')
xlabel('Time')
legend('q_3 Error','q_4 Error', 'Location', 'southeast')