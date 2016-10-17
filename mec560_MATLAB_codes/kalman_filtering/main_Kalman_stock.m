clc
close all
clear all


% second order model for stock price prediction
% X(t+1) = X(t) + T dX(t) + 1/2T^2 ddX(t);
% dX(t+1) =  dX(t) + T ddX(t);
% y(t) = X(t) + v(t)

% Due to high fluctuation, acceleration ddX(t) is regarded as a gaussian
% process with zero mean and a large covariance.

% X(t+1) = X(t) + T dX(t) + 1/2T^2 w(T);
% dX(t+1) =  dX(t) + T w(T);
% y(t) = X(t) + v(t)

%States = [X(t) ;dX(t)]
%  A = [1 T;
%          0 1];
% F = [ 1/2T^2 ;  T];

stocks = hist_stock_data('23012003','15042008','goog');

data_close = stocks(1).Close;

T = 1;

Ad = [1 T;
    0 1];
F = [ 1/2*T^2 ;  T];

P_mi = [1 0; 0 1];

Q = diag([1]);
R = diag([1]);
X_0 = [1;0];

X_hat_0 = [0;0];
P_pl = eye(2);

C = [1 0];

X_0 = [1;0];

X(:,1)= X_0;
Y(1) = C*X(:,1);
t(1) = 0;
X_hat(:,1)= X_hat_0;

for i = 1:length(data_close)-1
   
    
    t(i+1) = t(i)+T;
    
    % True process 
    X(:,i+1)=[data_close(i+1); 
                     (data_close(i+1)-data_close(i))/T];
    Y(i+1) = data_close(i+1);
    
    P_mi = Ad*P_pl*Ad' + F*Q*F';
    X_hat(:,i+1)=Ad * X_hat(:,i) ;
    
    
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
xlabel('Days')
ylabel('Google price')
axis([0 900 -0 800])
subplot(2,1,2)
plot(t,X(2,:),t,X_hat(2,:),'--')
xlabel('Days')
ylabel('Change in google price')
axis([0 900 -50 50])

figure;
plot(t,(X(1,:)-X_hat(1,:))./X(1,:)*100,'--')
axis([0 900 -5 5])
xlabel('Days')
ylabel('Percentage error in Google Price')








