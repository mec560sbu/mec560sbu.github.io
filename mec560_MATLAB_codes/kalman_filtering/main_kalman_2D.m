clc
close all
clear all

mu = [0 0];
Sigma = [.1 0; 0 10];

x2 = -5:.01:5; x1 = -1:.01:9;

pos = mu(1)+Sigma(1,1)*randn(1000,1);
vel = mu(2)+Sigma(2,2)*randn(1000,1);

figure;
hold on
scatter(pos,vel,'filled','b','markerfacealpha',0.1)
axis([-1 9 -5 5])
xlabel('Position'); ylabel('Velocity');

figure;
hold on
scatter(pos,vel,'filled','b','markerfacealpha',0.1)
scatter(pos+vel,vel,'filled','r','markerfacealpha',0.1)
axis([-1 9 -5 5])
xlabel('Position'); ylabel('Velocity');


figure;
hold on
scatter(pos+1,vel,'filled','b','markerfacealpha',0.1)
scatter(pos+vel,vel,'filled','r','markerfacealpha',0.1)
axis([-1 9 -5 5])
xlabel('Position'); ylabel('Velocity');


mu = [1 1];
Sigma = [.01 .0070; .007 .014];

xx1 = .7:.001:1.3;
norm = normpdf(xx1,1,.047);

figure;
hold on
plot(xx1,.02+.2*norm-5,'b', 'linewidth',2)
plot(.02+.05*norm-1,xx1,'k','linewidth',2)
scatter(pos+1,vel,'filled','b','markerfacealpha',0.1)
scatter(pos+vel,vel,'filled','r','markerfacealpha',0.1)
plot_gauss_2D(mu,Sigma,x1,x2,'k')
axis([-1 9 -5 5])

% 