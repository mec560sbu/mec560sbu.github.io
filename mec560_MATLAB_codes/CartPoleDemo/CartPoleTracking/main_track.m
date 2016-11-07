clc
close all
clear all

q = [0;pi;0;0];

t = 0;

dq = dynamics_cartpole(t,q);
time_span = 0:0.01:2.5;


[time,states] = ode45(@dynamics_cartpole,time_span,q);



figure;
subplot(2,1,1)
plot(time,states(:,1))
xlabel('time')
ylabel('x')
subplot(2,1,2)
plot(time,states(:,2))
xlabel('time')
ylabel('angle')

plotCartPole
