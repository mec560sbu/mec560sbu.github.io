clc;close all;clear all;


vx = .2;
vy = 1;
g = 9.81;
options = odeset('Events',@hit_event, 'RelTol',1e-8,'AbsTol',1e-10);
x_0 = [0;0;vy];
t_start = 0;
X_all = [];
t_all = [];
for i = 1:10;
    [t,X] = ode45(@(t,x)eqn_bouncing_ball(t,x,g,vx), [t_start:0.01:t_start+10],x_0,options);
    X_all = [X_all ; X];
    t_all = [t_all; t];
    x_0 = X_all(end,:);
    x_0(3) = -0.9*x_0(3);
    t_start = t(end);
end


close all
% Generating GIF
figure;

filename = 'bouncing_ball.gif';

for i = 1:length(t_all)
    plot(X_all(i,1),X_all(i,2),'ro','markerfacecolor','r')
    hold on;
    plot(X_all(1:i,1),X_all(1:i,2))
    axis([0 .3 -.01 .095])
    patch([0 0 .3 .3],[-.01 0 0 -.01],'r')
    frame = getframe;
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i == 1;
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',.01 );
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',.01);
    end
    
end