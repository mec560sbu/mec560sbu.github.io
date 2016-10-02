clc
close all
clear all


A = [0 1; 0 0];
B = [0; 1];

sys_fun = @(t,X,u)[A*X+B*u];

X = -100:10:100;

u_lim = 1;

t = 0:.1:110;

figure;
hold on;
for i_x = 1:length(X)
    X0 = [X(i_x);0];
    u = u_lim;
    [t,X_t] = ode45( @(t,X)sys_fun(t,X,u) ,t,X0);
    plot(X_t(:,1),X_t(:,2),'r');
    plot(X_t(:,1),-X_t(:,2),'r');
    
    if X(i_x) == 0
        plot(X_t(:,1),X_t(:,2),'r','linewidth',4);
        plot(X_t(:,1),-X_t(:,2),'r','linewidth',4);
    end
    
    
    
    u = -u_lim;
    [t,X_t] = ode45( @(t,X)sys_fun(t,X,u) ,t,X0);
    plot(X_t(:,1),X_t(:,2),'k');
    plot(X_t(:,1),-X_t(:,2),'k');
    
    if X(i_x) == 0
        plot(X_t(:,1),X_t(:,2),'k','linewidth',4);
        plot(X_t(:,1),-X_t(:,2),'k','linewidth',4);
    end
    
    if X(i_x) == 10
        tt = 0:0.1:sqrt(10);
        u = -1;
        [tt,X_t] = ode45( @(t,X)sys_fun(t,X,u) ,tt,X0);
        plot(X_t(:,1),X_t(:,2),'g','linewidth',4);
        u = 1;
        X0_s = X_t(end,:);
        [tt,X_t] = ode45( @(t,X)sys_fun(t,X,u) ,tt,X0_s);
        plot(X_t(:,1),X_t(:,2),'g','linewidth',4);
        
    end
end

axis([-50 50 -10 10])
line([-50 50],[0 0],'color','k')
line([0 0],[-50 50],'color','k')
ylabel('dX')
xlabel('X')