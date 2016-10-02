clc
close all
clear all


rng(20);
N_grid =31;

t_f = 8;
states = randn(2*N_grid,1);
control = 0*randn(N_grid,1);

X0 = [t_f;states;control];



lb = [0.01;
    -Inf*ones(size(states));
    -1*ones(size(control));
    ];
ub = [20;
    Inf*ones(size(states));
    1*ones(size(control));
    ];
% lb = [];
% ub = [];

% x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
% filename = 'di_min_time_ln_X21.gif';

mov = [];

close all
figure
X_opt = X0;
% load X_opt.mat
tic

type = 1;
for i_opt = 1:2,
    X0 = X_opt;
    options = optimset('display','iter','diffmaxchange',1.1*1e-5, ...
        'diffminchange',1e-5,'MaxFunEvals',200000,'Maxiter',1000);
    
    
    if i_opt == 1
        X_opt = X0;
    else
        X_opt = fmincon(@(X)obj_cost(X,N_grid,type),X0,[],[],[],[],lb,ub,@(X)cons_fun_doubleIntegrator(X,type),options);
        
    end
    % Cost fun
    t_f = X_opt(1);
    X1 = X_opt(2:N_grid+1);
    X2 = X_opt(2+N_grid:2*N_grid+1);
    u = X_opt(2+2*N_grid:3*N_grid+1);
    sol_all = [];
    t_all = [];
    
    time_points = (0:1/(N_grid-1):1)*t_f;
    for i = 1:(N_grid-1)
        X_0(1,:) = X1(i);
        X_0(2,:) = X2(i);
        t_start = time_points(i);
        t_end = time_points(i+1);
        
        tt = t_start:(t_end-t_start)/10:t_end;
        [t,sol_int] = ode45(@(t,y)sys_dyn_doubleIntegrator(t,y,X_opt,type),tt,X_0);
        sol_all = [sol_all;sol_int];
        t_all = [t_all;t];
    end
    
    X_colpt(1,:) = X_opt(2:N_grid+1);
    X_colpt(2,:) = X_opt(2+N_grid:2*N_grid+1);
    u_colpt(1,:) = X_opt(2+2*N_grid:3*N_grid+1);
    
    plot(t_all,sol_all,time_points(1:end-1),X_colpt(:,1:end-1),'ks',time_points(2:end),X_colpt(:,2:end),'ro')
    xlabel('time');
    ylabel('states');
    title('Progress of optimization')
    axis([0 t_f -1.2 10.8])
    
    
    switch type
        case 1
            u_t = interp1(time_points,u,t_all,'previous');
        case 2
            u_t = interp1(time_points,u,t_all);
        case 3
            u_t = interp1(time_points,u,t_all,'spline');
    end
%     
    plot(t_all,u_t,time_points(1:end-1),u_colpt(:,1:end-1),'ks',time_points(2:end),u_colpt(:,2:end),'ro')
    xlabel('time');
    ylabel('Control');
    title(['Progress of optimization: iteration # ' num2str(10*i_opt)])
    axis([0 t_f -1.2 1.2])
    
%     frame = getframe(1);
%     im = frame2im(frame);
%     [imind,cm] = rgb2ind(im,256);
%     if i_opt  == 1;
%         imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',.5);
%     else
%         imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',.5);
%     end
%     
    
    pause(0.1)
    
end
toc