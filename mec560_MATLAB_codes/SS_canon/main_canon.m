clc
close all
clear all


N_grid = 10;

t_f =3;
v_x  = 4;
v_y = 10;

X0 = [t_f;v_x;v_y];

[C,Ceq] = cons_fun_cannon(X0);
cost= cost_fun_cannon(X0);
figure;

filename = 'di_min_time_ln_X11.gif';


for i_opt = 1:20
    
    
    options = optimset('display','iter','diffmaxchange',1.1*1e-5, ...
        'diffminchange',1e-5,'MaxFunEvals',200000,'Maxiter',2);
    
    
    X_opt = fmincon(@(X)cost_fun_cannon(X),X0,[],[],[],[],[],[],@(X)cons_fun_cannon(X),options);
    t_f = X_opt(1);
    vx = X_opt(2);
    vy = X_opt(3);
    
    t = (0:0.01:1)*t_f;
    
    X_pos = vx*t;
    Y_pos = vy*t-1/2*9.81*t.^2;
    
    plot(X_pos,Y_pos,'b',10,10,'rs',0,0,'ko');
    
    xlabel('X');
    ylabel('Y');
    axis([-2 12 -2 12])
    X0 = X_opt;
    
         frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i_opt  == 1;
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',.5);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',.5);
    end
    
    
    
end