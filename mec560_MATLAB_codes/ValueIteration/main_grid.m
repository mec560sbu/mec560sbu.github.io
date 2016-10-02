clc
close all
clear all


x = 0:.1:4;
y = 0:.1:4;

x_agent = [.53];
y_agent = [3.5];

Act = zeros(length(x),length(y));
V = zeros(length(x),length(y));
V_new = zeros(length(x),length(y));

[xx,yy] = meshgrid(x,y);

figure;
plot(xx,yy,'k.')
hold on;
x_goal = 2;
y_goal = 2;


x_start = 0;
y_start = 0;

[i_x,i_y]= xy_to_indices(x_goal,y_goal);
ix_goal = i_x;
iy_goal = i_y;

i_do_nth = [i_x,i_y];

Act(i_x,i_y) = 100;
plot(x_goal,y_goal,'ro')

obs_locs = [1 2  1.5 2.5;
    .5 3  2.5 3.5;
    .5 .5  2.5 1;
    2.25 1  2.5 2.5;
    ];


for i = 1:size(obs_locs,1)
    patch( [obs_locs(i,1) obs_locs(i,1)  obs_locs(i,3) obs_locs(i,3)  ], ...
        [obs_locs(i,2) obs_locs(i,4)  obs_locs(i,4) obs_locs(i,2)  ],'g' );
    
    [ix_obs_st,iy_obs_st]= xy_to_indices( obs_locs(i,1), obs_locs(i,2));
    [ix_obs_en,iy_obs_en]= xy_to_indices( obs_locs(i,3), obs_locs(i,4));
    
    Act(ix_obs_st:ix_obs_en,iy_obs_st:iy_obs_en) = -100;
end


close all
figure;
V = Act;
gam = .95;
filename = ['Value_growth' num2str(i) '.gif'];

for i_Q = 1:80
    for i_x = 2:length(x)-1
        for i_y = 2:length(y)-1
            
            
            iv_x = [1 0 -1 1 0 -1 1 0 -1];
            iv_y = [1 1 1 0 0 0 -1 -1 -1];
            for i_v = 1:9,
                d =sqrt( (iv_x(i_v))^2 + (iv_y(i_v))^2 );
                Va(i_v) = V(i_x+iv_x(i_v),i_y+iv_y(i_v));
                Rew(i_v) = -10*d + Act(i_x+iv_x(i_v),i_y+iv_y(i_v));
            end
            
            [V_max , i_vmax]= max(Va);
            V_new(i_x,i_y) =  gam*V_max + Rew(i_vmax);
            if V(i_x,i_y) == -100,
                V_new(i_x,i_y) = V(i_x,i_y);
            end
        end
    end
    if mod(i_Q,1) == 0
        
        surf(yy,xx,V);xlabel('X');ylabel('Y');zlabel('Value')
        title(['Value after ' num2str(i_Q) ' iterations.'])
        axis([0 4 0 4 -200 2100])
        view(i_Q*6,30);
        frame = getframe(1);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if i_Q == 1;
            imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',.1);
        else
            imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',.1);
        end
    end
    
    V = V_new;
    
    
end


save V.mat V


filename = ['Obs_Avoidance' num2str(i) '.gif'];
close all

figure;
plot(xx,yy,'k.')
hold on;
for i = 1:size(obs_locs,1)
    patch( [obs_locs(i,1) obs_locs(i,1)  obs_locs(i,3) obs_locs(i,3)  ], ...
        [obs_locs(i,2) obs_locs(i,4)  obs_locs(i,4) obs_locs(i,2)  ],'g' );
    
end

Va = [];
hold on;
i_move = 1;
for i = 1:length(x_agent)
    [i_x,i_y]= xy_to_indices(x_agent(i),y_agent(i));
    stop_mov = 0
    while stop_mov == 0
        iv_x = [1 0 -1 1  -1 1 0 -1];
        iv_y = [1 1 1 0 0 -1 -1 -1];
        for i_v = 1:8,
            Va(i_v) = V(i_x+iv_x(i_v),i_y+iv_y(i_v));
        end
        
        [V_max , i_vmax]= max(Va);
        x_agent(i) = x( i_x+iv_x(i_vmax));
        y_agent(i) = y( i_y+iv_y(i_vmax));
        plot(x_agent(i),y_agent(i),'bx')
        plot(x_agent(i),y_agent(i),'b*')
        
        if (i_x==ix_goal)&(i_y==iy_goal)
            stop_mov = 1;
        end
        
        title(['Value after ' num2str(i_Q) ' iterations.'])
        frame = getframe(1);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if i_move == 1;
            imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',.1);
        else
            imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',.1);
        end
        i_move = i_move+1;
        
        
        [i_x,i_y]= xy_to_indices(x_agent(i),y_agent(i));
        
        
        pause(0.01);
    end
end

plot_gif_V;
