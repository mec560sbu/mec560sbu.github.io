figure;
V = Act;
gam = .9;
for i_Q = 1:1000
    for i_x = 2:length(x)-1
        for i_y = 2:length(y)-1
            
            
            iv_x = [1 0 -1 1 0 -1 1 0 -1];
            iv_y = [1 1 1 0 0 0 -1 -1 -1];
            for i_v = 1:9,
                d =sqrt( (iv_x(i_v))^2 + (iv_y(i_v))^2 );
                Va(i_v) = V(i_x+iv_x(i_v),i_y+iv_y(i_v));
                Rew(i_v) = -d + Act(i_x+iv_x(i_v),i_y+iv_y(i_v));
            end
            
            [V_max , i_vmax]= max(Va);
            V_new(i_x,i_y) =  gam*V_max + Rew(i_vmax);
            if V(i_x,i_y) == -100,
                V_new(i_x,i_y) = V(i_x,i_y);
            end
        end
    end
    V = V_new;
    
end
figure
surf(V_new);

x_agent = [3,1,.2,2];
y_agent = [1.5,.4,3.8,3.8];


figure;
plot(xx,yy,'k.','markersize',2)
hold on;
for i = 1:size(obs_locs,1)
    patch( [obs_locs(i,1) obs_locs(i,1)  obs_locs(i,3) obs_locs(i,3)  ], ...
        [obs_locs(i,2) obs_locs(i,4)  obs_locs(i,4) obs_locs(i,2)  ],'r','facealpha',0.5 );
    
end

Va = [];
hold on;
for i = 1:length(x_agent)
    [i_x,i_y]= xy_to_indices(x_agent(i),y_agent(i));
    stop_mov = 0;
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
        
        [i_x,i_y]= xy_to_indices(x_agent(i),y_agent(i));
        
        pause(0.01);
    end
end

plot(x_goal,y_goal,'go','markerfacecolor','g')

