%% Generate path
function path = genPath(x, y, x_intial, y_intial, x_end, y_end, obs_locs, buffer)

Act = zeros(length(x),length(y));
V = zeros(length(x),length(y));

[xx,yy] = meshgrid(x,y);

figure;
plot(xx,yy,'k.')
hold on;

x_start = 0;
y_start = 0;

[i_x,i_y]= xy_to_indices(x_end,y_end);
ix_end = i_x;
iy_goal = i_y;

Act(i_x,i_y) = 0;
plot(x_end,y_end,'ro')

% Specify buffer size & create virtual buffer

obs_buffer = zeros(length(obs_locs),4);
for i = 1:size(obs_buffer,1)
    obs_buffer(i,1) = obs_locs(i,1)-buffer;
    obs_buffer(i,2) = obs_locs(i,2)-buffer;
    obs_buffer(i,3) = obs_locs(i,3)+buffer;
    obs_buffer(i,4) = obs_locs(i,4)+buffer;
end

% Obstacle polygon creation
for i = 1:size(obs_locs,1)
    patch( [obs_locs(i,1) obs_locs(i,1)  obs_locs(i,3) obs_locs(i,3)  ], ...
        [obs_locs(i,2) obs_locs(i,4)  obs_locs(i,4) obs_locs(i,2)  ],'g' );
    
    [ix_obs_st,iy_obs_st]= xy_to_indices( obs_locs(i,1), obs_locs(i,2));
    [ix_obs_en,iy_obs_en]= xy_to_indices( obs_locs(i,3), obs_locs(i,4));
    
    [ix_buffer_st,iy_buffer_st]= xy_to_indices( obs_buffer(i,1), obs_buffer(i,2));
    [ix_buffer_en,iy_buffer_en]= xy_to_indices( obs_buffer(i,3), obs_buffer(i,4));
    
    Act(ix_buffer_st:ix_buffer_en,iy_buffer_st:iy_buffer_en) =  1000;
end


close all
figure;
V = 2000*ones(size(Act));
gam = .9;
filename = ['Value_growth' num2str(i) '.gif'];

changed = 1;
i_V = 1;
while changed == 1
    changed = 0;
    V_old = V;
    for i_x = 1:length(x)
        for i_y = 1:length(y)
            
            if (i_x == ix_end) &&(i_y == iy_goal)
                if V(i_x,i_y) > 0
                    changed = 1;
                    V(i_x,i_y) = 0;
                end
            end
            
            if Act(i_x,i_y) ~= 1000;
                iv_x = [1 -1 0 0 1 -1 1  -1];
                iv_y = [0 0 -1 1  1 -1 -1 1];
                V_new = [];
                for i_v = 1:8,
                    val = check_ind(i_x+iv_x(i_v),i_y+iv_y(i_v));
                    if val == 1
                        V_new  = V(i_x+iv_x(i_v),i_y+iv_y(i_v)) + 10*sqrt(iv_x(i_v)^2+iv_y(i_v)^2);
                        
                        if V_new< V(i_x,i_y)
                            V(i_x,i_y) = V_new;
                            changed = 1;
                            
                        end
                    end
                end
                
                
            else
                V(i_x,i_y) = 2000;
            end
            
            
            
            
            
        end
    end
    
    
    surf(yy,xx,V_old);xlabel('X');ylabel('Y');zlabel('Value')
    title(['Value after ' num2str(i_V) ' iterations.'])
    axis([0 10 0 10 -200 2100])
    view(i_V*3,30);
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i_V == 1;
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',.1);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',.1);
    end
    i_V = i_V+1;
end


for i = i_V:i_V+60
    surf(yy,xx,V);xlabel('X');ylabel('Y');zlabel('Value')
    title(['Value after ' num2str(i) ' iterations (Converged).'])
    axis([0 10 0 10 -200 2100])
    view(i*3,30);
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',.1);
    
end



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
m = 0;
path=[];

for i = 1:length(x_intial)
    [i_x,i_y]= xy_to_indices(x_intial(i),y_intial(i));
    stop_mov = 0;
    while stop_mov == 0
        m = m+1;
        iv_x = [1 -1 0 0 1 -1 1  -1];
        iv_y = [0 0 -1 1  1 -1 -1 1];
        for i_v = 1:8,
            Va(i_v) = V(i_x+iv_x(i_v),i_y+iv_y(i_v)) + 10*sqrt(iv_x(i_v)^2+iv_y(i_v)^2);
        end
        
        [V_min , i_vmin]= min(Va);
        x_intial(i) = x( i_x+iv_x(i_vmin));
        y_intial(i) = y( i_y+iv_y(i_vmin));
        path(m,:) = [x_intial y_intial];
        plot(x_intial(i),y_intial(i),'bx')
        plot(x_intial(i),y_intial(i),'b*')
        
        if (i_x==ix_end)&&(i_y==iy_goal)
            stop_mov = 1;
        end
        
        frame = getframe(1);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if i_move == 1;
            imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',.1);
        else
            imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',.1);
        end
        i_move = i_move+1;
        
        [i_x,i_y]= xy_to_indices(x_intial(i),y_intial(i));
        
        pause(0.01);
    end
end
