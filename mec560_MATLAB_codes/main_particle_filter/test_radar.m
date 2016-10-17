
x_robot = 6
y_robot = 4
range =8


plot_env
hold on

p_robot_LB = [x_robot-range,y_robot-range];
p_robot_LT = [x_robot-range,y_robot+range];
p_robot_RB = [x_robot+range,y_robot-range];
p_robot_RT = [x_robot+range,y_robot+range];
ind_np_B = find(obs_locs(:,2)>=p_robot_RT(2));
ind_np_L = find(obs_locs(:,1)>=p_robot_RT(1));
ind_np_LB =  union(ind_np_B,ind_np_L);
ind_np_T = find(p_robot_LB(2)>=obs_locs(:,4) );
ind_np_R = find(p_robot_LB(2)>=obs_locs(:,3) );
ind_np_TR =  union(ind_np_T,ind_np_R);
ind_np = union(ind_np_LB,ind_np_TR);

obs_pos = obs_locs;
obs_pos(ind_np,:) = [];
th_all = (th:th:2*pi)';


x_th = range* cos(th_all)+x_robot;
y_th = range* sin(th_all)+y_robot;
x_line = [x_robot*ones(size(x_th)) x_th];
y_line = [y_robot*ones(size(y_th)) y_th];

d_all = 100*ones(size(th_all));



for i_th = 1:length(th_all)
    
    X_radar(i_th) = x_th(i_th);
    Y_radar(i_th) = y_th(i_th);
    
    for i_obs = 1:size(obs_pos,1),
        
        X_pts = [];
        Y_pts = [];
        % left edge
        x_i = obs_pos(i_obs,1);
        if (x_robot-x_i)*(x_i-x_th(i_th))>=0
            if abs(y_robot-y_th(i))>1e-6
                y_i = y_robot + (x_i - x_robot)*((y_th(i_th) - y_robot)/(x_th(i_th) - x_robot));
            else
                y_i = (y_robot + y_th(i))/2;
            end
            if ((obs_pos(i_obs,2)-y_i)*(y_i-obs_pos(i_obs,4))>0)
                X_pts = [X_pts x_i];
                Y_pts = [Y_pts y_i];
            end
        end
        
        % Right edge
        x_i = obs_pos(i_obs,3);
        if (x_robot-x_i)*(x_i-x_th(i_th))>=0
            if abs(y_robot-y_th(i))>1e-6
                y_i = y_robot + (x_i - x_robot)*((y_th(i_th) - y_robot)/(x_th(i_th) - x_robot));
            else
                y_i = (y_robot + y_th(i))/2;
            end
            if ((obs_pos(i_obs,2)-y_i)*(y_i-obs_pos(i_obs,4))>0)
                X_pts = [X_pts x_i];
                Y_pts = [Y_pts y_i];
            end
        end
        
        
        % Top edge
        y_i = obs_pos(i_obs,4);
        if (y_robot-y_i)*(y_i-y_th(i_th))>=0
            if abs(x_robot-x_th(i))>1e-6
                x_i = x_robot + (y_i - y_robot)*((x_th(i_th) - x_robot)/(y_th(i_th) - y_robot));
            else
                x_i = (x_robot + x_th(i))/2;
            end
            if ((obs_pos(i_obs,1)-x_i)*(x_i-obs_pos(i_obs,3))>0)
                X_pts = [X_pts x_i];
                Y_pts = [Y_pts y_i];
            end
        end
        
        % Bottom edge
        y_i = obs_pos(i_obs,2);
        if (y_robot-y_i)*(y_i-y_th(i_th))>=0
            if abs(x_robot-x_th(i))>1e-6
                x_i = x_robot + (y_i - y_robot)*((x_th(i_th) - x_robot)/(y_th(i_th) - y_robot));
            else
                x_i = (x_robot + x_th(i))/2;
            end
            if ((obs_pos(i_obs,1)-x_i)*(x_i-obs_pos(i_obs,3))>0)
                X_pts = [X_pts x_i];
                Y_pts = [Y_pts y_i];
            end
        end
        
        
        if length(X_pts) ~=0
            
            d_radar  = (X_pts-x_robot).^2 +  (Y_pts-y_robot).^2;
            [d_min,i_d_min] = min(d_radar);
            if d_min < d_all(i_th)
                X_radar(i_th)  = X_pts(i_d_min);
                Y_radar(i_th)  = Y_pts(i_d_min);
                d_all(i_th) = d_min;
            end
        end
        
        
    end
end


plot(X_radar,Y_radar,'ms')
for i=1:length(X_radar)
    line([X_radar(i) x_robot],[Y_radar(i), y_robot])
end