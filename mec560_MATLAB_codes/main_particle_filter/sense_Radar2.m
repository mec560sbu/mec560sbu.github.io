function [D_sense,P_sense = sense_Radar(x_robot,y_robot,range,obs_locs,th)
    
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

for i_th = 1:length(th_all)

end

% x_th = range* cos(th_all)+x_robot;
% y_th = range* sin(th_all)+y_robot;
% x_line = [x_robot*ones(size(x_th)) x_th];
% y_line = [y_robot*ones(size(y_th)) y_th];
% line([x_robot x_th(i_th)],[y_robot y_th(i_th)]);
% test1 =  (x_robot-obs_pos(i_obs,1))*(x_th(i_th)-obs_pos(i_obs,1))<0;
% test2 =  (x_robot-obs_pos(i_obs,3))*(x_th(i_th)-obs_pos(i_obs,3))<0;
% test3 =  (y_robot-obs_pos(i_obs,2))*(y_th(i_th)-obs_pos(i_obs,2))<0;
% test4 =  (y_robot-obs_pos(i_obs,4))*(y_th(i_th)-obs_pos(i_obs,4))<0;
% th_all(i_th);
% test = test1 + test2 + test3 + test4;
% P_sense(i_th,:) = [x_th(i_th) y_th(i_th)];
% if test ~= 0,
%     X_pts = [];
%     Y_pts = [];
%     if test1 ~= 0
%         X_pts = [X_pts obs_pos(i_obs,1)];
%         y_i = y_robot + sin(atan2(y_th(i_th)-y_robot,x_th(i_th)-x_robot));
%         Y_pts = [Y_pts y_i];
%     end
%     if test2 ~= 0
%         X_pts = [X_pts obs_pos(i_obs,3)];
%         y_i = y_robot + sin(atan2(y_th(i_th)-y_robot,x_th(i_th)-x_robot));
%         Y_pts = [Y_pts y_i];
%     end
%     if test3 ~= 0
%         Y_pts = [Y_pts obs_pos(i_obs,2)];
%         x_i = x_robot + cos(atan2(y_th(i_th)-y_robot,x_th(i_th)-x_robot));
%         X_pts = [X_pts x_i];
%     end
%     if test4 ~= 0
%         Y_pts = [Y_pts obs_pos(i_obs,4)];
%         x_i = x_robot + cos(atan2(y_th(i_th)-y_robot,x_th(i_th)-x_robot));
%         X_pts = [X_pts x_i];
%     end
%     
%     d = (X_pts-x_robot).^2 + (Y_pts-y_robot).^2;
%     [min_d,ind_min_d] =min(d);
%     
%     P_sense(i_th,:) = [X_pts(ind_min_d) Y_pts(ind_min_d) ];
% end
% end


% D_sense = sqrt((P_sense(:,1)-x_robot).^2 + (P_sense(:,2)-y_robot).^2 );