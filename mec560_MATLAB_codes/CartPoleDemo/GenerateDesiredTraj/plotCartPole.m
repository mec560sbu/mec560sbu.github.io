% Plot Cartpole
close all
model_params

time = solution.phase.time;
states = solution.phase.state;
controls = solution.phase.control;


pp_x = spline(time,states(:,1));
pp_th = spline(time,states(:,2));
pp_dx = spline(time,states(:,3));
pp_dth = spline(time,states(:,4));
pp_u = spline(time,controls);

pp_states.x = pp_x;
pp_states.th = pp_th;
pp_states.dx = pp_dx;
pp_states.dth = pp_dth;
pp_controls.u = pp_u;





dt_sim = .01;
tf = time(end);
N_sim = tf/dt_sim;

t_new = (0:1/(N_sim-1):1)*tf;

filename = ['DC_gamma_' num2str(gamma) '.gif'];

figure;
for i = 1:1:length(t_new)
    
    x_st = ppval(t_new(i),pp_states.x);
    th_st = ppval(t_new(i),pp_states.th);
    dx_st = ppval(t_new(i),pp_states.dx);
    dth_st = ppval(t_new(i),pp_states.dth);
    
    
    plot(x_st,0,'ko')
    hold on
    patch([x_st+.5, x_st+.5, x_st-.5, x_st-.5],...
        [-.2, .2, .2, -.2],...
        'red')
    
    axis([-5 5 -4 6])
    plot(x_st,0,'ko')
    
    u_f = 0; % set 0 for now, as control doesnt come in eqn for state.
    pos_mass = P_mass(x_st,th_st,dx_st,dth_st,m_cart,m_mass,l,u_f,g);
    
    plot(pos_mass(1),pos_mass(2),'ko')
    xlabel('X')
    ylabel('Y')
    title(['Cost = t_f + ' num2str(gamma) ' \times U^2' ])
    text(.5,2.5,['t = ' num2str(t_new(i))] )
    
    
    
    line([ x_st pos_mass(1)],[0 pos_mass(2)])
    p1 = [x_st .5];
    dp = [.5 .5];    
%     quiver(p1(1),p1(2),0,dp(2),0)
%     axis equal
    hold off
    
    

    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i  == 1;
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',.01);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',.01);
    end
    
    pause(.1)
end


