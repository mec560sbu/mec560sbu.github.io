% Plot Cartpole
close all
model_params




filename = ['Stabilize_LQR.gif'];

figure;
for i = 1:length(time)
    
     x_st = states(i,1);
    th_st = states(i,2);
    dx_st = states(i,3);
    dth_st = states(i,4);
    
    
    plot(x_st,0,'ko')
    hold on
    patch([x_st+.5, x_st+.5, x_st-.5, x_st-.5],...
        [-.2, .2, .2, -.2],...
        'red')
    
    axis([-2 2 -1.4 2.6])
    plot(x_st,0,'ko')
    
    u_f = 0; % set 0 for now, as control doesnt come in eqn for state.
    pos_mass = P_mass(x_st,th_st,dx_st,dth_st,m_cart,m_mass,l,u_f,g);
    
    plot(pos_mass(1),pos_mass(2),'ko')
    xlabel('X')
    ylabel('Y')
    text(.5,2.5,['t = ' num2str(time(i))] )
    
    
    
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


