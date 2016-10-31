function dq_all = get_dynamics(x_all,th_all,dx_all,dth_all,u_all)


model_params;

for i = 1:length(th_all);
    
    th = th_all(i);    
    x = x_all(i);    
    dth = dth_all(i);    
    dx = dx_all(i);
    u_th = u_all(i);
    ff = ff_cartpole(x,th,dx,dth,m_cart,m_mass,l,u_th,g);
    M = M_cartpole(x,th,dx,dth,m_cart,m_mass,l,u_th,g);


    ddq = M\ff;
    
    dq_all(i,:) = [dx dth ddq'];
end