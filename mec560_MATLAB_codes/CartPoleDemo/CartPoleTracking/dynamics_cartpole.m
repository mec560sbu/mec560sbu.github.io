function dq = dynamics_cartpole(t,q)

model_params;

x = q(1);
th = q(2);
dx = q(3);
dth = q(4);

% Generating control based on linearized system:

if (norm(th)<.1 & norm(dth)<1)
    x_d = 0;
    th_d = 0;
    dx_d = 0;
    dth_d = 0;
    u_f_d = 0;
else
    generate_desired_Polynomials;
    
    x_d = ppval(pp_states.x,t);
    th_d = ppval(pp_states.th,t);
    dx_d = ppval(pp_states.dx,t);
    dth_d = ppval(pp_states.dth,t);
    u_f_d = ppval(pp_controls.u,t);
    
    
end

q_d = [x_d;th_d;dx_d;dth_d];

ff_d = ff_cartpole(x_d,th_d,dx_d,dth_d,m_cart,m_mass,l,u_f_d,g);
M_d = M_cartpole(x_d,th_d,dx_d,dth_d,m_cart,m_mass,l,u_f_d,g);
dMdq_d = dMdq_cartpole(x_d,th_d,dx_d,dth_d,m_cart,m_mass,l,u_f_d,g);
dfdq_d = dfdq_cartpole(x_d,th_d,dx_d,dth_d,m_cart,m_mass,l,u_f_d,g);
dfd_dq_d = dfd_dq_cartpole(x_d,th_d,dx_d,dth_d,m_cart,m_mass,l,u_f_d,g);
dfdu_d = dfdu_cartpole(x_d,th_d,dx_d,dth_d,m_cart,m_mass,l,u_f_d,g);


u_f = -[100 100 10 10]*(q -q_d)  + u_f_d;

ff = ff_cartpole(x,th,dx,dth,m_cart,m_mass,l,u_f,g);
M = M_cartpole(x,th,dx,dth,m_cart,m_mass,l,u_f,g);


ddq = M\ff;

dq = [q(3:4);
    ddq];