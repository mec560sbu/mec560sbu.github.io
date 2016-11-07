function dq = dynamics_cartpole(t,q)

model_params;

x = q(1);
th = q(2);
dx = q(3);
dth = q(4);

% Generating control based on linearized system:
x_d = 0;
th_d = 0;
dx_d = 0;
dth_d = 0;
u_f_d = 0;

ff_d = ff_cartpole(x_d,th_d,dx_d,dth_d,m_cart,m_mass,l,u_f_d,g);
M_d = M_cartpole(x_d,th_d,dx_d,dth_d,m_cart,m_mass,l,u_f_d,g);
dMdq_d = dMdq_cartpole(x_d,th_d,dx_d,dth_d,m_cart,m_mass,l,u_f_d,g);
dfdq_d = dfdq_cartpole(x_d,th_d,dx_d,dth_d,m_cart,m_mass,l,u_f_d,g);
dfd_dq_d = dfd_dq_cartpole(x_d,th_d,dx_d,dth_d,m_cart,m_mass,l,u_f_d,g);
dfdu_d = dfdu_cartpole(x_d,th_d,dx_d,dth_d,m_cart,m_mass,l,u_f_d,g);

[A_lin,B_lin] = get_linearized_system(dMdq_d,dfdq_d,dfd_dq_d,dfdu_d,M_d,ff_d);

Q = 10*diag([1 1 1 1]);
R = 0.001;

P = care(A_lin,B_lin,Q,R); % Generate control based on linearized system
K = inv(R)*B_lin'*P;

u_f = -1*K*q;

ff = ff_cartpole(x,th,dx,dth,m_cart,m_mass,l,u_f,g);
M = M_cartpole(x,th,dx,dth,m_cart,m_mass,l,u_f,g);


ddq = M\ff;

dq = [q(3:4);
            ddq];