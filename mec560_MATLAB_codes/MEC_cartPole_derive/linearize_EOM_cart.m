syms x_d dx_d  'real'
syms th_d dth_d 'real'
syms u_th_d 'real'

vars_d = {'x_d','th_d','dx_d','dth_d','u_th_d', ...
    'm_cart','m_mass','l','g'};


tau_d = u_th_d;

tau_v = [u_th];

X_q = [x;th; dx;dth];
X_q_d = [x_d;th_d; dx_d;dth_d];

[dfdq,dMdq,dfdu] = linearize_MF(Mr,ff,X_q,tau_v,X_q_d,tau_d);



write_diff = 1;
if write_diff == 1;
    write_file(dfdq,'dfdq_cartpole.m',vars_d); % Writing dFFdQ
    write_file(dfdu,'dfdu_cartpole.m',vars_d); % Writing dFFdU
    write_file(dMdq,'dMdq_cartpole.m',vars_d); % Writing dMdQ

end



%% Test if calculations are correct. 
% Only possible with small number of states. 

Mr_d = subs(Mr,[x th],[x_d th_d]);

Mr_d= simplify(simplify(Mr_d));

Mi = inv(simplify(Mr));

for i = 1:length(X_q)/2
    dmidq(:,:,i) = subs(diff(Mi,X_q(i)),[x;th],[x_d;th_d]);
    
    dmidq_c(:,:,i) = - inv(Mr_d)*dMdq(:,:,i)*inv(Mr_d);
    
    simplify(dmidq(:,:,i)-dmidq_c(:,:,i)) % SHould return 0 if calculations are correct. 
end



