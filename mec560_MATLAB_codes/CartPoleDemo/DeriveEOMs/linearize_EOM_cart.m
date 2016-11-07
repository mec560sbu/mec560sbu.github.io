syms x_d dx_d  'real'
syms th_d dth_d 'real'
syms u_th_d 'real'

vars_d = {'x_d','th_d','dx_d','dth_d','u_th_d', ...
    'm_cart','m_mass','l','g'};


tau_d = u_th_d;

tau_v = [u_th];

X_q = [x;th; dx;dth];
X_q_d = [x_d;th_d; dx_d;dth_d];

[dfdq,dfd_dq,dMdq,dfdu] = linearize_MF(Mr,ff,X_q,tau_v,X_q_d,tau_d);



write_diff = 1;
if write_diff == 1;
    write_file(dfdq,'dfdq_cartpole.m',vars_d); % Writing dFFdQ
    write_file(dfd_dq,'dfd_dq_cartpole.m',vars_d); % Writing dFFd_dQ
    write_file(dfdu,'dfdu_cartpole.m',vars_d); % Writing dFFdU
    write_file(dMdq,'dMdq_cartpole.m',vars_d); % Writing dMdQ

end

% Note the A_lin and B_lin matrices can be computed and written for
% calculations later, but in most cases, it is not possible to compute
% these matrices in symbolic form directly, especially in cases wehre the
% number of states is very high. So dMdq,dfdq,dfd_dq,dfdu,Mr_d,ff matrices
% are computed and saved as function, and linearized matrix is calculated
% on the fly according with the equation below. 



% Test if calculations are correct. 
% Only possible with small number of states. 
display('Testing dMidq, should retun 0s')

Mr_d = subs(Mr,[x th],[x_d th_d]);

Mr_d= simplify(simplify(Mr_d));
ff_d = simplify(subs(ff,[x th dx dth],[x_d th_d dx_d dth_d]));

[A_lin,B_lin] = get_linearized_system(dMdq,dfdq,dfd_dq,dfdu,Mr_d,ff_d)


Mi = inv(simplify(Mr));

for i = 1:length(X_q)/2
    dmidq(:,:,i) = subs(diff(Mi,X_q(i)),[x;th],[x_d;th_d]);
    
    dmidq_c(:,:,i) = - inv(Mr_d)*dMdq(:,:,i)*inv(Mr_d);
    
    simplify(dmidq(:,:,i)-dmidq_c(:,:,i)) ;% SHould return 0 if calculations are correct. 
end



if norm(simplify(dmidq(:,:,i)-dmidq_c(:,:,i)))==0
    display('Direct calculation and analytical results returned the same values')
    display('dMidq calculated correctly')
else
    display('Direct calculation and analytical results returned DIFFERENT values')
    display('CHECK CALCULATIONS of dMidq')
end





Miff = inv(simplify(Mr))*ff;
Miff = simplify(Miff);


display('Testing dMiffdq, should retun 0s')
for i = 1:length(X_q)/2
    
    dmiffdq(:,i) = subs(diff(Miff,X_q(i)),[x;th],[x_d;th_d]);
    
    Miff_col1 = dmidq(:,:,i)*subs(ff, [x;th],[x_d;th_d]);
    Miff_col2 = subs(Mi*diff(ff,X_q(i)), [x;th],[x_d;th_d]);
    
    dmiffdq_c(:,i) =Miff_col1 + Miff_col2 ;
end

if norm(simplify(dmiffdq_c-dmiffdq))==0
    display('Direct calculation and analytical results returned the same values')    
    display('dMiffdq calculated correctly')
else
    display('Direct calculation and analytical results returned DIFFERENT values')
    display('CHECK CALCULATIONS of dMiffdq')
end



display('Testing A_lin and B_lin expressions, should retun 0s')

dX_q = A_lin*X_q;
dX_q(1) - X_q(length(X_q)/2+1,:)
dX_q(2) - X_q(length(X_q)/2+2,:)

display('Errors from matrix calculations')

err_dAdx = A_lin(3:4,1) - subs(dmiffdq_c(:,1),[x th dx dth],[x_d th_d dx_d dth_d])
err_dAdth = simplify(A_lin(3:4,2) - subs(dmiffdq_c(:,2) ,[x th dx dth],[x_d th_d dx_d dth_d]))
err_dAd_dx = simplify(A_lin(3:4,3) - subs(inv(Mr)*dfd_dq(:,1) ,[x th dx dth],[x_d th_d dx_d dth_d]))
err_dAd_dth = simplify(A_lin(3:4,4) - subs(inv(Mr)*dfd_dq(:,2) ,[x th dx dth],[x_d th_d dx_d dth_d]))

err_dBdq = simplify(B_lin(3:4,:) - subs(inv(Mr)*dfdu,[x th dx dth],[x_d th_d dx_d dth_d]))


display('Errors from directly computing derivatives')

e1 = simplify(A_lin(3:4,1) - subs(diff(Miff,X_q(1)),[x;th;dx;dth],[x_d;th_d;dx_d;dth_d]))
e2 = simplify(A_lin(3:4,2) - subs(diff(Miff,X_q(2)),[x;th;dx;dth],[x_d;th_d;dx_d;dth_d]))
e3 = simplify(A_lin(3:4,3) - subs(diff(Miff,X_q(3)),[x;th;dx;dth],[x_d;th_d;dx_d;dth_d]))
e4 = simplify(A_lin(3:4,4) - subs(diff(Miff,X_q(4)),[x;th;dx;dth],[x_d;th_d;dx_d;dth_d]))


for i = 1:length(X_q)/2
    
    dmiffdq(:,i) = subs(diff(Miff,X_q(i)),[x;th],[x_d;th_d]);
    
    Miff_col1 = dmidq(:,:,i)*subs(ff, [x;th],[x_d;th_d]);
    Miff_col2 = subs(Mi*diff(ff,X_q(i)), [x;th],[x_d;th_d]);
    
    dmiffdq_c(:,i) =Miff_col1 + Miff_col2 ;
end
