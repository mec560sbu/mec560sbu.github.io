function [A_lin,B_lin] = get_linearized_system(dMdq,dfdq,dfd_dq,dfdu,M_d,ff_d)


N_states = size(M_d,1);
N_controls = size(dfdu,2);


Mi = inv(M_d);
for i = 1:N_states
    dmidq_c = - inv(M_d)*dMdq(:,:,i)*inv(M_d);
    Miff_p1 = dmidq_c*ff_d; % Dmi/dq*ff term
    Miff_p2 = Mi*dfdq(:,i); % Mi*df/dq term
    
    dmiffdq_c(:,i) =Miff_p1 + Miff_p2 ; 
end

A_lin = [ zeros(N_states,N_states)  eye(N_states);
                dmiffdq_c Mi*dfd_dq];
            
B_lin = [ zeros(N_states,N_controls);
                Mi*dfdu]; 