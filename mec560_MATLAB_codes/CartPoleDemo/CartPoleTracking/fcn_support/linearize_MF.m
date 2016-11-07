function [dfdq,dfd_dq,dMdq,dfdu] = linearize(M,ff,x,u,x_d,u_d)

n_states = length(x);
n_controls = length(u);

for i_s = 1:n_states/2,
    dfdq(:,i_s) = subs(diff(ff,x(i_s)) , [x;u],[x_d;u_d]);
    
end

for i_s = n_states/2+1:n_states,
    x(i_s)
    dfd_dq(:,i_s-n_states/2) = subs(diff(ff,x(i_s)) , [x;u],[x_d;u_d]);
    
end


for i_s = 1:n_states/2,
    dMdq(:,:,i_s) = subs(diff(M,x(i_s)) , [x],[x_d]);
end
for i_u = 1:n_controls,
    dfdu(:,i_u) =subs(diff(ff,u(i_u)) , [x;u],[x_d;u_d]);
end