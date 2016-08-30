function [dfdq_partial_subs,dfdtau_partial_subs] = linearize_DCG(D,C,G,x,tau_v,x0,ddq0)



N_dim = length(x)/2;

f_xu(1:N_dim,:) = x(N_dim+1:2*N_dim,:);
f_xu_ddq = inv(D) * ( tau_v - C*x(N_dim+1:2*N_dim,:)-G);
f_xu(N_dim+1:2*N_dim,:) = f_xu_ddq;

 D0 = subs(D,x,x0);
 C0 = subs(C,x,x0);
 G0 = subs(G,x,x0);

 tau0 =  D0*ddq0+C0*x0(N_dim+1:2*N_dim) + G0;
for i = 1:2*N_dim
    dfdq_partial(:,i) = diff(f_xu,x(i)) ;
end


dfdq_partial_subs = subs(dfdq_partial, [x;tau_v], [x0;tau0]) ;


for i = 1:length(tau_v)
    dfdtau_partial(:,i) = diff(f_xu,tau_v(i)) ;
end

dfdtau_partial_subs = subs(dfdtau_partial, [x;tau_v], [x0;tau0]) ;
