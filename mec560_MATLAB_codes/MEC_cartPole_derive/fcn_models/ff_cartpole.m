function p = ff_cartpole(x,th,dx,dth,m_cart,m_mass,l,u_th,g)
%FF_CARTPOLE
%    P = FF_CARTPOLE(X,TH,DX,DTH,M_CART,M_MASS,L,U_TH,G)

%    This function was generated by the Symbolic Math Toolbox version 7.0.
%    31-Oct-2016 00:31:28

p = [-dth.^2.*l.*m_mass.*cos(th);u_th];
