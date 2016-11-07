q_v = [x;
    th;];
dq_v = [dx;
    dth;];

dq = get_vel(q_v,q_v,dq_v);


% Velocity of the cart
v_cart = get_vel(P_cart,q_v,dq_v);
v_cart = simplify(v_cart);

KE_cart = 1/2*m_cart*v_cart.'*v_cart;
PE_cart =  m_cart*g*P_cart(2);

% use .' because ' results in conjugate transpose.

% Velocity of the mass
v_mass = get_vel(P_mass,q_v,dq_v);
v_mass = simplify(v_mass);

KE_mass = 1/2*m_mass*v_mass.'*v_mass; % No rotational term as its point mass
PE_mass = m_mass*g*P_mass(2);

% Energetics

KE_total = KE_mass + KE_cart;
PE_total = PE_mass + PE_cart;

[Mr,Cr,Gr] = get_mat(KE_total,PE_total,q_v,dq_v);
U = [0;u_th];

ff = U - Cr*dq_v - Gr;



write_kin = 1;
if write_kin == 1;
    write_file(ff,'ff_cartpole.m',vars); % Writing FF
    write_file(Mr,'M_cartpole.m',vars); % Writing M
    
    write_file(KE_total,'KE_cartpolel.m',vars); % Writing KE
    write_file(PE_total,'PE_cartpole.m',vars); % Writing PE
end


