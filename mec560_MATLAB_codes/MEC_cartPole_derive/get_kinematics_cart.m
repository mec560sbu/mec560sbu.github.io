% Position of mass 1

q = [x;th];
dq = [dx;dth];


% Position of cart
P_cart = [ x;0];

% Position of mass 2
P_mass = [ x-l*cos(th);
    l*sin(th)];

write_kin = 1;
if write_kin == 1;
    write_file(P_cart,'P_cart.m',vars); % Writing KE
    write_file(P_mass,'P_mass.m',vars); % Writing PE
end

