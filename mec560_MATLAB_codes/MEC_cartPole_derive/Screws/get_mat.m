function [M,C,G]=get_mat(T,V,q,dq)

M = inertia_mat(T,dq);

C = D2C(M,q,dq);
G = Gvector(V,q);