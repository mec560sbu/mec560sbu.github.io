function G = Gvector(V,q)
for i = 1:length(q),
    G(i,1)=diff(V,q(i));
end