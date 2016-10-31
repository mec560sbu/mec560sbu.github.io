function Ip = sp_int(pp,x0),

a = pp.coefs;
ord = pp.order;
tb = pp.breaks;
[m,n] = size(a);

for i = 1:n,
    Ia(:,i) = a(:,i)/(n-i+1);
end

p_0 = x0;
for i = 1:m-1,
    s = x0;
    for j = n:-1:1,
        s = s + Ia(i,n-j+1)*(tb(i+1)-tb(i))^j;
    end
    x0 = s;
    p_0 = [p_0 ; x0];
end

Ia = [Ia p_0];
Ip = mkpp(tb,Ia);