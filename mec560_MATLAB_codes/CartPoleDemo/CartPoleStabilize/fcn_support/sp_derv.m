function Dp = sp_derv(pp,n),
ord = pp.order;
xb =  pp.breaks;
a =  pp.coefs;
k = ord-n;

m = [];

for i = 1:k,
    mm = 1;
    for j = 1:n,
        mm = mm*(ord - i + 1 -j);
    end    
    m = [m mm];      
    da(:,i) = mm*a(:,i);
end

Dp = mkpp(xb,da);