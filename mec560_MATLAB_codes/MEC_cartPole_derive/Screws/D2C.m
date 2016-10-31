function C = D2C(D,q,dq)

n = size(D,1);
C = sym(zeros(n,n));
for k = 1:n,
    for j = 1:n,        
        for i = 1:n,
            if i==1,
                C(k,j)=1/2*(diff(D(k,j),q(i))+diff(D(k,i),q(j))-diff(D(i,j),q(k)))*dq(i);
            else
                C(k,j)=C(k,j)+1/2*(diff(D(k,j),q(i))+diff(D(k,i),q(j))-diff(D(i,j),q(k)))*dq(i);
            end
            [k j i];
        end
    end
end
