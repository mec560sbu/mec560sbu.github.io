function C = D2C_body(D,q,dq,mat)

n = size(D,1);
C = sym(zeros(n,n));
ind = find(mat == 1);

% for k = 1:n,
%     for j = 1:n,        
%         for i = 1:n,
%             if i==1,
%                 
%                 C(k,j)=1/2*(diff(D(k,j),q(i))+diff(D(k,i),q(j))-diff(D(i,j),q(k)))*dq(i);
%             else
%                 C(k,j)=C(k,j)+1/2*(diff(D(k,j),q(i))+diff(D(k,i),q(j))-diff(D(i,j),q(k)))*dq(i);
%             end
%         end
%     end
%     k
% end



for k = ind,
    for j = ind,        
        for i = ind,
            if i==1,
                
                C(k,j)=1/2*(diff(D(k,j),q(i))+diff(D(k,i),q(j))-diff(D(i,j),q(k)))*dq(i);
            else
                C(k,j)=C(k,j)+1/2*(diff(D(k,j),q(i))+diff(D(k,i),q(j))-diff(D(i,j),q(k)))*dq(i);
            end
            [i j k]
        end
    end
    k
end
