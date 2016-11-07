function MM = inertia_mat(T,dq)
for i = 1:length(dq),
    for j = 1:length(dq),
        mm = diff(T,dq(i));
        MM(i,j)=diff(mm,dq(j));
    end
end