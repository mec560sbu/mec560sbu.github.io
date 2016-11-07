function yes_no = isZero(M)
n = size(M,1); 
m = size(M,2);
for i = 1:n
    for j = 1:m
        if M(i,j) ~= 0
            yes_no = false;
            return;
        end
    end
end
yes_no = true;
