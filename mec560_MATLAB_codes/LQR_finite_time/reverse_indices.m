function P_rev = reverse_indices(P)

 for  i = 1:size(P,1)
     P_rev(i,:) = P(size(P,1)-i+1,:);
 end