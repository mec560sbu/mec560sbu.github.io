function diff_p_q_res = get_vel(p,q,dq)


for i = 1:length(q)
    diff_p_q(:,i) = diff(p,q(i))*dq(i); 
end

diff_p_q_res = simplify(expand(sum(diff_p_q,2)));