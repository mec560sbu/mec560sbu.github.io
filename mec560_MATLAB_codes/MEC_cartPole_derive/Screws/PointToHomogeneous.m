function q_bar = PointToHomogeneous(q)
if ~isvector(q) || size(q,1) ~= 3
    error('Input should be a 3 x 1 vector');
end
q_bar = [q;1];
