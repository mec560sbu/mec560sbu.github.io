function v_bar = VectorToHomogeneous(v)
if ~isvector(v) || size(v,1) ~= 3
    error('Input should be a 3 x 1 vector');
end
v_bar = [v;0];
