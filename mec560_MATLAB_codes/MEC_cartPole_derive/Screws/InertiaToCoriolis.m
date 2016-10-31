function Cee = InertiaToCoriolis(M, q, dq)
if ~isvector(q) || size(q, 2) ~= 1
    error('2nd input should be a column vector');
end
no_of_joints = length(q);
if ~isvector(dq) || size(dq, 1) ~= no_of_joints
    error('3rd input should have same shape as 2nd input');
end
if size(M,1) ~= no_of_joints || size(M,2) ~= no_of_joints
	error('1st input is not a valid mass-inertia matrix');
end

Cee = vpa(zeros(no_of_joints,no_of_joints));
for k = 1:no_of_joints
    for j = 1:no_of_joints
        for i = 1:no_of_joints
            Cee(k,j) = Cee(k,j)+...
                1/2*(diff(M(k,j),q(i))+...
                diff(M(k,i),q(j))-...
                diff(M(i,j),q(k)))*dq(i);
        end
    end
end
Cee = simple(Cee);
