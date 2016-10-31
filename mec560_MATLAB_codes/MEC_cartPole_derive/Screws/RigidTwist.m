function zeta = RigidTwist(g)
R = RigidOrientation(g);
p = RigidPosition(g);
w = RotationAxis(R);
if isZero(w)
    theta = sqrt(p.'*p);
    v = p/theta;
else
    w = w/sqrt(w.'*w);
    theta = RotationAngle(R);
	v = ((eye(3) - R)*AxisToSkew(w) + w*(w.')*theta)\p;
end
zeta = [v; w];
