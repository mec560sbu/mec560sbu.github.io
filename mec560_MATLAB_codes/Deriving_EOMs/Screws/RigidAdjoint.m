function Ad_g = RigidAdjoint(g)
R = RigidOrientation(g);
p = RigidPosition(g);
Ad_g = [R AxisToSkew(p)*R; zeros(3,3) R];
