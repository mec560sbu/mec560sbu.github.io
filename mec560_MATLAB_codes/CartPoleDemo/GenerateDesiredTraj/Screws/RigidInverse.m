function g_inv = RigidInverse(g)
R = RigidOrientation(g);
p = RigidPosition(g);
g_inv = [R.' -R.'*p; 0 0 0 1];
