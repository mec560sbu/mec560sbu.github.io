function theta = RotationAngle(R)
theta = acos((trace(R)-1)/2);
