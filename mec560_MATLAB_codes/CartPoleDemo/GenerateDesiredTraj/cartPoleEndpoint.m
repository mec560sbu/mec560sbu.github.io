function output = cartPoleEndpoint(input)
% input.auxdata.dynamics;


q = input.phase.integral;
tf = input.phase.finaltime;

gamma = input.auxdata.gamma;


output.objective = tf+gamma*q;