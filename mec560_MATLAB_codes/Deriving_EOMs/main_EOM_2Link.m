clc
close all
clear all

addpath Screws
addpath fcn_support
% Defining symbols
syms m1 m2 l1 l2 q1 q2 dq1 dq2 ddq1 ddq2 tau1 tau2 g real
syms q10  q20

% Position vectors



P1 = [  l1 * cos(q1);
    l1 * sin(q1)];

P2 = [  l1 * cos(q1) + l2 * cos(q1+q2);
    l1 * sin(q1)+ l2 * sin(q1+q2)];

q_v = [q1;q2];
dq_v = [dq1;dq2];
% 

% Taking derivative to compute velocities
V1 = get_vel(P1 ,q_v,dq_v);
V2 =get_vel(P2,q_v,dq_v);

% Computing Kinetic energy and potential energy
KE1 =simplify(1/2*m1*V1'*V1);
KE2 =simplify(1/2*m2*V2'*V2);

PE1 = m1*g*P1(2);
PE2 = m2*g*P2(2);

% Define Lagrangian
KE_total = KE1 + KE2;
PE_total = PE1 + PE2;


[D,C,G] = get_mat(KE_total, PE_total, q_v,dq_v);
D = simplify(D);
C = simplify(C);
G = simplify(G);


% Now express this in the form of dx/dt = f(x,u)
x = [q1;q2;dq1;dq2]; % Vector of state space
ddq0 = [0;0]; % Vector of SS joint accelerations
x0 = [q10;q20;0;0]; % Vector of SS joint angles and velocites
tau_v = [tau1;tau2]; % Vector of torques
% Function to calculate Linearized representation
[A_lin,B_lin] = linearize_DCG(D,C,G,x,tau_v,x0,ddq0);
A_lin = simplify(A_lin)
B_lin = simplify(B_lin)


