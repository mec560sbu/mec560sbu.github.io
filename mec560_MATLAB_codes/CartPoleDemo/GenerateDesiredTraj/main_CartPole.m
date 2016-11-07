%-------------------------- Cart-pole Problem -----------------------%
%--------------------------------------------------------------------%
clear all
close all
clc

addpath fcn_models

x0 = 0;
xf = 0;
th0 = pi;
thf = 0;
dx0 = 0;
dxf = 0;
dth0 = 0;

dthf = 0;

% Auxillary data:
%-------------------------------------------------------------------%
%-------------------- Data Required by Problem ---------------------%
%-------------------------------------------------------------------%
% Some model related constants
gamma = .0001;
auxdata.gamma = gamma ;


% Parameters:
%-------------------------------------------------------------------%
%-------------------- Data Required by Problem ---------------------%
%-------------------------------------------------------------------%
% Some model related constants
ndof     = 2;
nstates  = 2*ndof;
joints   = {'X' 'Theta'};
njoints  = size(joints,2);

dofnames = {'x','Theta'};
N  = nstates;        % number of state variables in model
Nu = 1;         % number of controls

%-------------------------------------------------------------------%
%----------------------------- Bounds ------------------------------%
%-------------------------------------------------------------------%

%-------------------------------------------------------------------%
t0  = 0;
tf  = 5;

xMin = [-10 -10 -100 -100];       %minimum of coordinates
xMax = [10 10 100 100];       %maximum of coordinates

uMin = [-1000]; %minimum of torques
uMax = [1000]; %maximum of torques

% setting up bounds
bounds.phase.initialtime.lower  = 0;
bounds.phase.initialtime.upper  = 0;
bounds.phase.finaltime.lower    = 0.05; 
bounds.phase.finaltime.upper    = tf;
bounds.phase.initialstate.lower = [x0 th0 dx0 dth0];
bounds.phase.initialstate.upper = [x0 th0 dx0 dth0];
bounds.phase.state.lower        = xMin;
bounds.phase.state.upper        = xMax;
bounds.phase.finalstate.lower   = [xf thf dxf dthf];
bounds.phase.finalstate.upper   = [xf thf dxf dthf];
bounds.phase.control.lower      = uMin; 
bounds.phase.control.upper      = uMax; 
bounds.phase.integral.lower     = 0;
bounds.phase.integral.upper     = 10000;



%-------------------------------------------------------------------%
%--------------------------- Initial Guess -------------------------%
%-------------------------------------------------------------------%
rng(0);

xGuess = [x0;xf]; 
thGuess = [th0;thf]; 
dxGuess = [dx0;dxf]; 
dthGuess = [dth0;dthf];
uGuess = [0;0];
tGuess = [0;tf]; 

guess.phase.time  = tGuess;
guess.phase.state = [xGuess,thGuess,dxGuess,dthGuess];
guess.phase.control        = uGuess;
guess.phase.integral         = 0;

% 
% load solution.mat
% guess  = solution

%-------------------------------------------------------------------%
%--------------------------- Problem Setup -------------------------%
%-------------------------------------------------------------------%
setup.name                        = 'cartpole-Problem';
setup.functions.continuous        = @cartPoleContinuous;
setup.functions.endpoint          = @cartPoleEndpoint;
setup.bounds                      = bounds;
setup.auxdata                     = auxdata;
setup.functions.report            = @report;
setup.guess                       = guess;
setup.nlp.solver                  = 'ipopt';
setup.derivatives.supplier        = 'sparseCD';
setup.derivatives.derivativelevel = 'first';
setup.scales.method               = 'none';
setup.derivatives.dependencies    = 'full';
setup.mesh.method                 = 'hp-PattersonRao';
setup.mesh.tolerance              = 1e-1;
setup.method                      = 'RPM-Integration';

%-------------------------------------------------------------------%
%------------------- Solve Problem Using GPOPS2 --------------------%
%-------------------------------------------------------------------%
output = gpops2(setup);
output.result.nlptime
solution = output.result.solution;

%-------------------------------------------------------------------%
%--------------------------- Plot Solution -------------------------%
%-------------------------------------------------------------------%









plotCartPole













