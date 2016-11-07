clc
close all
clear all

addpath Screws;
addpath fcn_support;
addpath fcn_models;


display('Defining symbols')
define_syms;

display('Deriving kinematics')
get_kinematics_cart;

display('Deriving EOMs')
get_EOMs_cart;

display('Linearizing EOMs')
linearize_EOM_cart;