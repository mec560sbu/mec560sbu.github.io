clc
close all
clear all

addpath Screws;
addpath fcn_support;
addpath fcn_models;


define_syms;

get_kinematics_cart;

get_EOMs_cart;

linearize_EOM_cart;