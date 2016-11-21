---
layout: post
comments: true
title:  "Linearization: Nonlinear Dynamic Inversion, (input-output) Feedback Linearization and input-state linearization (SISO only)"
excerpt: "Linearization: Nonlinear Dynamic Inversion, (input-output) Feedback Linearization and input-state linearization (SISO only)."
date:   2016-10-01 11:00:00
mathjax: true
---


### Vivek Yadav, PhD

### Motivation


For many systems, the task of designing control can be simplified by using a simpler more intuitive coordinate representation. For example, consider the system given by, 

$$ \left[ \begin{array}{c} \dot{X_1} \\ \dot{X_2} \\ \dot{X_3} \end{array} \right] = \left[ \begin{array}{ccc} 0 & 1 & 0 \\ 0 & 0 & 1 \\ -2 & -3 & 3 \end{array} \right] \left[ \begin{array}{c} X_1 \\ X_2 \\ X_3 \end{array} \right] + \left[ \begin{array}{c} 0 \\ 0 \\ 1 \end{array} \right] u $$ 

and another system given by, 

$$ \left[ \begin{array}{c} \dot{X_1} \\ \dot{X_2} \\ \dot{X_3} \end{array} \right] = \left[ \begin{array}{ccc} 0 & 1 & 0 \\ 0 & 1 & 1 \\ -1 & -3 & 3 \end{array} \right] \left[ \begin{array}{c} X_1 \\ X_2 \\ X_3 \end{array} \right] + \left[ \begin{array}{c} 0 \\ 0 \\ 1 \end{array} \right] u $$ 

The two systems are similar, however, the first form is more useful for control, because the terms in the last row of state matrix represent the coefficients of characteristic polynomial. This idea in nonlinear system becomes more complex because the state transition matrix itself depends on several variables. The techniques of nonlinear dynamic inversion and feedback linearization allow us to unwrap a complex nonlinear system into a simpler linear system. The main idea is to perform a variable transformation that gives a more managable system. The conecepts of dynamic inversion are presented first. 


### Basics of dynamic inversion (or input-state feedback linearization): 

Say we have a simple single input, single output system given by 

$$ \dot{X} = f(X) + g(X) u $$


$$ \left[ \begin{array}{c} \dot{Z}_1 \\ \dot{Z}_2 \\ \vdots \\ \dot{Z}_n \end{array}  \right]  = \left[ \begin{array}{c} Z_2 \\ Z_3 \\ \vdots \\ b(Z) \end{array}  \right]  + \left[ \begin{array}{c} 0 \\ 0 \\ \vdots \\ a(Z)  \end{array}  \right] u $$

The system above can be linearized by using an auxillary control input \\( v \\) as, 

$$ v = b(Z) + a(Z) u \implies u = a(Z)^{-1}(v - b(Z))$$

The system dynamics after subtituting for \\(u \\) becomes, 

$$ \left[ \begin{array}{c} \dot{Z}_1 \\ \dot{Z}_2 \\ \vdots \\ \dot{Z}_n \end{array}  \right]  = \left[ \begin{array}{c} Z_2 \\ Z_3 \\ \vdots \\ 0 \end{array}  \right]  + \left[ \begin{array}{c} 0 \\ 0 \\ \vdots \\ 1 \end{array}  \right] v $$

Or the system reduces to, 

$$ \frac{d^n Z}{dt^n} = v $$

By choosing 

$$ v = -k_0Z -k_1Z_1 \dots -k_{n-1} Z_{n-1} $$

The system reduces to, 

$$ \frac{d^n Z}{dt^n} = -k_0Z -k_1 Z_1 \dots -k_{n-1} Z_{n-1}  $$

or 

$$ \frac{d^n Z}{dt^n} + k_{n-1} \frac{d^{n-1} Z}{dt^{n-1}} \dots k_1 \frac{dZ}{dt} + k_0 Z = 0  $$

Therefore, by choosing gains \\( k\\), the system can be made to follow an n-order linear dynamic system. For tracking task, we derive error dynamics and follow the same approach, as above with X replaced by the tracking error \\( e = X - X_d \\). The process of finding \\(v \\) and \\( u \\) are referred as outer loop and inner loop control also. The outer loop control stabilizes a virtual first order system, whereas the inner loop control \\(u\\) makes sure that the system's control law follows commanded control \\( v\\). 



### Example: 


Consider the example of the system given by, 

$$ \dot{X}_1  = -2 X_1 + a X_2 + sin(X_1) $$ 

$$ \dot{X}_2  = - X_2 cos(X_1) + cos(2 X_1) u $$

With transformation, 

$$ z_1 = X_1 $$ 

$$ z_2 = sin(X_1) + aX_2 $$

The states of the system becomes, 

$$ \dot{z}_1  = -2 z_1 + z_2 $$ 


$$ \dot{z}_2 = cos(z_1) \dot{z_1} + a \dot{X}_2 = cos(z_1) (-2 z_1 + z_2)  - aX_2 cos(X_1) + acos(2 X_1) u  $$

$$ \dot{z}_2 = cos(z_1) (-2 z_1 + z_2)  + (sin(z_1)-z_2) cos(z_1) + a cos(2 z_1) u  $$

$$ \dot{z}_2 = -2 z_1  cos(z_1)   + sin(z_1)cos(z_1) + a cos(2 z_1) u  $$

By defining, 

$$ v =  -2 z_1  cos(z_1)   + sin(z_1)cos(z_1) + a cos(2 z_1) u  $$ 

The system reduces to, 

$$ \dot{z}_1  = -2 z_1 + z_2 $$ 

$$ \dot{z}_2 = v $$

We can design \\( v \\) to drive system states \\( Z \\) to any desired value. Note \\( v \\) is not the true control, the original control \\( u \\) is computed as, 

$$ u = \frac{1}{a cos(2 z_1)} (v + 2 z_1  cos(z_1)  - sin(z_1)cos(z_1) ) $$ 


Note, in the expression above, we are dividing by \\( cos(2 z_1 ) \\), and it is possible that the control \\( u \\) goes to infinity. 


#### Considerations for nonlinear dynamic inversion, 

1. Full state measurement is required to invert the control \\( u = a(X)^{-1}(v-b(X)) \\).
2. The matrix \\( a(X) \\) must be invertible. 
3. \\( a(X) \\) may got zero for some values of states, and these singularities can result in \\( u \rightarrow \infty \\). 

Note, by following the nonlinear dynamic inversion scheme above, it appears that we are completely cancelling out the effect of nonlinearities. Any dynamic advantage afforded by nonlinearities is captured by the desired trajectory or reference trajectory, that is computed using optimization routines. 

### Input-output linearization (working principle): 

In some cases it may be difficult to obtain a simple expression as above, i.e. the system dynamics need not be in companion form. Further, in many applications we are more interested in making the output of the dynamics follow a particular dynamics. In these cases, it is helpful to linearize the output of the system. Therefore, the linearized system with inner loop control \\( u = a(X)^{-1}(u - b(X)) \\) becomes, \\( y^{(r)} = v \\).


#### Example 1: 

Consider the system given by

$$ \dot{X}_1 = X_2^2 + sin(X_3) $$

$$ \dot{X}_2 = cos(X_2) $$

$$ \dot{X}_3 = X_1 + u $$

with measurement \\( y = sin(X_1) \\).

We start with, 

$$ y = sin(X_1) $$

As there is no control input \\( u \\), we compute derivative of the measurement, 

$$ \dot{y} = cos(X_1)  \dot{X}_1 = cos(X_1) X_2^2 + cos(X_1) sin(X_3) $$

As there is no control input \\( u \\) in \\( \dot{y} \\), we compute derivative of the measurement, 


$$ \ddot{y} = -sin(X_1) X_2^2 \dot{X}_1 + 2 cos(X_1) X_2 \dot{X}_2  -sin (X_1) sin(X_3) \dot{X}_1 +cos(X_1) cos(X_3) \dot{X}_3  $$

$$ \ddot{y} = -sin(X_1) X_2^2 (X_2^2 + sin(X_3) )+ 2 cos(X_1) X_2 cos(X_2)   -sin (X_1) sin(X_3) (X_1^2+sin(X_3)) +cos(X_1) cos(X_3) (X_1 + u )  $$


$$ \ddot{y} = \underbrace{2 cos(X_1)^2 X_2 + X_1 cos(X_1) cos(X_3) - (X_2^2 + sin(X_3))^2 sin(X_1)}_{a(X)} + \underbrace{cos(X_1) cos(X_3)}_{b(X)} u $$






#### Example 2:

$$ \dot{X}_1 = sin(X_2) + (X_2+1)X_3 $$

$$ \dot{X}_2 = X_1^5 + X_3 $$

$$ \dot{X}_3 = X_1^2 + u $$


Say, we are interested in the measurement \\( X_1 \\).

$$ Z_1 = X_1 $$ 

There is no control input in this term, so we differentiate and get, 

$$ Z_2 = \dot{Z}_1 = \dot{X}_1 =  sin(X_2) + (X_2+1)X_3  $$ 

There is no control input in this term, so we differentiate again and get, 

$$ Z_3 = \dot{Z}_2 =  cos(X_2) \dot{X_2} + (X_2+1)\dot{X}_3 + X_3\dot{X}_2 = cos(X_2) (X_1^5 + X_3)+ (X_2+1)( X_1^2 + u ) + X_3 (X_1^5 + X_3)   $$ 


$$ Z_3 = \ddot{Z}_1 = (X_3 + cos(X_2)) (X_1^5 + X_3)+ (X_2+1)X_1^2 +  (X_2+1)u   $$ 

Therefore, choosing \\( v  =  (X_3 + cos(X_2)) (X_1^5 + X_3)+ (X_2+1)X_1^2 +  (X_2+1)u  \\), we get, 

$$ \ddot{Z}_1 = v $$

The transformed system is a second order linear system, and control input \\( v \\) can be chosen to have any second order behavior of \\( Z_1 \\). 


*** NOTE: In both the examples above, we started with 3 state system, and got a linearized 2-state system of measurement. Therefore, there is another degree of freedom of dynamics that is not represented by this input-output linearization. ***


### Zero dynamics

Zero dynamics are the dynamics of system when the output is zero. In most cases, we try to drive the measurement output (or error) to zero, and therefore the residual dynamics is expected to be similar to zero dynamics. For linear systems, we can show that if the zero dynamics is stable, then the residual dynamics is also stable. However, for nonlinear systems, this need not be true. Regardless, we can say that if the zero dynamics is stable (via Lyapunov's first method), the dynamics of the residual system will be stable, atleast locally. We will first investigate zero dynamics for a simple linear system. 


### Zero dynamics for linear systems

Consider a system in canonical form, 


$$ \dot{X} = \left[ \begin{array}{cccc} 0 & 1 & 0 & \dots \\ 0 & 0 & 1 & \dots  \\ \vdots &  \vdots &  \vdots &  \vdots \\ - a_0 & -a_1 & \dots & -a_{n-1} \end{array} \right] X + \left[ \begin{array}{c}  0 \\ 0 \\ \vdots  \\ 1 \end{array}\right] u $$ 

with a single measurement

$$ y =   \left[ \begin{array}{cccccccc} b_0 & b_1 & b_2 & \dots & b_m & (n-m  -1)~zeros \end{array} \right] X $$ 

The measurement equation has first \\( m \\) terms, and remaing \\( n-m -1 \\) terms are zero. We first rewrite measurement equation as, 

$$ y =   \left[ \begin{array}{cccccccc} b_0 & b_1 & b_2 & \dots & b_m & 0 & \dots & 0 \end{array} \right] X = b_0 X_1 + b_1 X_2 + \dots + b_m X_{m+1} $$ 

By taking derivative of measurement \\( y \\), we get, 

$$ \dot{y} = b_0 \dot{X}_1 + b_1 \dot{X}_2 + \dots + b_m \dot{X}_{m+1} = \left[ \begin{array}{cccccccc} 0  & b_0 & b_1 & b_2 & \dots & b_m & (n-m-2)~zeros\end{array} \right] X  $$ 

Therefore, by taking derivative, the b-values shifted one to right. Similarly, it can be shown that by taking \\( n-m-1\\) derivatives, we get

$$ y^{(n-m-1)} = \left[ \begin{array}{cccccccc} (n-m)~zeros  & b_0 & b_1 & b_2 & \dots & b_m \end{array} \right] X  $$ 

Taking derivative once again gives, 

$$ y^{(n-m)} = \left[ \begin{array}{cccccccc} (n-m+1)~zeros  & b_0 & b_1 & b_2 & \dots & b_{m-1} \end{array} \right] X   + b_m \dot{X_n} = u + additional~terms = v $$ 

Note, for this linear system, the number of poles was \\( n \\) and number of zeros are \\( m \\), and the relative degree \\( r \\) of the system is the difference of number of poles and zeros. 

By choosing \\( v \\) appropriately, we can design a stable \\( n-m \\) order close loop dynamics for \\( y \\). However, we need to investigate the dynamics of the residual \\( m \\) order system. This can be investigated by looking at the laplace transform of the measurement function. Note, as we are commanding \\(z\\) to follow a desired trajectory, \\( z \\) behaves as input to the resulting \\( m \\) order dynamics. Therefore, the transfer function between \\( x \\) and \\(z \\) can be written as,  

$$ Z(s) = (b_0  + b_1 s + \dots + b_m s^m) X(s) $$ 

$$ \frac{X(s)}{Z(s)} = \frac{1}{b_0  + b_1 s + \dots + b_m s^m}$$ 

Therefore, if the poles of this system are all on the negative half plane, then the residual dynamics is stable. In other words, if the zeros of the original system are on negative half of the plane, then the resulting zero-dynamics is stable. Systems with stable zero dynamics are referred to as minimum phase systems, and systems without stable zero dynamics are referred to as non-minimum phase systems. 


#### Linear system zero dynamics examples: 

Consider the system given by, 

##### System 1:
$$ \dot{X}_1 = X_2 + u $$ 

$$ \dot{X}_2 =  u $$ 

Say we are interested in controlling \\( X_1 \\). Then the zero dyanmics are given by, 

$$ z = X_1 = 0 $$

Taking derivative gives, 

$$ \dot{z} = \dot{X}_1 = X_2 + u = 0$$

Therefore, 

$$ u = - X_2 $$

Substituting this ins the last equation gives, 

$$ \dot{X}_2 =  u = - X_2 $$

Therefore, by choosing a reduced order system \\( [z, \dot{z}] \\), we get a stable overall performance because the zero dynamics is stable. 

##### System 2:
$$ \dot{X}_1 = X_2 + u $$ 

$$ \dot{X}_2 =  -u $$ 

By following the same steps above, we get, the zero dynamics as, 

$$ \dot{X}_2 =  X_2 $$

Therefore, in this case, as the residual dynamics is not stable, using \\( [z, \dot{z}] \\) for control results in unstable internal dynamics, inspite of the fact that \\( [z, \dot{z}] \\) system is stable. 


#### Zero dynamics nonlinear system:

$$ \dot{X}_1 = sin(X_2) + (X_2+1)X_3 $$

$$ \dot{X}_2 = X_1^5 + X_3 $$

$$ \dot{X}_3 = X_1^2 + u $$


Say, we are interested in the measurement \\( X_1 \\).

$$ Z_1 = X_1 $$ 

There is no control input in this term, so we differentiate and get, 

$$ Z_2 = \dot{Z}_1 = \dot{X}_1 =  sin(X_2) + (X_2+1)X_3  $$ 

There is no control input in this term, so we differentiate again and get, 

$$ Z_3 = \ddot{Z}_1 = (X_3 + cos(X_2)) (X_1^5 + X_3)+ (X_2+1)X_1^2 +  (X_2+1)u   $$ 

When  \\(Z_1= 0\\), we get, the zero dynamics as, 


$$ 0 = sin(X_2) + (X_2+1)X_3 $$

$$ \dot{X}_2 =  X_3 $$

$$ \dot{X}_3 = u $$

$$ 0 = (X_3 + cos(X_2)) X_3 +  (X_2+1)u   $$ 

After simplifications, the zero dynamics becomes, 


$$ \dot{X}_2 =  X_3 $$

$$ \dot{X}_3 = - \frac{(X_3 + cos(X_2)) X_3}{(X_2+1)} $$

The zero dynamics in this case is not stable, because if \\(X_3 << 0 \\), then \\( X_2 \rightarrow 0 \\) and \\(X_3 \\) will diverge to minus infinity. 









## Input-output linearization with Lie derivatives


We will now define general tools on computing the input-output linearization of the system. 

### Relative degree 

In the previous example, we had to differentiate the measurement output twice to get the input appear in the derivative of measurement. This is referred to as relative degree of the system. Therefore, by differentiating the measurement \\( r \\) times, where \\( r  \\) is the relative degree of the system, we get the control appear in the derivative.  


### Feedback linearization

Consider the system given by, 

$$ \dot{X} = f(X) + g(X) u $$

with measurements, 

$$ y = h(X) $$

We will now compute derivative of states with respect to time, 

$$ \dot{y} = \frac{d h}{d X} \dot{X} = \frac{d h}{d X} (f(X) + g(X) u) = \frac{d h}{d X} f(X) = L_f h(X) $$


Recall Lie derivative \\( L_f h(X) \\) is given by, 

$$  L_f h(X)  = \frac{\partial h}{ \partial X} f(X)$$

As relative degree is \\( r \\), the control term \\( u \\) will not appear in the first \\( r-1\\) derivatives. \\( L_gL_f^{(i)} h(X) = 0 \\) for \\( i<r \\)  becuase system has relative degree \\( r \\). Therefore, 

$$ y^{(r-1)} = L_f^{(r-1)} h(X)  $$


$$ y^{(r)} = \frac{ d L_f^{(r-1)} h(X)}{dX} \dot{X} =   \frac{ d L_f^{(r-1)} h(X)}{dX} f(X) +  \frac{ d L_f^{(r-1)} h(X)}{dX}  g(X) u  $$


$$ y^{(r)} =    L_f^{(r)} h(X)  +  L_g L_f^{(r-1)} h(X) u  $$

Therefore, choosing the inner loop control \\( u \\) as

$$  u =   L_g L_f^{r-1} h(X) ^{-1}  \left(   v -  L_f^{r} h(X)  \right) $$ 

After this transformation, we get, the original system changed as, 


$$ y^{(r)} =    L_f^{r} h(X)  +  L_g L_f^{r-1} h(X) u  $$


$$ y^{(r)} =   v  $$

Therefore, based on Lie derivatives, we can construct a representation of the system that is of order \\( r \\). However, the true system was of order \\( n \\), therefore, we need to account for additional \\( n-r \\) order dynamics to fully understand the behavior of the system. If the residual \\( n-r \\) dynamics is stable, then the whole system is stable, however, if the residual \\( n-r \\) dynamics is not stable, then the internal states of the system may diverge. One way to investigate the leftover dynamics is to investigate the zero-dynamics of the system. However, as these dynamics are very specific to nonlinear system being studied, it is very difficult to come up with general rules for zero dynamics. 

### Normal form

We saw above that by using Lie Algebra, we can express output of a n-dimension system

$$ \dot{X} = f(X) + g(X) u $$

with measurement \\( y \\) as, 

$$ y^{(r)} =   v  $$

Therefore, if we chose a variable transform corresponding to first \\( r-1 \\) derivatives of \\( y \\), we get a reduced representation of dynamics. However, there is a \\( n-r \\) order dynamics that is not accounted for. The system in normal form can be written in terms of states \\( [ \mu_1~\mu_2~\dots ~\mu_r]^T = [ y~ \dot{y}~\dots ~y^{(r)}]^T\\) and additional \\( n-r \\) variables \\( \psi \\) as, 

$$ \dot{\mu} = \left[ \begin{array}{c} \mu_2 \\ \mu_3 \\ \dots \\ \alpha(\mu,\psi) +  \beta(\mu,\psi) u \end{array} \right] $$

and 

$$ \dot{\psi} = w(\mu,\psi) $$

with output \\( y = \mu_1 \\). For the transformation between original states and \\( [ \mu, \psi] \\) to be valid, the transformation must be one-to-one and invertible (or Diffiomorphism). \\( \psi \\) can be found that satisfy, 

$$ \nabla \psi_i g = L_g \psi_i = 0 ~ for ~1\leq i \leq n-r $$


### Example: Normal form  


Consider the system, 

$$ \dot{X} = \left[ \begin{array}{c} -X_1 \\ 2X_1 X_2 + sin(X_2) \\ 2X_2  \end{array} \right] +  \left[ \begin{array}{c} e^{2X_2} \\ .5 \\ 0  \end{array} \right] u $$

with measurement \\( y = X_3 \\). Following the procedure above, we first define, 

$$ \mu_1 = y = X_3$$ 

$$ \mu_2 = \dot{\mu}_1 = \dot{y} = \dot{X}_3 = 2X_2 $$ 

$$  \dot{\mu}_2  = \ddot{y} = 2\dot{X}_2 = 2(2X_1 X_2 + sin(X_2) ) + u $$ 


We now need to find a third variable \\( \psi \\) such that \\( L_g \psi = 0\\)



$$  L_g \psi = \frac{\partial \psi }{\partial X} g = \left[\begin{array}{ccc} \frac{\partial \psi }{\partial X_1} & \frac{\partial \psi }{\partial X_2} & \frac{\partial \psi }{\partial X_3}    \end{array}\right] \left[ \begin{array}{c} e^{2X_2} \\ .5 \\ 0  \end{array} \right] = 0 $$



$$ e^{2X_2} \frac{\partial \psi }{\partial X_1} + .5 \frac{\partial \psi }{\partial X_2} = 0$$






A possible solution for equation above is, 

$$\psi = 1 + X_1 - e^{2X_2}  $$

Therefore, varaible transformation, 

$$ \mu_1 = X_3 $$

$$ \mu_2 = 2 X_2 $$

$$ \psi =  1 + X_1 - e^{2X_2}  $$

The states \\(X \\) can be obtained using, 

$$ X_3 = \mu_1  $$

$$  X_2 = \frac{\mu_2}{2} $$

$$  X_1 = \psi + e^{\mu_2}-1$$


Transforms the original system into, 

$$ \dot{\mu}_1 = \mu_2 $$

$$ \dot{\mu}_2 = 2( (\psi + e^{\mu_2}) \mu_2 + sin(\mu_2/2) ) + u $$

$$ \dot{\psi} = \dot{X}_1 - 2e^{2X_2} \dot{X}_2 = (1-\psi-e^{\mu_2})(1+2\mu_2e^{\mu_2} - 2 sin(\mu_2/2)e^{\mu_2}  $$

Note, the zero dynamics is, 

$$ \dot{\psi} = - \psi $$ 

which is stable. Therefore, the feedback linearization scheme can be used to follow any desired trajectory, and the underlying residual dynamics is expected to be stable. 

### Input-state linearization 

Now the main question becomes, given a general nonlinear system, can we find a variable transformation such that the resulting system is a \\( n \\) order system, and has a compact linear-like form, i.e. is it possible to obtain an input-state linearization that simplifies the nonlinear system into a much simpler system. Before, getting into specific details, lets recap lie brackets. 

#### Lie Brackets: Recap

Lie bracket between two vector fields is given by, 

$$ adj_f g = [f,g] (X) = \frac{\partial g}{\partial X} f(X) -  \frac{\partial f}{\partial X} g(X) $$

Lie brackets can be interpreted as a new direction that can be obtained by combining vector fields \\( f \\) and \\( g\\). 

##### Definition: Involutive

We call a group of vectors \\( g: [g_1, g_2, \dots , g_n] \\) involutive, if lie bracket between any two vector fields results in a linear combination of vectors in \\( g \\). i.e. dynamics along no new directions can be obtained by combining vector fields in \\( g \\). 

### Necessary condition for existance of simplified n-order system. 

Given a system, 

$$ \dot{X} = f(X) + g(X) u $$ 

we can get a transformation of variables from \\( X \\) to \\(Z\\) such that, \\( Z  = [ Z_1 ~ \dot{Z}_1~ \ddot{Z}_2 ~ \dots ~ Z_1^{(n-1)} ] \\) form the states of the system, and the dynamics are in canonical form, with 

$$ Z_1^{(n)} = \alpha(X) + \beta(X) u = v  $$

if the follwoing 2 conditions hold,  

1. (Controllability) The vectors \\(\left[ \begin{array}{cccccc} adj_f^0 g(X) & adj_f^1 g(X) & adj_f^2 g(X) & \dots &  adj_f^{n-2} g(X) &  adj_f^{n-1} g(X) \end{array} \right] \\) are all linearly independent or has full rank, and 
2. (Integrability/involutivity) \\(\left[ \begin{array}{cccccc} adj_f^0 g(X) & adj_f^1 g(X) & adj_f^2 g(X) & \dots &  adj_f^{n-2} g(X) &\end{array} \right] \\) is involutive. 

The second condition is called the integrability condition, because \\( z_1 \\) can be computed from the following relation, 

$$ \left[ \begin{array}{cccccc} adj_f^0 g(X) & adj_f^1 g(X) & adj_f^2 g(X) & \dots &  adj_f^{n-2} g(X) &  adj_f^{n-1} g(X) \end{array} \right] \nabla z_1 = \left[ \begin{array}{c} 0 \\ 0 \\ \vdots \\ 1  \end{array} \right]$$ 


where \\( \nabla z_1 \\) is the gradient of \\( z_1 \\). Therefore, the transformation of variables can be computed by integrating the gradient function. The second condition ensures that \\( \nabla z_1 \\)  is integrable.  More specifically, \\( \nabla z_1 \\) forms a gradient field. 

Note, the conditions above are very specific, and depend on the state variables, therefore, it is possible that in some region of space, a one to one transformation may not exist. However, if conditions 1 and 2 above are satisfied, then a linear transform of the system can be obtained by following these steps,

- Construct vector fields, \\( ad_f^ig \\) for i in \\(0\\) to \\( n - 1 \\). 
- Check if controllability and integrability conditions are satisfied. 
- If both conditions are satisfied, then compute the first \\( z_1 \\) such that, \\( \nabla z_1 ad_f^i g = 0 \\) for \\( i = 0 \\) to \\( n-2 \\), and \\( \nabla z_1 ad_f^{(n-1)} g \neq 0 \\). 
- Define \\( Z(X)  = \left[ \begin{array}{cccccc}  z_1 & L_f z_1 & L_f^2 z_1 & \dots &  L_f^{n-2} z_1 &  L_f^{n-1} z_1 \end{array} \right]  \\) and, 

$$ \alpha(X) = - \frac{L_f^nz_1}{L_gL_f^{n-1}z_1} $$ 

$$ \beta(X) =  \frac{1}{L_gL_f^{n-1}z_1} $$ 



### Example: Flexible robot joint

Consider the robot joint given by, 

$$ I \ddot{q}_1 + MgL sin(q_1) + K(q_1 - q_2 ) = 0$$ 

$$ J \ddot{q}_2 - K(q_1 - q_2 ) =  u $$ 

We wish to linearize this system using the input-control linear transform scheme presented before. 

### Solution: 

We first express the system in a state space form using states \\( X = \left[ q_1 ~ q_2 ~ \dot{q}_1 ~ \dot{q}_2  \right]^T \\). The state dynamics can be written as,


$$ \dot{X} = \left[ \begin{array}{c} X_3 \\ X_4 \\ -\frac{MgL}{I} sin(X_1) -\frac{k}{I} (X_1 - X_2) \\ \frac{K}{J}(X_1 - X_2) \\  \end{array} \right] + \left[ \begin{array}{c} 0 \\ 0 \\ 0 \\ \frac{1}{J} \\  \end{array} \right] u $$ 




The controllability matrix form, \\(\left[ \begin{array}{cccccc} adj_f^0 g(X) & adj_f^1 g(X) & adj_f^2 g(X) &  adj_f^{3} g(X) \end{array} \right] \\) using, 

$$ adj_f^0 g(X) g(X)  = \left[ \begin{array}{c} 0 \\ 0 \\ 0 \\ \frac{1}{J} \\  \end{array} \right]  $$ 

$$ adj_f g = [f,g] (X) = \frac{\partial g}{\partial X} f(X) -  \frac{\partial f}{\partial X} g(X) = 0 - \underbrace{ \left[ \begin{array}{c} 0 & 0 & 1 & 0 \\  0 & 0 & 0 & 1  \\ -\frac{k}{I}-\frac{MgL}{I}cos(X_1) & \frac{k}{I} & 0 & 0 \\ \frac{K}{J} & - \frac{K}{J} & 0 & 0 \end{array} \right]}_{\frac{\partial f}{\partial X}} \left[ \begin{array}{c} 0 \\ 0 \\ 0 \\  \frac{1}{J}   \end{array} \right] = \left[ \begin{array}{c} 0 \\ -\frac{1}{J} \\ 0 \\ 0   \end{array} \right] $$
 


$$ adj_f^2 g = [f,adj_f g] (X) = \frac{\partial adj_f g}{\partial X} f(X) -  \frac{\partial f}{\partial X} adj_f g (X) = 0 - \underbrace{ \left[ \begin{array}{c} 0 & 0 & 1 & 0 \\  0 & 0 & 0 & 1  \\ -\frac{k}{I}-\frac{MgL}{I}cos(X_1) & \frac{k}{I} & 0 & 0 \\ \frac{K}{J} & - \frac{K}{J} & 0 & 0 \end{array} \right]}_{\frac{\partial f}{\partial X}} \left[ \begin{array}{c} 0 \\ - \frac{1}{J} \\ 0 \\ 0   \end{array} \right]  = \left[ \begin{array}{c} 0 \\ 0 \\ - \frac{K}{JI} \\ - \frac{K}{J^2}   \end{array} \right] $$



$$ adj_f^3 g = [f,adj_f^2 g] (X) = \frac{\partial adj_f^2 g}{\partial X} f(X) -  \frac{\partial f}{\partial X} adj_f^2 g (X) = 0 - \underbrace{ \left[ \begin{array}{c} 0 & 0 & 1 & 0 \\  0 & 0 & 0 & 1  \\ -\frac{k}{I}-\frac{MgL}{I}cos(X_1) & \frac{k}{I} & 0 & 0 \\ \frac{K}{J} & - \frac{K}{J} & 0 & 0 \end{array} \right]}_{\frac{\partial f}{\partial X}} \left[ \begin{array}{c} 0 \\ 0 \\ - \frac{K}{JI} \\ - \frac{K}{J^2}   \end{array} \right]   = \left[ \begin{array}{c} - \frac{K}{JI} \\ - \frac{K}{J^2} \\ 0 \\ 0    \end{array} \right] $$


Therefore, the controllability matrix is, 

$$\left[ \begin{array}{cccccc} adj_f^0 g(X) & adj_f^1 g(X) & adj_f^2 g(X) &  adj_f^{3} g(X) \end{array} \right] = \left[ \begin{array}{c} 0 & 0 & 0 & - \frac{K}{JI} \\  0 & -\frac{1}{J}  & 0 & - \frac{K}{J^2}  \\ 0 & 0 &  - \frac{K}{JI} & 0 \\ \frac{1}{J} & 0 &  - \frac{K}{J^2} & 0 \end{array} \right] $$

which is a full-rank matrix. Further, as the controllability matrix is a constant, the controllability law applies for all states. Further, as all the adjoints are constants, their derivative is zero, which is a linear combination of them. Therefore, integrability/involutivility condition is also satisfied, therefore, the origrinal system is input-control linearizable. We can therefore move to the next step of designing state transformation. Recall, we need to find a \\(z_1 \\) such that, 


$$ \left[ \begin{array}{cccc} adj_f^0 g(X) & adj_f^1 g(X) & adj_f^2 g(X) & adj_f^{3} g(X)  \end{array} \right] \nabla z_1 = \left[ \begin{array}{c} 0 \\ 0 \\ 0 \\ 1  \end{array} \right]$$ 


$$ \left[ \begin{array}{c} 0 & 0 & 0 & - \frac{K}{JI} \\  0 & -\frac{1}{J}  & 0 & - \frac{K}{J^2}  \\ 0 & 0 &  - \frac{K}{JI} & 0 \\ \frac{1}{J} & 0 &  - \frac{K}{J^2} & 0 \end{array} \right]  \left[ \begin{array}{c} \frac{\partial z_1}{X_1} \\  \frac{\partial z_1}{X_2} \\  \frac{\partial z_1}{X_3} \\  \frac{\partial z_1}{X_4}  \end{array} \right] =  \left[ \begin{array}{c} 0 \\ 0 \\ 0 \\ 1  \end{array} \right]$$ 

The matrix equation above gives, 

$$ \begin{array}{cccc}  \frac{\partial z_1}{X_1} = 1 &  \frac{\partial z_1}{X_2} = 0 & \frac{\partial z_1}{X_3} = 0 & \frac{\partial z_1}{X_4} = 0 &   \end{array} $$ 


Therefore, we choose \\( z_1 = x_1 \\), the state transforms can be computed as, 


$$ z_1 = x_1  $$ 

$$ z_2 = \dot{z}_1 =  L_fz_1 =  X_3 $$ 


$$ z_3 = \dot{z}_2 = L_f^2z_1 =   -\frac{MgL}{I} sin(X_1) -\frac{k}{I} (X_1 - X_2) $$ 


$$ z_4 = \dot{z}_3 = L_f^3 z_1 = -\frac{MgL}{I} cos(X_1) -\frac{k}{I} (X_3 - X_4) $$ 

The control appears in the derivative of \\( z_4 \\) as \\( L_gL_f^3z_1 \neq 0 \\) and \\( L_gL_f^iz_1 = 0 \\) for i = 0,1 and 2. 

$$ \dot{z}_4 =  L_f^4z_1 + L_gL_f^3z_1 u $$

$$ \dot{z}_4 = \frac{MgL}{I}sin(X_1) \left( X_3^2 + \frac{MgL}{I} cos(X_1) + \frac{K}{I} \right) + \frac{K}{I}(X_1 - X_2 ) \left( \frac{K}{I} + \frac{K}{J} + \frac{MgL}{I} cos(X_1) \right)  + \frac{K}{JI}u = v $$ 


Therefore, by this transformation, the states of the system change as, 


$$ \ddddot{Z}_1 = Z^{(4)}_1= \frac{d^4z_1}{dt^4} = v $$ 

which is a linear system, and can be made to follow any fourth order state dynamics. 

## ADDITIONAL NOTES
#### Interpretation of conditions: 

##### First condition controllability 

The first condition is also the controllability condition we previously defined, 

$$ M_{fg} = \left[ \begin{array}{ccccc} adj_f^0 g(X) & adj_f^1 g(X) & adj_f^2 g(X) & \dots &  adj_f^{n-1} g(X) &\end{array} \right] $$

In the special case of linear system we have, 

$$ \dot{X} = AX + Bu $$

In this example, \\( f(X) = AX \\) and \\( g(X) = B \\). Therefore, 

$$ adj_f^0 g = g (X) = B $$

$$ adj_f g = [f,g] (X) = \frac{\partial g}{\partial X} f(X) -  \frac{\partial f}{\partial X} g(X) =  0 - AB =  - AB $$

$$ adj_f^2 g = [f,[f,g]] (X) = \frac{\partial [f,g]}{\partial X} f(X) -  \frac{\partial f}{\partial X} [f,g] (X) = A^2B $$

Therefore, the first condition reduces to 

$$ M_{fg} = \left[ \begin{array}{ccccc} B & -AB & A^2 B & \dots &  (-1)^{n-1}A^{n-1} B &\end{array} \right] $$

Note, the rank of the matrix above, is the same as the rank of the matrix 

$$ M_{fg} = \left[ \begin{array}{ccccc} B & AB & A^2 B & \dots &  A^{n-1} B &\end{array} \right] $$

and we have the same condition for the linear system to be controllable. 



```matlab

```
