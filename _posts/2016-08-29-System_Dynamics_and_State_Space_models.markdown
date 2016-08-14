---
layout: post
comments: true
title:  "System dynamics and state space models"
excerpt: "System dynamics and how to express them in state space form"
date:   2016-08-29 11:00:00
mathjax: true
---

### System dynamics 

#### 1. Signals

Signals map from a domain \\(T\\) (typically time or sample number) to a set \\(W\\) where \\(W\\) is a vector space, typically \\(\in R^n\\) for some \\(n\\). 

1. Discrete singals: Discrete signals are map from integer space \\(Z\\) to some \\(R^n\\).
2. Continuous singals: Continuous signals are map from real space (\\(R\\)) to some \\(R^n\\). In addition, continuous signals are constrained by conditions for piecewise-continuity or local integrability. 
3. Discrete-Continuous singals: In some cases, it is difficult to model a signal or phenomena as purely continuous or discrete. In such cases, a mixed approach is taken. For example, trying to monitor a time varying continuous signal using digital sensing results in mixed singal types. 

#### 2. System dynamics or behavioral models
“System dynamics is a perspective and set of conceptual tools that enable us to understand the structure and dynamics of complex systems. System dynamics is also a rigorous modeling method that enables us to build formal computer simulations of complex systems and use them to design more effective policies and organizations. Together, these tools allow us to create management flight simulators-microworlds where space and time can be compressed and slowed so we can experience the long-term side effects of decisions, speed learning, develop our understanding of complex systems, and design structures and strategies for greater success.” - *John Sterman, “Business Dynamics: Systems Thinking and Modeling for a Complex World”*

System dynamics refers to the mathematical techniques used to describe the behavior of a complex dynamic system. This is typically achieved by utilizing differential equations or difference equations. When differential equations are employed, the theory is called continuous dynamical systems. When difference equations are employed, the theory is called discrete dynamical systems. In some applications, it may not be possible to describe a system as either continuous or discrete dynamical systems, in such cases, the system may be modeled using both the differential and difference equations. 

Types of dynamic systems:

1. Linear: A system is linear if its behavioral model is a vector space. 
2. Time-invariant: A system is time-invariant if its behavioral model is independent of time-shift, i.e. for a signal \\(w: T \rightarrow W\\), if \\(w(t) \in B\\) implies \\(w(t-\tau) \in B \\) for any \\(\tau \in T\\).
3. Memoryless: A system is called memory less if the possible future values are independent of the past values. 
4. Strictly memoryless: A system is strictly memory less if the constraint imposed on the system are purely algebraic and have no derivative, integrals, etc. 

#### 3. Input-output models: 

Behavioral models treat all constraints equally without any emphasis on their role or interpratation. Now we will make distinction between some of the components, more specifically, we will call one set of singals input (\\(x : T \rightarrow R^n \\)) and another output (\\(y: T \rightarrow R^m\\)), where \\(n\\) and \\(m\\) are number of dimensions of \\(x(t)\\) and \\(y(t)\\), for \\(t\in T\\). A behavioral system is a map that relates input signals to output signals.
 
$$B = {(x,y): y = F(x)}.$$

Types of input-output models*:

1. Linear: If the behavioral model \\(B\\) can be expressed as a linear map \\(S\\), such that \\(B = {(x,y): y = S(x)}\\), and \\(S(\alpha x_a + \beta x_b) = \alpha S(x_a)+\beta S(x_b) = \alpha y_a+\beta y_b \\) for every \\(\alpha , \beta \in R\\), then \\(B\\) is a linear system.   
2. Time-invariant: An input-output system \\(S\\) is time-invariant if its behavioral model is independent of time-shift, i.e. \\(S(u(t-\tau)) = S(u(t))\\).
3. Memoryless: An input-output system is called memory less if the possible future values are independent of the past values. 
4. Causal: An input-output system is causal if the future values depend only on values until the current time, i.e. the future values depend only on the past values.

The definitions above are not complete, as they do not allow us to predict the output of the signal given past history of the input and output signals. Next, state space models are presented that allow us to relate inputs to output signals. 

\\(*\\) Note, \\(x\\) was used to denote input signals, but in some applications this is also denoted by \\(u\\). The underlying ideas are the same, however \\(x\\) is typically used to denote the states of the system and \\(u\\) the control input. 

### State space models:

State space models are described for **causal** systems where the next system's behavior depends solely on the history of underlying signals and current input. We first define a concept of 'state' for a system. 

#### State: 
State of a causal system \\(x(t)\\) at time \\(t\\) is the vector of variables containing all the required information, along with the control input \\(u(t)\\) between \\(t\\) and \\(t+h\\) to predict the state of the system at \\(x(t+h)\\). Consider the example of a car moving along a straight line whose equations of motion are given by \\(F = M a = M \ddot{x}\\). In this case, \\(F\\) is the control input, and the position (\\(x\\)) and velocity (\\(\dot{x}\\)) form the states of the system.

Note, the choice of states of a system are not unique. The choice of the states may vary depending upon the coordinate system chosen, types of measurement or feedback availalble, or nature of uncertainity. Prudently choosing the states of system can significantly simplify the resulting control laws.

In some cases, it may not be possible to express the state of a system using a fixed number of variables. Consider the example of flow control, in this case, the mass flow rate, velocity etc are all infinite dimensional quantities. A lower dimensional approximation can be obtained by suitably discretizing the system. However, this results in an approximate representation of dynamics, and any attempt to reduce the approximation error results in significantly increasing the number of state variables. 

#### Dimension of a system: 
We define the dimension of a causal system as the minimal number of variables sufficient to describe the system’s state (i.e., the dimension of the smallest state vector). For our car example, the dimension of the system is 2. 

#### State Space model: 

State space model of a finite-dimension linear system can be written as, 

$$ \frac{d x(t) }{dt} = A(t) x(t) + B(t) u(t) $$

where, \\(x(t)\\) is a n-dimensional state vector, \\(A(t)\\) is a \\(n \times n\\) matrix, \\(u(t)\\) is a m-dimensional input (or control) vector and \\(B(t)\\) is a \\(n \times m\\) matrix relating control inputs to the rate of change of states. The state of system are typically not directly measurable, and are monitored using measurement signals given by, 

$$ y(t) = C(t) x(t) + D(t) u(t)$$

where \\(y(t)\\) is a p-dimensional measurement vector, \\(C(t)\\) is a \\(p\times n\\) dimensional matrix, \\(u(t)\\) is the 

$$ x[k+1] = A[k] x[k] + B[k] u[k] $$

$$ y[k] = C[k] x[k] + D[k] u[k]$$

In cases where the dynamics do not depend on time, the state space equations reduce to, 

$$ \frac{d x(t) }{dt} = A x(t) + B u(t) $$

$$ y(t) = C x(t) + D u(t),$$

for continuous systems and 

$$ x[k+1] = A x[k] + B u[k] $$

$$ y[k] = C x[k] + D u[k]$$

for discrete systems. 

State space representations are extensively studied because several systems can be expressed in linear form as presented above. We will now look into a few examples of systems that can be written in linear form. Note, we always work with first order differential equations, in cases where the differential equation is not first order, we introduce another variable into the state representation to make the differential equations first order. 

#### State space: examples

#### 1. Car moving along a straight line

Consider the example of a car moving along a straight line, in this case the equations of motion are given by, 

$$  \ddot{x} = a = \frac{F}{M}$$

and say the measurement available is the velocity of the car, 

$$ y(t) = \dot{x}(t)$$

The equation \\(  \ddot{x} = a = \frac{F}{M}\\) is not first order in \\(x\\), we therefore introduce another variable \\(v = \dot{x}\\) and rewrite the above equation as, 

$$  \dot{v} = \frac{F}{M}$$

$$ y(t) = v(t)$$



These equations can be written in the state space form as follows, 

$$ \frac{d}{dt} \left[ \begin{array}{c}
x   \\
v
\end{array} \right] = \underbrace{\left[ \begin{array}{cc}
0 & 1   \\
0 & 0
\end{array} \right]}_A \left[ \begin{array}{c}
x   \\
v
\end{array} \right] + \underbrace{\left[ \begin{array}{c}
0   \\
\frac{1}{M}
\end{array} \right]}_B F $$

and 

$$y(t) = \underbrace{\left[ \begin{array}{cc}
0 & 1
\end{array} \right]}_C  \left[ \begin{array}{c}
x(t)   \\
v(t)
\end{array} \right]. $$

#### 2. Spring mass damper

The equations of motion of spring-mass-damper system can be written as, 

$$ M \ddot{x} + B \dot{x} + Kx = F $$ 

with measurement \\(y = x\\). The state dynamic equation is not first order, however, defining \\(v = \dot{x}\\), and rearranging terms gives us two first order systems.

$$v = \dot{x}$$

$$ \dot{v} = - \frac{B}{M} v -\frac{k}{M}x + \frac{F}{M} $$ 

These equations can be written in the state space form as follows, 

$$ \frac{d}{dt} \left[ \begin{array}{c}
x   \\
v
\end{array} \right] = \underbrace{\left[ \begin{array}{cc}
0 & 1   \\
-\frac{K}{M} & -\frac{B}{M}
\end{array} \right]}_A \left[ \begin{array}{c}
x   \\
v
\end{array} \right] + \underbrace{\left[ \begin{array}{c}
0   \\
\frac{1}{M}
\end{array} \right]}_B F $$

and 

$$y(t) = \underbrace{\left[ \begin{array}{cc}
1 & 0
\end{array} \right]}_C  \left[ \begin{array}{c}
x(t)   \\
v(t)
\end{array} \right]. $$

Next, we will go over techniques to obtain solution for systems represented in a state space form.















