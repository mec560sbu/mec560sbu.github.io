---
layout: post
comments: true
title:  "Controllability and observability of linear dynamic systems"
excerpt: "Controllability and observability of linear dynamic systems."
date:   2016-09-19 11:00:00
mathjax: true
---


#### Vivek Yadav

Before getting into design control laws, it is crucial to first determine if the system is controllable or not? Further, as control laws are based on state estimates obtained from the available sensor measurements, it is important to know if the state estimates required for control are observable. These two specific questions are addressed by the concepts fo controllability and observability. 

#### Controllability: 

We say a system is controllable if given any initial state vector and a final state vector, we can find at least one control law that ensures that the final state vector can be achieved. 

##### Controllability of a continuous linear system
Say the continuous dynamic system is given by  

$$ \frac{d x(t) }{dt} = A x(t) + B u(t) .$$

$$ x(t) = e^{At} x_0 +  \int_{0}^{t} e^{A(t- \lambda) } B u(\lambda) d\lambda .  $$

Therefore, given a final state \\(x_f \\), for some time \\(t_f\\), we have

$$ x_f = e^{At} x_0 +  \int_{0}^{t_f} e^{A(t_f- \lambda) } B u(\lambda) d\lambda .  $$

Using Cayley-Hamilton theorm we can express \\( e^{A(t_f- \lambda)}  \\), 

$$ e^{A(t_f- \lambda)} = \sum_{i=0}^{n-1} a_i(t_f,\lambda) A^{i}$$

Recall, the solution of this equation is

$$ x_f = e^{At_f}c(t_f)  = e^{At_f} x_0 +  \int_{0}^{t_f} e^{A(t_f- \lambda) } B u(\lambda) d\lambda .  $$

Substituting \\( e^{A(t- \lambda)} \\) gives

$$ x_f = e^{At_f} x_0 +  \int_{0}^{t_f}  \sum_{i=0}^{n-1} a_i(t_f,\lambda) A^{i} B u(\lambda) d\lambda .  $$

As \\( A \\) and \\( B\\) are constant, we can rearrange the solution equation as

$$ x_f - e^{At_f} x_0 =   \underbrace{\left[ \begin{array}{ccccc} B & AB & A^2B & \dots & A^{n-1}B \end{array}  \right]}_{controllability~matrix}  \left[ \begin{array}{c} \int_{0}^{t} a_0(t_f,\lambda) d\lambda  \\ \int_{0}^{t} a_1(t_f,\lambda) d\lambda  \\ \int_{0}^{t} a_2(t_f,\lambda) d\lambda  \\ \vdots \\ \int_{0}^{t} a_{i-1}(t_f,\lambda) d\lambda  \end{array}  \right] .  $$

The equation above has solution for any value of \\(x_f\\) and \\(x_0\\) if and only if \\( \left[ \begin{array}{ccccc} B & AB & A^2B & \dots & A^{n-1}B \end{array}  \right]  \\) has rank \\( n \\). The \\( \left[ \begin{array}{ccccc} B & AB & A^2B & \dots & A^{n-1}B \end{array}  \right]  \\) matrix is also refered as the controllability matrix. ** Controllability matrix can be computed in MATLAB using 'ctrb' command.  **

##### Controllability of a discrete linear system

Controllability of a discrete system is also defined in a similar manner. Lets consider a discrete system given by

$$ x[k+1] = A x[k] + Bu[k] . $$

The system above is controllable if for any given values of initial and final states, there exists a sequence of control inputs that when applied to the system above, take the system from the initial state to the final state. The solution for the system above, given the initial state \\( x_0 \\), given \\( n \\) steps between them is, 

$$ x[n] = x_f = A^{n} x[0] + \sum_{i=0}^{n-1}A^{n-i-1} B u[i] $$

$$ x_f = A^{n} x[0] + A^{n-1} B u[0]+A^{n-2} B u[1]+\dots+ B u[n-1]$$


$$ x_f - A^{n} x[0] =   \underbrace{\left[ \begin{array}{ccccc} B & AB & A^2B & \dots & A^{n-1}B \end{array}  \right]}_{controllability~matrix}  \left[ \begin{array}{c} u[n-1]  \\ u[n-2]  \\ u[n-3]  \\ \vdots \\  u[0]  \end{array}  \right] .  $$

The equation above has solution for any value of \\(x_f\\) and \\(x_0\\) if and only if \\( \left[ \begin{array}{ccccc} B & AB & A^2B & \dots & A^{n-1}B \end{array}  \right]  \\) has rank \\( n \\). The \\( \left[ \begin{array}{ccccc} B & AB & A^2B & \dots & A^{n-1}B \end{array}  \right]  \\) matrix is also refered as the controllability matrix. 

Controllability is also expressed via a Grammian matrix P, 

$$ P(T) =  \int_{0}^{T} e^{ A \lambda } BB^T  e^{A^T \lambda }    d\lambda .  $$

If \\( P(T) \\) is non-singular for all \\( T\\) then the underlying dynamic system is controllable. 



##### Similarity transforms

Two square-matrices  \\( A \\)  and  \\( C \\)  are called similar if there exists an invertible square matrix  \\( P \\)  such that, \\( C= P^{-1}AP \\). The transformation  \\(P^{-1}AP \\) is also called similarity. Similarity transforms allow us to express system dynamics in special forms that allow us to draw inferences about the system dynamics easily and design of control easier. Before getting into useful similarity transforms, lets first look into some properties of similarity transforms,

1. Matrices \\( A \\) and \\( C \\) have the same eigen values. Say some \\( \lambda \\) satisfies, \\( det( A - \lambda I ) = 0  \\), then, 
    
    $$ det( C - \lambda I ) = det( P^{-1} A P - \lambda P^{-1} P) = det( P^{-1}( A - \lambda I  ) P )    $$
    
    The experession above is 0 when \\( ( A - \lambda I  )  \\) loses rank, i.e. for eigen values of \\( A \\). Therefore, \\( A \\) and \\( C \\) have the same eigen values, and if we applied similairty tranform to the system dynamics matrix \\( A \\), the corresponding stability properties will not change. 
2. Consider the system \\( \dot{x} = Ax + Bu \\), with the transformation of variables \\( \hat{x} = P x \\). The system dynamics equation modifies as, 

    $$ \dot{ \hat{x}} = P^{-1} A P  + P^{-1}  B. $$
3. By choosing the values of \\( P \\) appropriately, the system dynamics equations can be modified into special forms that are easier for control design and analysis. A few examples of such transforms are, 
    - Jordan tranform
    - Diagonal transform (From HW 1)
    - Canonical transform (From HW 2)
4. Applying similarity transforms does not change the controllability of the system. 

##### Output controllability

In many control applications, we do not have direct measure of the states of the system. The system is monitored using mesurements from the system. For such systems, output controllability is defined with reference to the ability to control the output. A system is said to be output controllable such that for any starting and final output of the system, one or many sequence of control inputs exist that when applied to the system take its output from the initial to final values. A system 

$$ \frac{d x(t) }{dt} = A x(t) + B u(t) .$$

with measurements 

$$ y = C x + D u $$ 

is said to be output controllable if the matrix \\( \left[ \begin{array}{cccccc} CB & CAB & CA^2B & \dots & CA^{n-1}B & D \end{array}  \right]  \\) has rank \\( n \\).



##### When do systems become uncontrollable?

There are several factors that can give rise to an uncontrolable system. The list below is just enumeration of a few of them. 

1. When some states are repeated. If the states are chosen poorely, it may happen that one or more states are completely defined by the other states. In such cases, it is not possible to control all the states independently. 
2. Any external contraints on the system that are not accounted for by the dynamic model. For example, the same set of equations of motion describe movements of a 2-link robotic manipulator that is free to move anywhere and one that is constrained to move along a line. 
3. Controller with smaller bandwidth. Recall, bandwidth is the frequency at which the power of the frequency response drops to -3dB and phase shifts by 180. A larger bandwidth implies the controller is very responsive and can apply the commanded control value.
4. Systems may also become uncontrollable if two or more control variables influence the system in the same way. 
5. Zero-dynamics are defined as the 'residual dynamics' of the system when the output of the system is zero. By definition, the states involved in zero-dynamics cannot always be controlled. We will discuss this concept in more detail later in the course. 

$$ \dot{x_1} = A_1 x_1 + B_1u $$ 

$$ \dot{x_2} = A_{12} x_1 + A_{22} x_2 + B_2 u $$  

with measurement \\(y = x_1\\). Say we apply control such that \\( y = 0\\) this gives \\( x_1 = 0 \\), and the system above reduces to

$$ \dot{x_1} = 0 $$ 

$$ \dot{x_2} = A_{22} x_2$$  

Depending on the value of \\( A_{22} )\\, this system may be controllable or not controllable. If \\( A_{22} \\) is negative, then the \\( x_2 \\) goes to 0, else it blows up. 

#### Stabalizability
In addition to controllability, a concept of stabalizability is also defined. A system is stabalizable if for any given initial conditions, there exists atleast one series of control inputs that drive the system states to 0 either in finite time or asymptotically. All controlable states are stabalizable.
