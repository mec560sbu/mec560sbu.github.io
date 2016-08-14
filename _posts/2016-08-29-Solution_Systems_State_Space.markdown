---
layout: post
comments: true
title:  "Solution of linear differential equations in state space form"
excerpt: "Solution of state space models and their properties."
date:   2016-08-29 11:00:00
mathjax: true
---

###### Vivek Yadav

#### 1. Continuous systems

In general, dynamic systems can be modeled as, 

$$ \dot{x} = f(x,u,t). $$

$$ y = g(x,u,t). $$

In special case of linear time invariant systems, these equations can be written as 

$$ \frac{d x(t) }{dt} = A x(t) + B u(t) $$

$$ y(t) = C x(t) + D u(t).$$

We now wish to solve these systems of equation to obtain closed form solutions of the state vector \\( x(t) \\) and measurement vector \\( y(t) \\).

#### 1. When  \\(u(t) =0 \\) for all \\( t \\).

We first consider the simple example where \\( u(t) = 0 \\) for all \\( t \\). In this case, the equations reduce to, 

$$ \frac{d x(t) }{dt} = A x(t)  $$

$$ y(t) = C x(t)$$

Recall, that solution for scalar case \\( \dot{\alpha} = a \alpha \\) is \\( \alpha(t) = \alpha_0 exp(at) \\), a similar solution can be obtained for the case when \\( \alpha \\) is a vector and \\(A \\) a matrix. In this case, the solution is, 

$$x(t) = e^{At} x_0$$

Note, as \\(A \\) is a matrix, we need to define its exponent. 

$$e^{At} = I + At + \frac{A^2t^2}{2!} + \frac{A^3t^3}{3!} + \frac{A^4t^4}{4!} + \dots $$
 
If the state at some other instant \\( \tau \\) is known, then the solution of $\dot{x} = Ax$ becomes

$$x(t) = e^{A(t-\tau)} x(\tau)$$

We will next investigate the properties of this solution. Say the matrix A can be written as \\( A = V \Lambda V^{-1} \\), i.e. A has linearly independent eigen vectors. In this case, the solution can be rewritten as, 

$$x(t) = e^{A(t)} x_0 =  \left ( I + At + \frac{A^2t^2}{2!} + \dots \right) x_0$$

Rewriting \\( A = V \Lambda V^{-1} \\), and expanding,

$$x(t) = V \left ( I + \Lambda t + 
\frac{\Lambda^2t^2}{2!} + \dots \right) V^{-1} x_0$$

$$x(t) = V e^{\Lambda t} V^{-1} x_0,$$

where \\(e^{\Lambda t}\\) is the matrix with zero off-diagonal entires and whose \\(i^{th}\\) diagonal entry is \\(e^{\lambda_i t}\\). If all the real parts of the eigen values of \\(A\\) are negative, then the solution \\(x(t) \rightarrow 0\\), else \\(x(t) \rightarrow \infty\\). This property is crucial in designing control systems. As we will see later in this course, we want to design the control law in such a way that the modified eigenvalues of system dynamics are negative. 




#### 2. General solution.

We now seek a particular solution to the full equation \\( \frac{d x(t) }{dt} = A x(t) + B u(t) \\) of the form \\(x(t) = e^{At} c(t)\\). Substituting \\(x(t) = e^{At} c(t)\\) and expanding, 

$$ A e^{At} c(t) + e^{At} \frac{d c }{dt} = A e^{At} c(t)  + B u(t), $$

Simplifying and rearranging, 

$$ \frac{d c }{dt} =  e^{-At} B u(t).  $$

Integrating with respect to time,

$$ c(t)  =  \int_{0}^{t} e^{-A \lambda } B u(\lambda) d\lambda + constant.  $$

Therefore, the solution for  \\( \frac{d x(t) }{dt} = A x(t) + B u(t) \\) is

$$ x(t) = e^{At}c(t)  =  e^{At} \int_{0}^{t} e^{-A \lambda } B u(\lambda) d\lambda + e^{At} constant.  $$

Noting that \\(x(0) = x_0\\),
 
$$ x(t) = e^{At}c(t)  = e^{At} x_0 +  \int_{0}^{t} e^{A(t- \lambda) } B u(\lambda) d\lambda .  $$

Therefore, the solution of the differential equation in state space form is, 

$$ x(t) = e^{At} x_0 +  \int_{0}^{t} e^{A(t- \lambda) } B u(\lambda) d\lambda .  $$

For the case where \\(x \\) is known at some intermediate point \\(\tau\\),

$$ x(t) = e^{A(t-\tau)} x(\tau) +  \int_{\tau}^{t} e^{A(t- \lambda) } B u(\lambda) d\lambda .  $$

This solution has the following properties, 

1. The solution for \\(x(t) \\) depends only on the control signals between 0 (or \\(\tau\\)) and current time (t). 
2. The solutions for  \\(x(t) \\) are bounded if \\(e^{At} \\) (or \\(e^{A(t-\tau)} \\) is bounded is no longer valid, because the solution depends on $B$ and control history \\(u(t)\\) also. 


#### 2. Discrete systems

Discrete systems are given by, 

In general, dynamic systems can be modeled as, 

$$ X[k+1] = f(x[k],u[k]). $$

$$ y[k] = g(x[k],u[k]). $$

In the special case of linear time invariant systems, these equations can be written as 

$$ x[k+1] = A x[k] + B u[k] $$

$$ y[k] = C x[k] + D u[k].$$

#### When u[k] = 0

When \\(u[k] = 0\\), the system dynamics reduces to,

$$ x[k+1] = A x[k] $$

It can easily be shown that the solution in this case is, 

$$ x[k+1] = A^{k+1} x[0] $$

Recall, \\(A^{k+1} = V \Lambda^{k+1} V^{-1}\\), therefore, 

$$ x[k+1] = V \Lambda^{k+1} V^{-1} x[0]. $$

If the absolute value of the eigen values of \\(A\\) are all less than 1, then the solutions \\(x[k+1]\\) converge to \\(0\\), else they go to \\(\infty\\).

Note: For continuous case, we got the condition for the solutions to go to zero as eigen values being negative, and for discrete case we got the condition that the absolute values of the eigen values must be less than 1. These two critera refer to the same condition, and this will be explained later in this section. 

#### Solution by induction

The equation for the first point is,

$$ x[1] = A x[0] + B u[0] $$

The equation for \\(x[2]\\) is. 

$$ x[2] = A x[1] + B u[1] $$

Substituting \\(x[1]\\) and rearranging,

$$ x[2] = A ( A x[0] + B u[0] ) + B u[1] $$

$$ x[2] = A^2 x[0] + A B u[0] + B u[1] $$

Similarly, \\(x[3]\\) is,

$$ x[3] = A^3 x[0] + A^2 B u[0] + A B u[1] + B u[2] $$

Based on the trends above, we propose that solution for \\(x[k]\\) is, 

$$ x[k] = A^{k} x[0] + \sum_{i=0}^{k-1}A^{k-i} B u[i] $$

Given \\(x[k]\\) above, \\(x[k+1]\\) is, 

$$ x[k+1] = A x[k] + B u[k]$$

Substituting \\(x[k]\\) and rearranging,

$$ x[k+1] = A \left( A^{k} x[0] + \sum_{i=0}^{k-1}A^{k-i-1} B u[i]  \right) + B u[k]$$

$$ x[k+1] = A^{k+1} x[0] + \sum_{i=0}^{k-1}A^{k-i} B u[i] + B u[k]$$

$$ x[k+1] = A^{k+1} x[0] + \sum_{i=0}^{k}A^{k-i} B u[i]$$

As the equation for \\(x[k+1]\\) has the same form as \\(x[k]\\) with the iterator \\(k\\) replaced by \\(k+1\\), we can conclude that the solution is 

$$ x[k] = A^{k} x[0] + \sum_{i=0}^{k-1}A^{k-i} B u[i] $$

This solution has the following properties, 

1. The solution for \\(x[k]\\) depends only on the control signals between 0 and current time instant (k). 
2. The solutions for \\(x[k]\\) are bounded if \\(A^{k}\\) is bounded is no longer valid, because the solution depends on $B$ and control history \\(u[k]\\) also. 

#### 3. Relation between continuous and discrete representations

Until now, we treated continuous and discrete systems independently, however, a continuous system can be written in discrete form too. Consider the continuous time system equation 

$$ \frac{d x(t) }{dt} = A x(t) + B u(t) $$

$$ y(t) = C x(t) + D u(t).$$

The derivative of $x[t]$ can be approximated as, 

$$ \frac{d x(t) }{dt} = \frac{x[t+\Delta t] - x[t]}{\Delta t} $$

where \\(\Delta t\\) is a 'reasonable' choice of discretization step. Appropriate choice of \\(\Delta t\\) is crucial in designing control laws and in studying dynamic systems via simulations. A smaller \\(\Delta t\\) although guarantees better accuracy results in large simulation times, where as a larger \\(\Delta t\\) gives faster simulation results but has poorer accuracy. Choosing \\(\Delta t\\) correctly can give accurate results in reasonable time. Simulation of dynamic systems will be dealt in a different section. Typically \\(\Delta t\\) is chosen as a very small number \\(0.001s\\), however, if the system is highly nonlinear and has millions of parameters (neural networks for example) a much smaller \\(\Delta t\\) (of the order of \\(10^-6\\)) is warranted.

The system dynamics equations now become, 

$$\frac{x[t+\Delta t] - x[t]}{\Delta t}  = A x(t) + B u(t) $$

$$x[t+\Delta t] = x[t]+ A \Delta t x(t) + B\Delta t u(t) $$

$$x[t+\Delta t] = (I+ A \Delta t) x(t) + B\Delta t u(t) $$

Note, if all the eigen values of \\(A\\) are positive, then the eigen values of \\((I+ A \Delta t)\\) for a small \\(\Delta t\\) are negative, and \\(e^{At}\\) or \\((I+ A \Delta t)^k\\) both will converge to \\(0\\) as \\(t,k \rightarrow \infty \\).

#### Conclusion:

In this section, we saw how solutions of system dynamics equations represetented in state space form can be calculated. We also saw that if the real part of eigen values of a continuous system are negative or a discrete system have magnitudes less than 1, then the system's state go to zero when no control is applied. This property is crucial and will be exploited to design control laws. However, before designing control laws, it is important to test if the system is actually controllable. Further, as control law is a function of estimated states, it is important to verify if all the states can be estimated, i.e. are observable. Concepts of controllability and observability are developed to address these specific questions, and we will study them next.



