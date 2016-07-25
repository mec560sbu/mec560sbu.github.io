---
layout: post
comments: true
title:  "Linear regression using matrix derivatives."
excerpt: "This post presents basic matrix calculus relations and demonstrates how they can be applied to obtain the coeffients in linear regression. Two methods, gradient descent and pseudoinverse-based solution are presented."
date:   2016-06-25 11:00:00
mathjax: true
---



###### Vivek Yadav, PhD

#### Overview

Matrix calculations are involved in almost all machine learning algorithms. This series of posts will present basics of matrix calculations and demonstrate how it can be used to develop learning rules. The most common technique is to parameterize the error function as a function of few scalars, calculate the derivative of the error with respect to the parameters and look for parameters that minimize the error cost function. The process of iteratively solving for the parameters that give the smallest minimum error is also refered as gradient descent. In this post we will go over basic matrix calculations, and will apply them to derive the coefficients for the best fit line. In certain special cases, where the predictor function is linear in terms of the unknown parameters, a closed form pseudoinverse solution can be obtained. This post presents both gradient descent and pseudoinverse-based solution for obtaining the coefficients in linear regression.


### 2. First order derivatives with respect to a scalar and vector
This section presents the basics of matrix calculus and shows how they are used to express derivatives of simple functions. First, lets clarify some notations, a scalar is represented by a lower case non-bold letter like $a$, a vector by a lower case bold letter such as \\( \textbf{a} \\) and a matrix by a upper case bold letter \\( \mathbf{A}  \\). 

#### 2.1 Derivative of a scalar function with respect to vector
Derivative of a scalar function with respect to a vector is the vector of the derivative of the scalar function with respect to individual components of the vector. Therefore, for a function \\(f \\) of the vector \\( \mathbf{x} \\), 

$$\frac{\partial{ f ( \mathbf{x}} ) }{\partial{\mathbf{x}}}= \left[ \frac{\partial{ f ( \mathbf{x}} ) }{\partial{x_1}}  \frac{\partial{ f ( \mathbf{x}} ) }{\partial{x_2}} ...  \frac{\partial{ f ( \mathbf{x}} ) }{\partial{x_n}}  \right]^T $$ 

Similarly, the derivative of the dot product of two vectors \\( \mathbf{a} \\) and \\( \mathbf{x} \\) in \\( R^n \\) can be written as, 

$$\frac{\partial{ \mathbf{x}^T \mathbf{a}}}{\partial{\mathbf{x}}} = \frac{\partial{ \mathbf{a}^T \mathbf{x}}}{\partial{\mathbf{x}}}= \mathbf{a} $$ 

Similarly, 

$$\frac{\partial{ \mathbf{A} \mathbf{x}}}{\partial{\mathbf{x}}} = \mathbf{A}$$ 

*** NOTE: If the function is scalar, and the vector with respect to which we are calculating the derivative is of dimension \\( n \times 1 \\) , then the derivative is of dimension \\( n \times 1 \\).***


#### 2.2 Derivative of a vector function with respect to a scalar
Derivative of a vector function with respect to a scalar is the vector of the derivative of the vector function with respect to the scalar variable. Therefore, for a function \\( \mathbf{f} \\) of the variable \\( x \\), 

$$\frac{\partial{  \mathbf{f (x}} ) }{\partial{x}}= \left[ \frac{\partial{ f_1 ( \mathbf{x}} ) }{\partial{x}}  \frac{\partial{ f_2( \mathbf{x}} ) }{\partial{x}} \ldots \frac{\partial{ f_m ( \mathbf{x}} ) }{\partial{x}} \right]^T $$ 


*** NOTE: If the function is a vector of dimension \\( m \times 1 \\) then its derivative with respect to a scalar is of dimension \\( m \times 1 \\).***



#### 2.3 Derivative of a vector function with respect to vector
Derivative of a vector function with respect to a vector is the matrix whose entries are individual component of the vector function with respect to to individual components of the vector. Therefore, for a vector function \\(  \mathbf{f} \\) of the vector \\(  \mathbf{x} \\), 

$$\frac{\partial{ \mathbf{ f(x}} ) }{\partial{\mathbf{x}}}= \left[ \frac{\partial{  \mathbf{ f(x)}} }{\partial{x_1}}  \frac{\partial{  \mathbf{ f(x)}} }{\partial{x_2}} ...  \frac{\partial{  \mathbf{ f(x)}} }{\partial{x_n}}  \right] $$ 


$$  \left( \begin{array}{c}
\frac{\partial{ f_1 ( \mathbf{x}} ) }{\partial{\mathbf{x}}} \\
\frac{\partial{ f_2 ( \mathbf{x}} ) }{\partial{\mathbf{x}}} \\
\vdots \\
\frac{\partial{ f_m ( \mathbf{x}} ) }{\partial{\mathbf{x_1}}} 
\end{array} \right) = \left( \begin{array}{cccc}
\frac{\partial{ f_1 ( \mathbf{x}} ) }{\partial{x_1}} & \frac{\partial{ f_1 ( \mathbf{x}} ) }{\partial{x_2}} & \ldots & \frac{\partial{ f_1 ( \mathbf{x}} ) }{\partial{x_n}}\\
\frac{\partial{ f_2 ( \mathbf{x}} ) }{\partial{x_1}} & \frac{\partial{ f_2 ( \mathbf{x}} ) }{\partial{x_2}} & \ldots & \frac{\partial{ f_2 ( \mathbf{x}} ) }{\partial{x_n}}\\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial{ f_m ( \mathbf{x}} ) }{\partial{x_1}} & \frac{\partial{ f_m ( \mathbf{x}} ) }{\partial{x_2}} & \ldots & \frac{\partial{ f_m ( \mathbf{x}} ) }{\partial{x_n}}
\end{array} \right) $$



*** NOTE: If the function is a vector of dimension \\(  m \times 1 \\), and the vector with respect to which we are calculating the derivative is of dimension \\(  n \times 1 \\), then the derivative is of dimension \\(  m \times n \\) .***

For the special case where,

$$ \alpha(x) = \mathbf{x^T A x} $$,

$$ \frac{\partial \alpha(\mathbf{x})}{ \partial \mathbf{x}} = \mathbf{x^T (A^T+A)}$$





### 3. Application of derivative of a scalar with respect to a vector
One example where the concepts derived above can be applied is line fitting. In line fitting, the goal is to find the vector of weights that minimize the error between the target value and predictions from a linear model. 

#### Problem statement
Given a set of \\( N \\) training points such that for each \\( i \in [0,N] \\), \\(  \mathbf{x}_i \in R^M \\)  maps to a scalar output \\(y_i\\), the goal is to fine a linear vector \\(\mathbf{a}\\) such that the error

$$ J \left(\mathbf{a},a_0 \right) = \frac{1}{2} \sum_{i}^{N} \sum_{j}^{m} \left( a_j x_{i,j} + a_0 - y_i    \right)^2 , $$

is minimized. Where \\(a_j \\) is the \\( j^{th} \\) component and \\( a_0 \\) is the intercept. 

#### Solution
First we will augment the independent vector \\( \mathbf{x}_i \\) by inserting a new element 1 to the end of it, and  we will augment vector \\( \mathbf{a} \\) by inserting \\( a_0 \\) at the end. This will result in the cost functin changing as, 
   
$$ J \left(\mathbf{W} \right) = \frac{1}{2}  \sum_{i}^{N} \sum_{j}^{m+1} \left( w_j x_{i,j} - y_i    \right)^2 , $$



The cost fuction above can be rewritten as a dot product of the error vector 

$$ e(W) = \left( X W - y    \right) .$$

where X is a matrix of n times m+1, y is a n-dimensional vector and W is (m+1)-dimensional vector.

Therefore, the cost function J is now, 
$$ J \left(\mathbf{W} \right) = \frac{1}{2} e(W)^T e(W)= \frac{1}{2} \left( \mathbf{ X W - y    }  \right)^T  \left(\mathbf{ X W - y    } \right) .$$

Expanding the transpose term, 
$$ J \left(\mathbf{W} \right) = \frac{1}{2} \left( \mathbf{W^T X^T - y^T}    \right)  \left(\mathbf{ X W - y    }\right) $$

$$  = \frac{1}{2} \left( \mathbf{ W^T X^T  X W - 2 y^T X W + y^T y}    \right)  $$

Note, all terms in the equation above are scalar. Taking derivative with respect to W gives, 

$$ \frac{\partial \mathbf{J}}{\partial \mathbf{W}} =  \left( \mathbf{ W^T X^T  X - y^T X }    \right)  $$

#### Gradient-descent method

Minimum is obtained when the derivative above is zero. Most commonly used approach is a gradient descent based solution where we start with some initial guess for W, and update it as, 

$$ \mathbf{W_{k+1}}  = \mathbf{W_{k}} - \mu \frac{\partial \mathbf{J}}{\partial \mathbf{W}} $$

It always a good idea to test if the analytically computed derivative is correct, this is done by using the central difference method, 
$$ \frac{\partial \mathbf{J}}{\partial \mathbf{W}}_{numerical} \approx  \frac{\mathbf{J(W+h) - J(W-h)}}{2h} $$

Central difference method is better suited because the central difference method has error of \\( O(h^2 \\), while the forward difference has error of \\( O(h) \\). Interested readers may look up Taylor series expansion or may choose to wait for a later post.


#### Pseudoinverse method
By taking transpose and solving for W, we get, 

$$ \mathbf{W} = \mathbf{ \underbrace{(X^T  X)^{-1}X^T}_{Pseudoinverse} ~ y } $$







### Example: Linear regression

In this part, we will generate some data and apply the methods presented above. We will get the equation of the best fit line using gradient descent and pseudo inverse methods, and compare them. 


```python
## Generate toy data
import numpy as np
import matplotlib.pyplot as plt
import time 
%pylab inline

X = 4*np.random.rand(100)
Y = 2*X + 1+ 1*np.random.randn(100)

X_stack = np.vstack((X,np.ones(np.shape(X)))).T

plt.plot(X,Y,'*')
plt.xlabel('X')
plt.ylabel('Y')

```

    Populating the interactive namespace from numpy and matplotlib


<div class='fig figcenter fighighlight'>
  <img src='/images/lin_reg0.png'>
</div>



#### Linear Regression using gradient descent

Gradient descent method is used to calculate the best-fit line. A small value of learning rate is used. We will discuss how to choose learning rate in a different post, but for now, lets assume that 0.00005 is a good choice for the learning rate. Gradient is computed using the equation presented in section 2.3, and the weights (or coefficients) are stored for each step. 


```python
def get_numerical_derv(W):
    h = 0.00001

    W1 = W + [h,0]
    errpl1 = np.dot(X_stack,W1)-Y
    W1 = W - [h,0]
    errmi1 = np.dot(X_stack,W1)-Y
    W2 = W + [0,h]
    errpl2 = np.dot(X_stack,W2)-Y
    W2 = W - [0,h]
    errmi2 = np.dot(X_stack,W2)-Y
    
    dJdW_num1 = (np.dot(errpl1,errpl1)-np.dot(errmi1,errmi1))/2./h/2.
    dJdW_num2 = (np.dot(errpl2,errpl2)-np.dot(errmi2,errmi2))/2./2./h

    dJdW_num = [dJdW_num1,dJdW_num2]
    return dJdW_num


```


```python
t0 = time.time()
W_all = []
err_all = []
W = np.zeros((2))
lr = 0.00005
h = 0.0001
for i in np.arange(0,250):
    

    
    W_all.append(W)
    err = np.dot(X_stack,W)-Y
    err_all.append(  np.dot(err,err) )
    XtX = np.dot(X_stack.T,X_stack)
    dJdW = np.dot(W.T,XtX) - np.dot(Y.T,X_stack)
    if (i%50)==0:
        dJdW_n = get_numerical_derv(W)
        print 'Iteration # ',i
        print 'Numerical gradient: [%0.2f,%0.2f]'%(dJdW_n[0],dJdW_n[1])
        print 'Analytical gradient:[%0.2f,%0.2f]'%(dJdW[0],dJdW[1])
    
    W = W - lr*dJdW

tf = time.time()
print 'Gradient descent took %0.6f s'%(tf-t0)

plt.plot(err_all)
plt.xlabel('iteration #')
plt.ylabel('RMS Error')


```

    Iteration #  0
    Numerical gradient: [-1327.59,-519.43]
    Analytical gradient:[-1327.59,-519.43]
    Iteration #  50
    Numerical gradient: [-257.34,-108.61]
    Analytical gradient:[-257.34,-108.61]
    Iteration #  100
    Numerical gradient: [-47.88,-27.81]
    Analytical gradient:[-47.88,-27.81]
    Iteration #  150
    Numerical gradient: [-6.99,-11.67]
    Analytical gradient:[-6.99,-11.67]
    Iteration #  200
    Numerical gradient: [0.90,-8.20]
    Analytical gradient:[0.90,-8.20]

    Gradient descent took 0.005565 s




<div class='fig figcenter fighighlight'>
  <img src='/images/lin_reg1.png'>
</div>



```python
plt.figure(figsize=(15,20))
for i in np.arange(0,8):
    num_fig = i*30
    Y_pred = W_all[num_fig][0]*X + W_all[num_fig][1]
    plt.subplot(4,2,i+1)
    plt.plot(X,Y,'*',X,Y_pred)
    title_str = 'After %d iterations: %0.2f X  + %0.2f'%(num_fig,
                          W_all[num_fig][0],W_all[num_fig][1])
    plt.title(title_str)
```

<div class='fig figcenter fighighlight'>
  <img src='/images/lin_reg2.png'>
</div>


#### Linear Regression using Pseudoinverse

The same calculations above can be performed using Pseudoinverse. 


```python
t0 = time.time()
XTX_inv = np.linalg.inv(np.dot(X_stack.T,X_stack))
XtY = np.dot(X_stack.T,Y)
W = np.dot(XTX_inv, XtY)
Y_pred = W[0]*X + W[1]
tf = time.time()
print 'Gradient descent took %0.6f s'%(tf-t0)

title_str = 'Predicted function is %0.2f X + %0.2f'%(W[0],W[1])

plt.plot(X,Y,'*',X,Y_pred)
plt.title(title_str)
plt.xlabel('X')
plt.ylabel('Y')

```

    Gradient descent took 0.000504 s


<div class='fig figcenter fighighlight'>
  <img src='/images/lin_reg3.png'>
</div>


### Conclusion

In this post, matrix equations to compute derivatives with respect to a scalar and vector were presented. For cases where the model is linear in terms of the unknown parameters, the 
These techniques were applied to compute best-fit line using 

Pseudoinverse is almost 10 times faster because it does not involve the iterative gradient descent process. However, pseudoinverse method is applicable only when the prediction function is linear in unknown parameters. 
