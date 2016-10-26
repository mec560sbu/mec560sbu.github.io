---
layout: page
title: MiniProject1
date: 2016-06-08 22:22:22
permalink: miniproject1
---



## Obstacle Avoidance for autonomous robot.
### Due: October 24, 2016 at 4 pm. NO LATE SUBMISSIONS ALLOWED

### The best submissions, 

The [best solution was submitted by Courtney Armstrong](https://nbviewer.jupyter.org/github/mec560sbu/mec560sbu.github.io/blob/master/Assignments2016/MiniProject1_solution1/MiniProject1.ipynb), the link to download the accompanying MATLAB code can be found [here](https://drive.google.com/drive/folders/0B51BYOSh3EKQUTZGMjE1ZDgxbk0?usp=sharing). 

Other good solutions, 

1. By [Stephen Leo](https://mec560sbu.github.io/Assignments2016/MiniProject1_solution1/Steven_Leo_MiniProjectWriteUp.pdf)
2. By [Benjamin Bradwin](https://mec560sbu.github.io/Assignments2016/MiniProject1_solution1/BenjaminBen_Final_Project_Paper.pdf)


#### Introductions: 

Autonomous robots have become more prevalent in recent times. For example, self-driving cars are slowly being accepted into main-stream society. With [US government releasing guidelines](http://money.cnn.com/2016/09/19/technology/autonomous-car-government-guidelines/) for self-driving cars, big technology giants like google, apple, microsoft and nvidia developing hardware and technology for seld-driving cars, companies like uber testing cars in field, and auto-makers like ford and volvo having plans to sell self-driven cars by 2020. Here is [a detailed list of coroporations working on self-driven car](http://www.bankrate.com/finance/auto/companies-testing-driverless-cars-1.aspx). Another example is autonomous flying vehicles or UAVs. One of the most basic task of an autonomous robot is to navigate in an environment with multiple obstacles. These obstacles can be buildings, road blocks etc. In this project you will generate control algorithms to achieve obstacle avoidance for an autonotmous robot.  


#### Approach

In previous classes, you saw how to obtain [the shortest path between two points](https://mec560sbu.github.io/2016/09/25/Opt_control/) in an environment with rectangular obstacles. The reinforcement-learning or dynamic programming approach gave a sequence of way-points that can be followed by a robot to reach the desired location. However, the trajectory had sharp corners, and such a trajectory may not be feasible for a robot to follow. This is especially a concern for unmanned aerial vehicles where the heading cannot be changed sharply without risking stalling. You can get a smoother trajectory by performing an optimization problem that minimizes a weighted sum between data and and change in distance. Such a criteria can be expressed as, 

$$  minimize \left( (XY_i  - \hat{XY}_i )^2 + \alpha ((\hat{XY}_i  - \hat{XY}_{i+1} )^2)\right)$$

For i going from 2 to \\\( N-1 \\) where N is then number of points. The start and end locations are the same as original trajectory. So the boundary constraints are, 

$$ XY_1  = \hat{XY}_1 $$ 

$$ XY_N  = \hat{XY}_N $$ 

The first term in the cost-function tries to minimize the distance between the original and approximated trajectory, and the second term penalizes sharp turns in trajectory. The optimization problem above can be solved either using the GPOPs II library or gradient descent. A gradient descent algorithm to obtain a smooth trajectory can be obtained as

$$  \hat{XY}_{i} = \hat{XY}_{i} + \beta (XY_{i}-\hat{XY}_{i}) + \gamma (\hat{XY}_{i-1} - 2 \hat{XY}_{i} + \hat{XY}_{i+1}) $$

For i going from 2 to \\\( N-1 \\) where N is then number of points. It is however possible that the resulting smooth trajectory crosses through an obstacle, one way to avoid this is to include distance from obstacle as an explicit constraint in your optimization and use nonlinear solvers, or create a 'buffer' zone around the obstacle. The choice of buffer zone is a design choice a control designer has to make. Once a smooth trajectory is obtained, it can be followed using [the pole-placement tracking method](https://mec560sbu.github.io/2016/09/19/Control_synthesis/). For the car robot itself,  assume that our autonomous robot is a point mass \\( (F = Ma,~ \ddot{x} = u_x, ~ \ddot{y} = u_y) \\) that can move freely in the XY-plane. No turning constraint or angle constraints are imposed on the robot model for this project. However, it is assumed that the maximum and minimum controls of the robot cannot exceed -1 and 1, and the velocity of the robot cannot exceed -2 and 2. Further assume that only position is available as a measurement, therefore you need to design an observer to estimate the states of the system. Once system estimates are available, Pole-placement technique can be used to design a PID-type control that drives the car along the given trajectory. 

#### Project Tasks

Submit at least 5-6 page report that details how you solved the obstacle avoidance problem, and all the accomplanying code. Please make sure to present links,references or numerical results that corraborate your statements and model choices. Inaccurate general declarations will result in deduction of points. Your report should address the following questions, and these questions will be used as a ruberic to score your work (each item is worth 20 points). 

1. Construct a 10 X 10 grid with discretizations at 0.1 along X- and Y-, with atleast 2 obstacles between the start and end. Assume that the start position is (1,1) and end position is (9,9) on the grid. Assume that at each instant, the next way point can be at left-right, top-bottom or at 45,135,-45 and -135 degree angles. *Note: Additional 10% bonus if you make a maze and your robot is able to navigate the maze without running into walls. *
2. Apply dynamic programming or reinforcement learning to generate a sequence of waypoints. Which algorithm did you end up using, and why? *Note: A popular algorithm of choice for self-driving cars is A-star algorithm. Additional 10% bonus if you apply this method for obstacle avoidance. *
3. Generate smooth trajectory from the waypoints generated in part 2. You may use either the gradient descent with buffer or full nonlinear program to get a smooth trajectory that avoids obstacles. If you chose the  gradient descent with buffer method, explain how you chose the buffer size. Show the effect of changing buffer size on the clearance between obstacle and the car. If you chose the gradient descent method, \\( \beta =0.5 \\) and \\( \gamma = 0.1\\). If you chose the full nonlinear progamming method, use \\( \alpha =0.2 \\). Once a smooth path is obtained, convert it into trajectory information (time-vs-position), so it can be followed using speed of 1.5 units. Note, diagonal paths are longer, so a uniform velocity solution will result in longer time along diagonals. *Note: The trajectory generated here does not start from zero, nor ends at 0. It is possible to include these as constraints, and solve a nonlinear program using direct collocation, multiple shooting or dynamic programming. This results in feasible control. Additional 10% bonus if your trajectory obeys the problem constraints. *
4. Design an observer to estimate the states of your robot. Which observer model did you end up choosing, and why? How did you test your observer model? 
5. Pole-placement technique can be used to track a time-dependent trajectory. Choose locations of poles and explain why you chose them there. Is this controller optimal? If so, how? If you used LQR technique to obtain the gain matrix for your controller, explain how you chose these Q and R values? Does your resulting controller restrict the control signal between -1 and 1, and velocity between -2 and 2? If not, how could you improve the controller design in a future iteration of this project?

Projects that meet the above criteria and desmonstrate technical prowess will be featured on the course page.

### Ethics statement:

All the work presented for the mini-project should be your own. All the codes must be written by you. You may download codes from google drive, and modify them for your purpose. No code sharing or copying of work among students is allowed. If you are found copying or offering your material to other students, you will get a 0 on the mini-project. 




