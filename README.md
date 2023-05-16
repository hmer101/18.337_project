# 18.337_project

Final project for MIT's class 18.337 - Parallel Computing and Scientific Machine Learning in Spring '23. The project explores using Universal Differential Equations (UDE) to model cable tension vectors acting on a quadrotor and a load in multi-quadrotor slung load system.


## Organization
<ul>
  <li>main.jl - Contains the main code to run tests and visualizations. Also contains the core UDE system.</li>
  <li>generateData.jl - Code to generate training data based on Euler-Newton kinematics equations for the multi-quadrotor slung load system.</li>
  <li>solveTrain.jl - Helper functions to solve the UDE and train the embedded neural network. Includes loss functions.</li>
  <li>plotting.jl - Functions to plot results.</li>
  <li>datatypes.jl - Contains a mutable structure ('DroneSwarmParams') to store useful information related to the core UDE.</li>
  <li>utils.jl - A variety of helpful utility functions for vector and matrix manipulation, and kinematics.</li>
</ul> 
