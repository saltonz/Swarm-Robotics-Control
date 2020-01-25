## Running a simulation

In the python directory, type 
``` 
python run_simulation.py 
```

## Video of Simulation
https://www.youtube.com/watch?v=57zwhvkJIPQ

## Description
Swarm Robotics control.

The goal contains formation control, consensus, moving object to targets. Using consensus protocol with leader, artificial potential field to achieve the goals. Another approach is to use rigidity matrix.

The frequency of controller is 60 Hz 

Details are introduced in the report paper.

## Requirements

>* Python 3.7
>* PyBullet https://github.com/bulletphysics/bullet3/tree/master/docs



Install PyBullet

```
pip install pybullet
```

```
conda install -c hcc pybullet
```

## Files

**/models/**

> Resources

**/python/**

> Code

* Swarm_simulation.py :

  Define the simulation world.

* Run_simulation.py:

  Runner of simulation

* Robot.py

  The definition of robot. Along with control law.