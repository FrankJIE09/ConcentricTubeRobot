# ConcentricTubeRobot

Concentric Tube Continuum Robot distal-end tip position controller cascaded with BVP controller.

Model based on [A geometrically exact model for externally loaded concentric-tube continuum robots](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3091283/)

## Documentation

1. controller.py

    Main code for CTR system. Calls CTR model and trajectory generator.
    
    Also contains classes for Jacobian Linearisation and Controller.

2. CTR_model.py

    Model for a three-tubed concentric tube continuum robot class.
    
3. CurvatureController.py

    End curvature BVP controller.

4. TrajectoryGenerator.py

    Generates a quintic\quadratic\cubic polynomial, or linear trajectory.


## Requirements

- Python 3.7.x (2.7 is not supported)

- pathos

    For multiprocessing
    
- numpy

- matplotlib

- scipy

## How to use

1. Clone the repo.

> git clone https://github.com/FrankJIE09/ConcentricTubeRobot.git

> cd ConcentricTubeRobot/

2. Install required, and missing libraries.

3. Execute python script.

## Demo Video

Check out this demonstration video to see the three-tube concentric tube continuum robot model in action:

 [Demo1 Forward Kinematics](https://www.bilibili.com/video/BV1hm421L78c/)

 [Demo2 Inverse Kinematics with Jacobian](https://www.bilibili.com/video/BV1gi421i7Ta/)

 [Demo3 Inverse Kinematics with MPC](https://www.bilibili.com/video/BV1eM4m1U7eC/)

## License

MIT

