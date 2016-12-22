# Sliding Window MSCKF
**Authors:** Doom

**!!!!NOTICE!!!!**
This implementation is still under constrution. Please DO NOT fork yet! 

MSCKF (Multi-State Constraint Kalman Filter) is a well known Visual Inertial Navigation System (VINS) method, proposed by Anastasios I. Mourikis et al. I have great interest in using this method in my own project, but there are no available code yet. So I decide to create a complete implementation of this method using C++. However, the most tough part of MSCKF is the feature/keyframe culling, in order to make this code working efficently and easily, I use a sliding window instead. The propagation and update part of our method using the derivation of MARS's quaternion report.

### Related Papers
* **A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation**, *Anastasios I. Mourikis and Stergios I. Roumeliotis*, ICRA '07
* **Indirect Kalman Filter for 3D Attitude Estimation A Tutorial for Quaternion Algebra**, *Nikolas Trawny and Stergios I. Roumeliotis*, Department of Computer Science & Engineering University of Minnesota

### Sensor
ZED + Xsens
![image](https://github.com/doomzzju/slidingWindow_msckf/blob/master/sensor.jpg)

## Change Log
### v0.1 (2016/12/20)
* Create this repo.
* Create the basic framework.

### v0.2 (2016/12/21)
* Create frame and feature classes.
* Create imu and some utils classes.
* Update CMakeLists.txt

### v0.3 (2016/12/22)
* Fix some bugs of frame class.
* Create sliding window algorithm classs.
* Achieve the propagation part of filter as the derivation of quaternion report @ UMN.

