# A Target Detection and Autonomous Landing for MAVs

In this work, we address the problem of target detection involved in an autonomous landing task for a Micro Aerial Vehicle (MAV). The challenge is to detect a flag located somewhere in the environment. The flag is posed on a pole, and to its right, a landing platform is located. Thus, the MAV has to detect the flag, fly towards it and once it is close enough, locate the landing platform nearby, aiming at centring over it to perform landing; all of this has to be carried out autonomously. In this context, the main problem is the detection of both the flag and the landing platform, whose shapes are known in advanced. Traditional computer vision algorithms could be used; however, the main challenges in this task are the changes in illumination, rotation and scale, and the fact that the flight controller uses the detection to perform the autonomous flight; hence the detection has to be stable and continuous on every camera frame. Motivated by this, we propose to use a Convolutional Neural Network optimised to be run on a small computer with limited computer processing budget. The MAV carries this computer, and it is used to process everything on board. To validate our system, we tested with rotated images, changes in scale and the presence of low illumination. Our method is compared against two conventional computer vision methods, namely, template and feature matching. In addition, we tested our system performance in a wide corridor, executing everything on board the MAV. We achieved a successful detection of the flag with a confidence metric of 0.9386 and 0.9826 for the Landing platform. In total, all the onboard computations ran at an average of 13.01 fps.

# Overview of our approach

![](/images/overview.png)

It consist of 3 steps: (1) Flag detection using the learning model and drone's onboard camera; (2) Autonomous landing platform search using the learning model; (3) Autonomous landing. 

# SSD7 Architecture

The SSD7 architecture consists of 7 convolutional layers, BatchNormalization, and Anchors shape in output.

![](/images/ssd7.png)

# Video

A video of this approach can be watched at: https://youtu.be/sYn9mo-2hvA

![](/images/test.gif)

# Recommended System

- Ubuntu 16.04
- ROS Kinetic Kame
- Python 2.7.12
- Cuda 9 or 10
- Cudnn 7 or 8
- Tensorflow 1.4.0
- Keras 2.1.4

# Dataset to train SSD7

https://mnemosyne.inaoep.mx/index.php/s/xHmiNVFF8xl7SeE

## Reference

If you use our dataset, models or code, please cite the following reference:

Cabrera-Ponce A.A., Martinez-Carranza J. (2020) Onboard CNN-Based Processing for Target Detection and Autonomous Landing for MAVs. MCPR 2020. Lecture Notes in Computer Science, vol 12088. Springer, Cham. https://doi.org/10.1007/978-3-030-49076-8_19

```
@inproceedings{cabrera2020onboard,
  title={Onboard CNN-Based Processing for Target Detection and Autonomous Landing for MAVs},
  author={Cabrera-Ponce, Aldrich A and Martinez-Carranza, Jose},
  booktitle={Mexican Conference on Pattern Recognition},
  pages={195--208},
  year={2020},
  organization={Springer}
}
```

# Acknowledgements

The first author is thankful to Consejo Nacional de Ciencia y Tecnología (CONACYT) for the scholarship No. 924462. We are also thankful to Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE), for allow us to use their facilities and financial support.
