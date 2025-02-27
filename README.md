# SEDS
Texas A&M Students for the Exploration and Development of Space (SEDS), NASA Lunar Autonomy Challenge (LAC) Team

# Challenge Context

 - [Challenge Documentation](https://lunar-autonomy-challenge.jhuapl.edu/Challenge-Documentation)
 - [Challenge Guidebook](https://lunar-autonomy-challenge.jhuapl.edu/Challenge-Information/2024_Lunar-Autonomy-Challenge-Guidebook-v2.pdf)
 - [Introduction Video](https://www.youtube.com/watch?v=psZINZ88Khc&feature=youtu.be)

# Getting Started

## Download the zip file

Download the zip from [this link](https://lac-content.s3.us-west-2.amazonaws.com/LunarAutonomyChallenge.zip)

Extract contents

## Clone the Repo

Clone this repo into the main LunarAutonomyChallenge directory

# Current works:

## Agent making
 - [challenge docs](https://lunar-autonomy-challenge.jhuapl.edu/Challenge-Documentation/index.php#creating_an_agent
 )
 - [agent filepath](/TAMU_SEDS_LAC/Lunatyx_agent.py)

## Mapping
 - CV stereovision -> photogrammatry
 - SLAM options
  - [ORB-slam -> Stereo vision](https://github.com/UZ-SLAMLab/ORB_SLAM3) (I was having some issues installing it)
  - [lightweight vio/vslam](https://github.com/Gongsta/vSLAM-py)
  - [pyslam](https://github.com/luigifreda/pyslam)

## Rock finding 
 - [opencv docs](https://answers.opencv.org/question/92133/detection-of-stones-rocks-on-field-surface/)

# Deliverables:

## 2/28
 - Agent that can map 9mx9m area 
    - must navigate the area in reasonable time
 - exports csv of 15cmx15cm cells containing 
    - height of each cell
    - presence of rock (boolean)

