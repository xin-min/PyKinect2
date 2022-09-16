from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np

import ctypes
import _ctypes
import pygame
import sys
import time
import csv
from datetime import datetime

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.color.THECOLORS["red"], 
                  pygame.color.THECOLORS["blue"], 
                  pygame.color.THECOLORS["green"], 
                  pygame.color.THECOLORS["orange"], 
                  pygame.color.THECOLORS["purple"], 
                  pygame.color.THECOLORS["yellow"], 
                  pygame.color.THECOLORS["violet"]]

KINECT_JOINTS = [
PyKinectV2.JointType_Head,
PyKinectV2.JointType_Neck, 
PyKinectV2.JointType_SpineShoulder,
PyKinectV2.JointType_SpineMid, 
PyKinectV2.JointType_SpineBase, 
PyKinectV2.JointType_ShoulderRight, 
PyKinectV2.JointType_ShoulderLeft, 
PyKinectV2.JointType_HipRight, 
PyKinectV2.JointType_HipLeft, 
PyKinectV2.JointType_ElbowRight, 
PyKinectV2.JointType_WristRight, 
PyKinectV2.JointType_HandRight, 
PyKinectV2.JointType_HandTipRight, 
PyKinectV2.JointType_ThumbRight, 
PyKinectV2.JointType_ElbowLeft, 
PyKinectV2.JointType_WristLeft, 
PyKinectV2.JointType_HandLeft, 
PyKinectV2.JointType_HandTipLeft, 
PyKinectV2.JointType_ThumbLeft, 
PyKinectV2.JointType_KneeRight, 
PyKinectV2.JointType_AnkleRight, 
PyKinectV2.JointType_FootRight, 
PyKinectV2.JointType_KneeLeft, 
PyKinectV2.JointType_AnkleLeft, 
PyKinectV2.JointType_FootLeft
]


HEADER = [
"datetime",
"Head",
"Neck",
"SpineShoulder",
"SpineMid",
"SpineBase",
"ShoulderRight",
"ShoulderLeft",
"HipRight",
"HipLeft",
"ElbowRight",
"WristRight",
"HandRight",
"HandTipRight",
"ThumbRight",
"ElbowLeft",
"WristLeft",
"HandLeft",
"HandTipLeft",
"ThumbLeft",
"KneeRight",
"AnkleRight",
"FootRight",
"KneeLeft",
"AnkleLeft",
"FootLeft"]


class BodyGameRuntime(object):
    def __init__(self):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Body Game")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Depth)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data 
        self._bodies = None

        # visualise the skeleton data
        self._canvas = np.zeros((self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 3), np.uint8)




    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good 
        
        # print(str(start))
        # print(str(end))
        # print("drawing")
        # cv2.line(self._canvas, start, end, color, 8) 

##################################################
        try:
            start = (int(jointPoints[joint0].x), int(jointPoints[joint0].y))
            end = (int(jointPoints[joint1].x), int(jointPoints[joint1].y))
            cv2.line(self._canvas, start, end, color, 8) 
            # cv2.line(image, start_point, end_point, color, thickness)
            # pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color, depth_points):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);
    
        # Right Arm    
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

        # Left Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);
        
        # now = datetime.now()
        # current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
        now = str(datetime.now())
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(self._canvas, now, (20, 40), font, 2, (255, 255, 255), 2)
        row = [now]
        for each_joint in KINECT_JOINTS:
            
            row.append((jointPoints[each_joint].x, jointPoints[each_joint].y))
            # ---------- depth
            # depth_x = depth_points[each_joint].x
            # depth_y = depth_points[each_joint].y
            # print((depth_x, depth_y, len(self._depth)))
            # depth_z = self._depth[int(depth_y * 512 + depth_x)]
            # row.append((depth_x,depth_y,depth_z))
        self._writer.writerow(row)


    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def run(self):

        # -------- Set up camera and video file -----------
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 30.0, (self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height))
        out2 = cv2.VideoWriter('output2.avi', fourcc, 30.0, (self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height))
        
        # -------- Set up text file -----------
        f = open("skeleton_data.csv", "a", newline='')
        self._writer = csv.writer(f)
        self._writer.writerow(HEADER)
        time.sleep(3)

        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop
                    out.release()
                    out2.release()
                    f.close()


                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    
            # --- Game logic should go here

            # --- Camera stuff ADDEDDDDDDD
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # out = cv2.VideoWriter('output.avi', fourcc, 30.0, (self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height))
            # print(self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height)
            # 1920x1080
            # --- Getting frames and drawing  
            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data 
            
            temp = pygame.time.get_ticks()

            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                # print(len(frame))
                self.draw_color_frame(frame, self._frame_surface)
                # frame = frame.reshape((2073600,4))
                frame = frame.reshape((1080,1920,4))

                frame = frame[:,:,0:3].astype('uint8')
                # np.flip(frame, axis=-1) 
                # print(frame.shape)
                ##################################### to be removed ##################
                # cv2.line(frame, (0,0), (temp%1080,temp%1080), (100, 0, 50), 8) 
                # cv2.line(frame, (1920,0), ((1920-temp)%1080,temp%1080), (50, 0, 100), 8)
                ##################################### to be removed ##################

                out.write(frame)
                self._canvas = frame
                frame = None

            # --- Cool! We have a body frame, so can get skeletons
            if self._kinect.has_new_body_frame(): 
                self._bodies = self._kinect.get_last_body_frame()

            # --- draw skeletons to _frame_surface (first initialise new frame)
            # self._canvas = np.zeros((self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 3), np.uint8)
            if self._bodies is not None: 
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked: 
                        continue 
                    
                    joints = body.joints 
                    
                    ##################################### to be edited ##################
                    # f.write(str(joints))

                    # convert joint coordinates to color space 
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    depth_points = self._kinect.body_joints_to_depth_space(joints)
                    self._depth = self._kinect.get_last_depth_frame()
                    # f.write(str(joint_points))

                    self.draw_body(joints, joint_points, SKELETON_COLORS[i],depth_points)

            # temp = pygame.time.get_ticks()

            ##################################### to be removed ##################
            # cv2.line(self._canvas, (0,0), (temp%1080,temp%1080), (100, 0, 50), 8) 
            # cv2.line(self._canvas, (1920,0), (1920-temp,temp), (50, 0, 100), 8)
            ##################################### to be removed ##################

            out2.write(self._canvas.astype('uint8'))

            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size) 
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0,0))
            surface_to_draw = None
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(60)
            print(self._clock.get_fps())

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Body Game"
game = BodyGameRuntime();
game.run();

