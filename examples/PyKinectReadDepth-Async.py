from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
import asyncio

import ctypes
import _ctypes
import pygame
import sys

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


class BodyGameRuntime(object):
    def __init__(self):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        # self._infoObject = pygame.display.Info()
        # self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
        #                                        pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        # pygame.display.set_caption("Kinect for Windows v2 Body Game")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data 
        self._bodies = None

        # visualise the skeleton data
        self._canvas = np.zeros((self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 3), np.uint8)

        # queue incoming frames
        self._queue = []
        self._jointqueue = []




    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1, frame):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good 
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)
##################################################
        try:
            # self._jointqueue.append((start,end))

            frame = cv2.line(frame, start, end, color, 8) 
            return frame

            # cv2.line(image, start_point, end_point, color, thickness)
            # pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass


    def draw_body(self, joints, jointPoints, color, frame):
        # Torso
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft, frame);
    
        # Right Arm    
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight, frame);

        # Left Arm
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft, frame);

        # Right Leg
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight, frame);

        # Left Leg
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft, frame);
        frame = self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft, frame);


    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    async def save_video_empty(self, loop):
        frame = self._queue.pop(0)
        self._out.write(frame)
        annotateVideoTask = loop.create_task(self.save_video(frame, loop))
        # annotateVideoTask.Run()
        await annotateVideoTask

        # loop.run_until_complete(annotateVideoTask)
        # loop_annotate = asyncio.new_event_loop()
        # loop_annotate.run_until_complete(self.save_video(frame))

    async def save_video(self,frame, loop):
        temp = pygame.time.get_ticks()
        cv2.line(frame, (0,0), (temp%1080,temp%1080), (100, 0, 50), 8) 
        cv2.line(frame, (1920,0), ((1920-temp)%1080,temp%1080), (50, 0, 100), 8)
        # cv2.line(frame, start, end, color, 8) 
        if len(self._jointqueue)>0:
            (joints, joint_points) = self._jointqueue.pop(0)
            print("going to draw body")
            frame = self.draw_body(joints, joint_points, SKELETON_COLORS[i], frame)
        self._out2.write(frame)


    async def run(self, loop):

        # -------- Set up camera and video file -----------
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._out = cv2.VideoWriter('output.avi', fourcc, 30.0, (self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height))
        self._out2 = cv2.VideoWriter('output2.avi', fourcc, 30.0, (self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height))
        loop1 = asyncio.new_event_loop()
        colorframe_present = 0

        
        # -------- Set up text file -----------
        f = open("skeleton_data.txt", "a")


        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            # for event in pygame.event.get(): # User did something
            #     if event.type == pygame.key.get_pressed():
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                    if event.mod & pygame.KMOD_LSHIFT:
                        # if event.type == pygame.QUIT: # If user clicked close
                        self._done = True # Flag that we are done so we exit this loop
                        counter = 0
                        # while loop.is_running()==True:
                        #     print(counter)
                        #     counter +=1
                        self._out.release()
                        self._out2.release()
                        f.close()      


            # elif event.type == pygame.VIDEORESIZE: # window resized
            #     self._screen = pygame.display.set_mode(event.dict['size'], 
            #                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    
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
                colorframe_present = 1
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
                self._queue.append(frame)
                # loop1.run_until_complete(self.save_video_empty())
                saveVideoTask = loop.create_task(self.save_video_empty(loop))
                await saveVideoTask
                # saveVideoTask.Run()
                # loop.run_until_complete(saveVideoTask)

                # out.write(frame)
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
                    f.write(joints)

                    # convert joint coordinates to color space 
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    self._jointqueue.append(joints, joint_points)
                    # self.draw_body(joints, joint_points, SKELETON_COLORS[i])
            else:
                if colorframe_present == 1:
                    self._jointqueue.append([0])

            colorframe_present = 0


            # temp = pygame.time.get_ticks()

            # ##################################### to be removed ##################
            # cv2.line(self._canvas, (0,0), (temp%1080,temp%1080), (100, 0, 50), 8) 
            # cv2.line(self._canvas, (1920,0), (1920-temp,temp), (50, 0, 100), 8)
            ##################################### to be removed ##################


            # out2.write(self._canvas.astype('uint8'))

            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size) 
            # h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            # target_height = int(h_to_w * self._screen.get_width())
            # surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            # self._screen.blit(surface_to_draw, (0,0))
            # surface_to_draw = None
            # pygame.display.update()

            # # --- Go ahead and update the screen with what we've drawn.
            # pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(60)
            print(self._clock.get_fps())

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Body Game"
game = BodyGameRuntime()
loop = asyncio.new_event_loop()
loop.run_until_complete(game.run(loop))
# asyncio.run(game.run());

