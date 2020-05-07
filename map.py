# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.9)
action2rotation = [0,5,-5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

# textureMask = CoreImage(source="./kivytest/simplemask1.png")


# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global patch
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = 1420
    goal_y = 622
    first_update = False
    global swap
    swap = 0
    patch = np.ones((80, 80))


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 10.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 10.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 10.

#        patch = np.ones((80, 80))
#        # take out a patch from self.car.x - 40 or 0 to self.car.x + 40 or longueur in x axis
#        if int(self.x) < 40:
#            src_fromX = 0
#            tgt_fromX = 40 - int(self.x)
#        else:
#            src_fromX = int(self.x) - 40
#            tgt_fromX = 0
#
#        if (int(self.x) + 40) > (longueur):
#            src_toX = (longueur)
#            tgt_toX = 40 + (longueur - int(self.x))
#        else:
#            src_toX = int(self.x) + 40
#            tgt_toX = 80
#
#        # take out a patch from self.car.y - 40 or 0 to self.car.y + 40 or largeur in y axis
#        if int(self.y) < 40:
#            src_fromY = 0
#            tgt_fromY = 40 - int(self.y)
#        else:
#            src_fromY = int(self.y) - 40
#            tgt_fromY = 0
#
#        if (self.y + 40) > (largeur):
#            src_toY = largeur
#            tgt_toY = 40 + (largeur - int(self.y))
#        else:
#            src_toY = int(self.y) + 40
#            tgt_toY = 80
#
#        patch[int(tgt_fromX):int(tgt_toX), int(tgt_fromY):int(tgt_toY)] = sand[int(src_fromX):int(src_toX), int(src_fromY):int(src_toY)]
        #patchimg = PILImage.fromarray(patch.astype("uint8") * 255)
        #patch_name = "./images/" + "patch_" + str(int(self.x)) + "_" + str(int(self.y)) + ".jpg"
        #patchimg.save(patch_name)


class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global patch
        

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        #last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]

        #patchimg = PILImage.fromarray(patch.astype("uint8") * 255)
        #patch_name = "./images/" + "patch_" + str(int(self.car.x)) + "_" + str(int(self.car.y)) + ".jpg"
        #patchimg.save(patch_name)

        last_signal = patch.reshape((1, 80, 80))

        action = brain.update(last_reward, last_signal)
        print(action)
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)

        patch = np.ones((80, 80))
        # take out a patch from self.car.x - 40 or 0 to self.car.x + 40 or longueur in x axis
        if int(self.car.x) < 40:
            src_fromX = 0
            tgt_fromX = 40 - int(self.car.x)
        else:
            src_fromX = int(self.car.x) - 40
            tgt_fromX = 0

        if (int(self.car.x) + 40) > (longueur):
            src_toX = (longueur)
            tgt_toX = 40 + (longueur - int(self.car.x))
        else:
            src_toX = int(self.car.x) + 40
            tgt_toX = 80

        # take out a patch from self.car.y - 40 or 0 to self.car.y + 40 or largeur in y axis
        if int(self.car.y) < 40:
            src_fromY = 0
            tgt_fromY = 40 - int(self.car.y)
        else:
            src_fromY = int(self.car.y) - 40
            tgt_fromY = 0

        if (self.y + 40) > (largeur):
            src_toY = largeur
            tgt_toY = 40 + (largeur - int(self.car.y))
        else:
            src_toY = int(self.car.y) + 40
            tgt_toY = 80

        patch[int(tgt_fromX):int(tgt_toX), int(tgt_fromY):int(tgt_toY)] = sand[int(src_fromX):int(src_toX), int(src_fromY):int(src_toY)]
        #patchimg = PILImage.fromarray(patch.astype("uint8") * 255)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = -0.2
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < last_distance:
                last_reward = 0.1
            # else:
            #     last_reward = last_reward +(-0.2)

        if self.car.x < 5:
            self.car.x = 5
            last_reward = -1
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            last_reward = -1
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -1
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -1

        if distance < 25:
            if swap == 1:
                goal_x = 1420
                goal_y = 622
                swap = 0
            else:
                goal_x = 9
                goal_y = 85
                swap = 1
        last_distance = distance

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        global patch
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))
        patch = np.ones((80, 80))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
