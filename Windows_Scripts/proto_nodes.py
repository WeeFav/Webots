import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class RoadPROTO:
    def __init__(self, data=None):
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 1, 0]
        self.id = ""
        self.name = "road"
        self.width = 7.0
        self.wayPoints = [
            [0, 0, 0],
            [1, 0, 0]
        ]
        self.appearance = ""

        if data:
            self.update_from_dict(data)

    def update_from_dict(self, data: dict):
        for key, value in data.items():
            if hasattr(self, key):
                if key in ["startingAngle", "endingAngle"]:
                    setattr(self, key, [value])
                else:
                    setattr(self, key, value)

    def to_dict(self):
        return self.__dict__

class CrossroadPROTO:
    def __init__(self, data=None):
        self.translation = [0, 0, 0]
        self.rotation = [0, 1, 0, 0]
        self.id = ""
        self.name = "crossroad"
        self.shape = [[0, 0, 0], [0, 0, 1], [1, 0, 0]] 
    
        if data:
            self.update_from_dict(data)

    def update_from_dict(self, data: dict):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return self.__dict__
    
class ForestPROTO:
    def __init__(self, data=None):
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 1, 0]
        self.id = ""
        self.name = "forest"
        self.shape = [[-20, -10], [20, -10], [0, 25]]
            
        # --- override with dict ---
        if data:
            self.update_from_dict(data)

    def update_from_dict(self, data: dict):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return self.__dict__
    
class SimpleBuildingPROTO:
    def __init__(self, data=None):
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 1, 0]
        self.id = ""
        self.name = "forest"
        self.corners = [[-20, -10], [20, -10], [0, 25]]
            
        # --- override with dict ---
        if data:
            self.update_from_dict(data)
            
        self.id = self.id.replace("(", "").replace(")", "")

    def update_from_dict(self, data: dict):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return self.__dict__
    
class TransformPROTO:
    def __init__(self, data=None):
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 1, 0]
        self.id = ""
        self.name = "transform"
        self.point = [[-20, -10], [20, -10], [0, 25]]
            
        # --- override with dict ---
        if data:
            self.update_from_dict(data)
            
        if self.id == "":
            self.id = self.name.replace("(", "").replace(")", "")

    def update_from_dict(self, data: dict):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return self.__dict__
    

