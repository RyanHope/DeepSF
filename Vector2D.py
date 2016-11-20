#This class represents a basic 2 dimensional linear algebra vector.
from __future__ import division
import math


class Vector2D(object):
    def __init__(self, x=0, y=0):
        object.__init__(self)
        self.x = x
        self.y = y

    def copy(self):
        n = Vector2D(self.x,self.y)
        return n

    def __isub__(self,v):
        self.x -= v.x
        self.y -= v.y
        return self

    def __iadd__(self,v):
        self.x += v.x
        self.y += v.y
        return self

    def __sub__(self,v):
        n = self.copy()
        n -= v
        return n

    def __add__(self,v):
        n = self.copy()
        n += v
        return n

    def __mul__(self,v):
        n = self.copy()
        n.x *= v.x
        n.y *= v.y
        return n

    def norm(self):
        '''Returns the norm of the vector'''
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        '''Normalizes the current vector'''
        norm = self.norm()
        self.x /= norm
        self.y /= norm

    def normal(self):
        '''Returns a normalized vector'''
        norm = self.norm()
        x = self.x/norm
        y = self.y/norm
        return Vector2D(x, y)

    def scalar_product(self, scalar):
        '''Returns the scalar product'''
        x = self.x * scalar
        y = self.y * scalar
        return Vector2D(x, y)

    def dot_product(self, hVector):
        '''Returns the dot product'''
        return (self.x * hVector.x) + (self.y * hVector.y)

    def projection(self, hVector):
        '''Returns the projection of a vector onto the current one'''
        return self.scalar_product(self.dot_product(hVector)/self.dot_product(self))

    def cosine (self, hVector):
        '''Returns the Cosine of the angle between the two vectors'''
        return self.dot_product(hVector) / (self.norm() * hVector.norm())

    def angle(self):
        '''Return the angle of the vector in radians.'''
        angle = math.atan2(self.y, self.x)
        if angle < 0:
            angle += 2*math.pi
        return angle

    @classmethod
    def parseVector(cls, hSource, hTarget):
        '''Returns the vector between two points'''
        X = hTarget.x - hSource.x
        Y = hTarget.y - hSource.y
        return  cls(X, Y)

if __name__ == '__main__':
    v = Vector2D(1, 1)
    print v
