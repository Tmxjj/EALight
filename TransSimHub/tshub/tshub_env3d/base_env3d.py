from abc import ABC, abstractmethod

class BaseSumoEnvironment3D(ABC):

    @abstractmethod
    def reset(self):
        raise NotImplementedError
    
    @abstractmethod
    def step(self):
        raise NotImplementedError