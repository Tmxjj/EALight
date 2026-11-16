import abc
from typing import Any, Dict

class BaseSensor(metaclass=abc.ABCMeta):
    """The sensor base class.
    """
    def step(self, tshub_obs: Dict[str, Any], **kwargs):
        """Update sensor state (根据 TSHub 返回值更新 sensor).
        """
        pass

    @abc.abstractmethod
    def teardown(self, **kwargs):
        """Clean up internal resources
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @property
    def mutable(self) -> bool:
        """If this sensor mutates on call."""
        return True

    @property
    def serializable(self) -> bool:
        """If this sensor can be serialized."""
        return True