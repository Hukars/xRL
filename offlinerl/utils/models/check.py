from typing import List, Optional, Union, cast

from .encoders import create_encoder_factory, EncoderFactory
from .q_functions import create_q_func_factory, QFunctionFactory

EncoderArg = Union[EncoderFactory, str]
QFuncArg = Union[QFunctionFactory, str]

def check_encoder(value: EncoderArg) -> EncoderFactory:
    """Checks value and returns EncoderFactory object.
    Returns:
        d3rlpy.encoders.EncoderFactory: encoder factory object.
    """
    if isinstance(value, EncoderFactory):
        return value
    if isinstance(value, str):
        return create_encoder_factory(value)
    raise ValueError("This argument must be str or EncoderFactory object.")
    
    
def check_q_func(value: QFuncArg) -> QFunctionFactory:
    """Checks value and returns QFunctionFactory object.
    Returns:
        d3rlpy.q_functions.QFunctionFactory: Q function factory object.
    """
    if isinstance(value, QFunctionFactory):
        return value
    if isinstance(value, str):
        return create_q_func_factory(value)
    raise ValueError("This argument must be str or QFunctionFactory object.")