from dataclasses import dataclass

import numba
import numpy as np

from numba import cuda, types
from numba.core.extending import (
    make_attribute_wrapper,
    models,
    register_model,
    typeof_impl,
)
from numba.core.typing.templates import AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_registry


def gpudataclass(this: type):
    this = dataclass(this)
    
    class ThisType(types.Type):
        def __init__(self):
            super().__init__(name=this.__name__)

    this_type = ThisType()

    @typeof_impl.register(this)
    def typeof_this(val, c):
        return ThisType()

    class ThisAttrsTemplate(AttributeTemplate):
        pass
    
    for name in (fields := this.__dataclass_fields__):
        typ = fields[name].type
        def resolver(self, this):
            return numba.from_dtype(typ)
        setattr(ThisAttrsTemplate, f"resolve_{name}", resolver)

    # Register the typing for Pixel attributes with Numba.
    @cuda_registry.register_attr
    class ThisAttrs(ThisAttrsTemplate):
        key = this_type

    @register_model(ThisType)
    class ThisModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            fields = this.__dataclass_fields__
            members = [
                (name, numba.from_dtype(fields[name].type))
                for name, 
                in fields
            ]
            super().__init__(dmm, fe_type, members)

    for name in this.__dataclass_fields__:
        make_attribute_wrapper(ThisType, name, name)
        
    return this
