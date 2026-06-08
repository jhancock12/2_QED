# Standard libraries

# Local modules

# Third-party libraries
import numpy as np

TOL = 0.0000001

def smart_round(number, dec_places):
    if isinstance(number, dict):
        for key in list(number):
            number[key] = smart_round(number[key], dec_places)
        return number
    elif isinstance(number, list) or isinstance(number, np.ndarray):
        for k in range(len(number)):
            number[k] = smart_round(number[k], dec_places)
        return number
    else:
        re = 0.0 if abs(number.real) < 1e-8 else number.real
        im = 0.0 if abs(number.imag) < 1e-8  else number.imag

        re = round(re, dec_places)
        im = round(im, dec_places)

        if im == 0.0:
            return float(re)
        return complex(re, im)
    
def dict_print(dictionary : dict):
    if not isinstance(dictionary, dict): raise TypeError(f"dictionary must be a dict, you have entered a {type(dictionary)}")

    for key in dictionary:
        print(f"{key} : {dictionary[key]}")