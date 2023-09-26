import rpyc
import math
from Method.util import vtkTeeth,vtkMeanTeeth, ToothNoExist
import vtk
import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import sys
import io

class MyService(rpyc.Service):
    def exposed_execute_function(self, f, *args, **kwargs):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        new_stdout = io.StringIO()
        new_stderr = io.StringIO()
        sys.stdout = new_stdout
        sys.stderr = new_stderr
        
        result = None
        try:
            result = f(*args, **kwargs)
        except Exception as e:
            sys.stderr.write(str(e))
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        output = new_stdout.getvalue()
        errors = new_stderr.getvalue()
        
        new_stdout.close()
        new_stderr.close()
        
        return result, output, errors
    
    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    def exposed_add_function(self, func_name, func_code):
        exec(func_code, globals())
        if not func_name == "imports":  # Ne pas ajouter "imports" comme une fonction
            setattr(self, f'exposed_{func_name}', eval(func_name))

    def exposed_exec_code(self, code):
        try:
            exec(code)
            return True
        except Exception as e:
            return str(e)


if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(MyService, port=18812)
    t.start()
