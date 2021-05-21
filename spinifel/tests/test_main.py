import os
import subprocess

# Use this to wrap python in jsrun cmd 
dirname = os.path.dirname

class Test:
    root_dir = dirname(dirname(os.path.realpath(__file__))) # spinifel repo root
    launch_args = "jsrun -n1".split()
    
    def test_skopi(self, ):
        args = self.launch_args + ['python','skopi_quaternion.py']
        subprocess.check_call(args)


