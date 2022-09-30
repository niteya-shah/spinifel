import os
import subprocess

# Use this to wrap python in jsrun cmd 
dirname = os.path.dirname

class Test:
    test_dir = dirname(os.path.realpath(__file__))
    launch_args = os.environ["SPINIFEL_TEST_LAUNCHER"].split()
    
    def test_FSC(self, ):
        args = self.launch_args + ['python', os.path.join(self.test_dir,'test_FSC.py')]
        subprocess.check_call(args)
    
