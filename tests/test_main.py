import os
import subprocess

# Use this to call run_summit_mult.sh script that setups
# the environment and calls test sripts.
dirname = os.path.dirname

class Test:
    root_dir = dirname(dirname(os.path.realpath(__file__))) # spinifel repo root
    launch_cmds = ["bash", os.path.join(root_dir,'scripts', 'run_summit_mult.sh')]
    launch_args = "-m -n 1 -a 1 -g 1 -r 1 -d 1 -c -f -l".split()
    
    def test_orientation_matching(self, ):
        # set root_dir for run_summit_mult.sh
        os.environ["root_dir"] = self.root_dir
        os.environ["test_data_dir"] = "/gpfs/alpine/proj-shared/chm137/data/testdata"
        test_script = "tests/orientation_matching.py" 
        args = self.launch_cmds + self.launch_args + [test_script]
        subprocess.check_call(args)

    def test_forward_and_adjoint(self, ):
        # set root_dir for run_summit_mult.sh
        os.environ["root_dir"] = self.root_dir
        os.environ["test_data_dir"] = "/gpfs/alpine/proj-shared/chm137/data/testdata"
        test_script = "tests/forward_and_adjoint.py" 
        args = self.launch_cmds + self.launch_args + [test_script]
        subprocess.check_call(args)


