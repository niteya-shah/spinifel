"""
New feature requests (most likely apply to main.py)
"""

# [NEW FEATURE 1] staging.pickle contains classs/arrays 
# needed for restarting the state prior to the crash.
if not load("staging.pickle"):
    inputs = sp.get_my_inputs()
    
for gen in range(10):
    # [NEW FEATURE 2] To prevent using too few images per reconstruction
    wait_for_images(n_req=1000)

    # Core reconstruction algorithm
    phase_results = sp.phasing()
    ac = sp.solve_ac()
    o = sp.match()
    
    # Potentialy save intermediate state here
    sp.stage("staging.pickle")


    # [NEW FEATURE 3] Try to see if there are more data that can be used
    # in the next generation
    slices_.append(sp.get_more_slices_())


