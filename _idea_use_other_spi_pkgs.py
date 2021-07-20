"""
Use spinifel to perform SPI reconstruction with different 
software packages.

-----------------------------------------------------------
tmol file example
[data]
datasource=[hdf5, psana]
[runtime]
backend=[cpu, gpu, legion]
algorithm=[internal, cmtip, dragonfile]

-----------------------------------------------------------
Requirements:
    1) tmol files have all parameters required to run other 
    software packages.
    2) each package (including the internal one) has a driver
    file that allow spinifel to pass settings and data then
    run "reconstruction".

Q1: Can toml file call another toml file 
"""

from spinifel import DataSource, Settings
from spinifel.spi import reconstruct            #(1)
# from spinifel.cmtip import reconstruct        #(2)
# from spinifel.dragonfly import reconstruct    #(3)

# Read settings from toml file
settings = Settings(my.toml)

# Read data
ds = DataSource(settings.data)

# Call reconstruction 
reconstruct(ds, settings) # call driver script for selected library




