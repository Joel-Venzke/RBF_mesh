from Node_Set import Node_Set
import numpy as np
from scipy.special import comb
# notes:
# random rotation cleans up the issue at large r (I think)
# the small r limit needs to be fixed. A quasi uniform node layout should do this
# find average distance to nearest neighbor on unit sphere and match this to delta_r
# scaling by the radius. use this distance for outer box of quasi uniform node set


node_set = Node_Set(r_max=10,
                 delta_r=1.0,
                 stencil_size=50,
                 md_degree=6,
                 rbf_order=7,
                 poly_order=2,
                 exp_order=3,)

node_set.check_quality()
node_set.calculate_opperator_weights()

function = node_set.node_set[:,0]**2 + 3.*node_set.node_set[:,0]**2 + 4.*node_set.node_set[:,0]**2