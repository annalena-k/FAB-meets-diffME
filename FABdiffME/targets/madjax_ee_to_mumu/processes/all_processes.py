from __future__ import division
from model.aloha_methods import *
from madjax.wavefunctions import *
from jax import vmap 
from jax import numpy as np 
class Matrix_1_epem_mupmum(object):

    def __init__(self):
        """define the object"""
        self.clean()

    def clean(self):
        self.jamp = []

    def get_external_masses(self, params):

        return ( (params["ZERO"], params["ZERO"]), (params["ZERO"], params["ZERO"]) )

    def smatrix(self,p, model):
        #  
        #  MadGraph5_aMC@NLO v. 3.5.0, 2023-05-12
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        # 
        # MadGraph5_aMC@NLO StandAlone Version
        # 
        # Returns amplitude squared summed/avg over colors
        # and helicities
        # for the point in phase space P(0:3,NEXTERNAL)
        #  
        # Process: e+ e- > mu+ mu- WEIGHTED<=4 @1
        #  
        # Clean additional output
        #
        self.clean()
        #  
        # CONSTANTS
        #  
        nexternal = 4
        ndiags = 2
        ncomb = 16
        #  
        # LOCAL VARIABLES 
        #  
        helicities = [ \
        [-1,1,1,-1],
        [-1,1,1,1],
        [-1,1,-1,-1],
        [-1,1,-1,1],
        [-1,-1,1,-1],
        [-1,-1,1,1],
        [-1,-1,-1,-1],
        [-1,-1,-1,1],
        [1,1,1,-1],
        [1,1,1,1],
        [1,1,-1,-1],
        [1,1,-1,1],
        [1,-1,1,-1],
        [1,-1,1,1],
        [1,-1,-1,-1],
        [1,-1,-1,1]]
        denominator = 4
        # ----------
        # BEGIN CODE
        # ----------
        self.amp2 = [0.] * ndiags
        self.helEvals = []
        ans = 0.

        # ----------
        # OLD CODE
        # ----------
        #for hel in helicities:
        #    t = self.matrix(p, hel, model)
        #    ans = ans + t
        #    self.helEvals.append([hel, t.real / denominator ])

        t = self.vmap_matrix( p, np.array(helicities), model )
        ans = np.sum(t)
        self.helEvals.append( (helicities, t.real / denominator) )
        
        ans = ans / denominator
        return ans.real
    
    def vmap_matrix(self, p, hel_batch, model):
        return vmap(self.matrix, in_axes=(None,0,None), out_axes=0)(p, hel_batch, model)

    def matrix(self, p, hel, model):
        #  
        #  MadGraph5_aMC@NLO v. 3.5.0, 2023-05-12
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        #
        # Returns amplitude squared summed/avg over colors
        # for the point with external lines W(0:6,NEXTERNAL)
        #
        # Process: e+ e- > mu+ mu- WEIGHTED<=4 @1
        #  
        #  
        # Process parameters
        #  
        ngraphs = 2
        nexternal = 4
        nwavefuncs = 5
        ncolor = 1
        ZERO = 0.
        #  
        # Color matrix
        #  
        denom = [1.];
        cf = [[1.]];
        #
        # Model parameters
        #
        mdl_WZ = model["mdl_WZ"]
        mdl_MZ = model["mdl_MZ"]
        GC_50 = model["GC_50"]
        GC_59 = model["GC_59"]
        GC_3 = model["GC_3"]
        # ----------
        # Begin code
        # ----------
        amp = [None] * ngraphs
        w = [None] * nwavefuncs
        w[0] = oxxxxx(p[0],ZERO,hel[0],-1)
        w[1] = ixxxxx(p[1],ZERO,hel[1],+1)
        w[2] = ixxxxx(p[2],ZERO,hel[2],-1)
        w[3] = oxxxxx(p[3],ZERO,hel[3],+1)
        w[4]= FFV1P0_3(w[1],w[0],GC_3,ZERO,ZERO)
        # Amplitude(s) for diagram number 1
        amp[0]= FFV1_0(w[2],w[3],w[4],GC_3)
        w[4]= FFV2_4_3(w[1],w[0],GC_50,GC_59,mdl_MZ,mdl_WZ)
        # Amplitude(s) for diagram number 2
        amp[1]= FFV2_4_0(w[2],w[3],w[4],GC_50,GC_59)

        jamp = [None] * ncolor

        jamp[0] = -amp[0]-amp[1]

        self.amp2[0]+=abs(amp[0]*amp[0].conjugate())
        self.amp2[1]+=abs(amp[1]*amp[1].conjugate())

        # ----------
        # OLD CODE
        # ----------
        #matrix = 0.
        #for i in range(ncolor):
        #    ztemp = 0
        #    for j in range(ncolor):
        #        ztemp = ztemp + cf[i][j]*jamp[j]
        #    matrix = matrix + ztemp * jamp[i].conjugate()/denom[i]   
        self.jamp.append(jamp)

        matrix = np.sum( np.dot(np.array(cf), np.array(jamp)) * np.array(jamp).conjugate() / np.array(denom) )

        return matrix

