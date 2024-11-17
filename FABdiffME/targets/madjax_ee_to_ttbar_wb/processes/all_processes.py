from __future__ import division
from model.aloha_methods import *
from madjax.wavefunctions import *
from jax import vmap 
from jax import numpy as np 
class Matrix_1_epem_ttx_t_wpb_tx_wmbx(object):

    def __init__(self):
        """define the object"""
        self.clean()

    def clean(self):
        self.jamp = []

    def get_external_masses(self, params):

        return ( (params["ZERO"], params["ZERO"]), (params["mdl_MW"], params["mdl_MB"], params["mdl_MW"], params["mdl_MB"]) )

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
        # Process: e+ e- > t t~ WEIGHTED<=4 @1
# *   Decay: t > w+ b WEIGHTED<=2
# *   Decay: t~ > w- b~ WEIGHTED<=2
        #  
        # Clean additional output
        #
        self.clean()
        #  
        # CONSTANTS
        #  
        nexternal = 6
        ndiags = 2
        ncomb = 144
        #  
        # LOCAL VARIABLES 
        #  
        helicities = [ \
        [-1,1,-1,-1,1,1],
        [-1,1,-1,-1,1,-1],
        [-1,1,-1,-1,0,1],
        [-1,1,-1,-1,0,-1],
        [-1,1,-1,-1,-1,1],
        [-1,1,-1,-1,-1,-1],
        [-1,1,-1,1,1,1],
        [-1,1,-1,1,1,-1],
        [-1,1,-1,1,0,1],
        [-1,1,-1,1,0,-1],
        [-1,1,-1,1,-1,1],
        [-1,1,-1,1,-1,-1],
        [-1,1,0,-1,1,1],
        [-1,1,0,-1,1,-1],
        [-1,1,0,-1,0,1],
        [-1,1,0,-1,0,-1],
        [-1,1,0,-1,-1,1],
        [-1,1,0,-1,-1,-1],
        [-1,1,0,1,1,1],
        [-1,1,0,1,1,-1],
        [-1,1,0,1,0,1],
        [-1,1,0,1,0,-1],
        [-1,1,0,1,-1,1],
        [-1,1,0,1,-1,-1],
        [-1,1,1,-1,1,1],
        [-1,1,1,-1,1,-1],
        [-1,1,1,-1,0,1],
        [-1,1,1,-1,0,-1],
        [-1,1,1,-1,-1,1],
        [-1,1,1,-1,-1,-1],
        [-1,1,1,1,1,1],
        [-1,1,1,1,1,-1],
        [-1,1,1,1,0,1],
        [-1,1,1,1,0,-1],
        [-1,1,1,1,-1,1],
        [-1,1,1,1,-1,-1],
        [-1,-1,-1,-1,1,1],
        [-1,-1,-1,-1,1,-1],
        [-1,-1,-1,-1,0,1],
        [-1,-1,-1,-1,0,-1],
        [-1,-1,-1,-1,-1,1],
        [-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,1,1,1],
        [-1,-1,-1,1,1,-1],
        [-1,-1,-1,1,0,1],
        [-1,-1,-1,1,0,-1],
        [-1,-1,-1,1,-1,1],
        [-1,-1,-1,1,-1,-1],
        [-1,-1,0,-1,1,1],
        [-1,-1,0,-1,1,-1],
        [-1,-1,0,-1,0,1],
        [-1,-1,0,-1,0,-1],
        [-1,-1,0,-1,-1,1],
        [-1,-1,0,-1,-1,-1],
        [-1,-1,0,1,1,1],
        [-1,-1,0,1,1,-1],
        [-1,-1,0,1,0,1],
        [-1,-1,0,1,0,-1],
        [-1,-1,0,1,-1,1],
        [-1,-1,0,1,-1,-1],
        [-1,-1,1,-1,1,1],
        [-1,-1,1,-1,1,-1],
        [-1,-1,1,-1,0,1],
        [-1,-1,1,-1,0,-1],
        [-1,-1,1,-1,-1,1],
        [-1,-1,1,-1,-1,-1],
        [-1,-1,1,1,1,1],
        [-1,-1,1,1,1,-1],
        [-1,-1,1,1,0,1],
        [-1,-1,1,1,0,-1],
        [-1,-1,1,1,-1,1],
        [-1,-1,1,1,-1,-1],
        [1,1,-1,-1,1,1],
        [1,1,-1,-1,1,-1],
        [1,1,-1,-1,0,1],
        [1,1,-1,-1,0,-1],
        [1,1,-1,-1,-1,1],
        [1,1,-1,-1,-1,-1],
        [1,1,-1,1,1,1],
        [1,1,-1,1,1,-1],
        [1,1,-1,1,0,1],
        [1,1,-1,1,0,-1],
        [1,1,-1,1,-1,1],
        [1,1,-1,1,-1,-1],
        [1,1,0,-1,1,1],
        [1,1,0,-1,1,-1],
        [1,1,0,-1,0,1],
        [1,1,0,-1,0,-1],
        [1,1,0,-1,-1,1],
        [1,1,0,-1,-1,-1],
        [1,1,0,1,1,1],
        [1,1,0,1,1,-1],
        [1,1,0,1,0,1],
        [1,1,0,1,0,-1],
        [1,1,0,1,-1,1],
        [1,1,0,1,-1,-1],
        [1,1,1,-1,1,1],
        [1,1,1,-1,1,-1],
        [1,1,1,-1,0,1],
        [1,1,1,-1,0,-1],
        [1,1,1,-1,-1,1],
        [1,1,1,-1,-1,-1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,-1],
        [1,1,1,1,0,1],
        [1,1,1,1,0,-1],
        [1,1,1,1,-1,1],
        [1,1,1,1,-1,-1],
        [1,-1,-1,-1,1,1],
        [1,-1,-1,-1,1,-1],
        [1,-1,-1,-1,0,1],
        [1,-1,-1,-1,0,-1],
        [1,-1,-1,-1,-1,1],
        [1,-1,-1,-1,-1,-1],
        [1,-1,-1,1,1,1],
        [1,-1,-1,1,1,-1],
        [1,-1,-1,1,0,1],
        [1,-1,-1,1,0,-1],
        [1,-1,-1,1,-1,1],
        [1,-1,-1,1,-1,-1],
        [1,-1,0,-1,1,1],
        [1,-1,0,-1,1,-1],
        [1,-1,0,-1,0,1],
        [1,-1,0,-1,0,-1],
        [1,-1,0,-1,-1,1],
        [1,-1,0,-1,-1,-1],
        [1,-1,0,1,1,1],
        [1,-1,0,1,1,-1],
        [1,-1,0,1,0,1],
        [1,-1,0,1,0,-1],
        [1,-1,0,1,-1,1],
        [1,-1,0,1,-1,-1],
        [1,-1,1,-1,1,1],
        [1,-1,1,-1,1,-1],
        [1,-1,1,-1,0,1],
        [1,-1,1,-1,0,-1],
        [1,-1,1,-1,-1,1],
        [1,-1,1,-1,-1,-1],
        [1,-1,1,1,1,1],
        [1,-1,1,1,1,-1],
        [1,-1,1,1,0,1],
        [1,-1,1,1,0,-1],
        [1,-1,1,1,-1,1],
        [1,-1,1,1,-1,-1]]
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
        # Process: e+ e- > t t~ WEIGHTED<=4 @1
# *   Decay: t > w+ b WEIGHTED<=2
# *   Decay: t~ > w- b~ WEIGHTED<=2
        #  
        #  
        # Process parameters
        #  
        ngraphs = 2
        nexternal = 6
        nwavefuncs = 6
        ncolor = 1
        ZERO = 0.
        #  
        # Color matrix
        #  
        denom = [1.];
        cf = [[3.]];
        #
        # Model parameters
        #
        mdl_MW = model["mdl_MW"]
        mdl_WZ = model["mdl_WZ"]
        mdl_MT = model["mdl_MT"]
        mdl_WT = model["mdl_WT"]
        mdl_WW = model["mdl_WW"]
        mdl_MZ = model["mdl_MZ"]
        mdl_MB = model["mdl_MB"]
        GC_59 = model["GC_59"]
        GC_100 = model["GC_100"]
        GC_50 = model["GC_50"]
        GC_58 = model["GC_58"]
        GC_2 = model["GC_2"]
        GC_3 = model["GC_3"]
        # ----------
        # Begin code
        # ----------
        amp = [None] * ngraphs
        w = [None] * nwavefuncs
        w[0] = oxxxxx(p[0],ZERO,hel[0],-1)
        w[1] = ixxxxx(p[1],ZERO,hel[1],+1)
        w[2] = vxxxxx(p[2],mdl_MW,hel[2],+1)
        w[3] = oxxxxx(p[3],mdl_MB,hel[3],+1)
        w[4]= FFV2_1(w[3],w[2],GC_100,mdl_MT,mdl_WT)
        w[3] = vxxxxx(p[4],mdl_MW,hel[4],+1)
        w[2] = ixxxxx(p[5],mdl_MB,hel[5],-1)
        w[5]= FFV2_2(w[2],w[3],GC_100,mdl_MT,mdl_WT)
        w[2]= FFV1P0_3(w[1],w[0],GC_3,ZERO,ZERO)
        # Amplitude(s) for diagram number 1
        amp[0]= FFV1_0(w[5],w[4],w[2],GC_2)
        w[2]= FFV2_4_3(w[1],w[0],GC_50,GC_59,mdl_MZ,mdl_WZ)
        # Amplitude(s) for diagram number 2
        amp[1]= FFV2_5_0(w[5],w[4],w[2],-GC_50,GC_58)

        jamp = [None] * ncolor

        jamp[0] = amp[0]+amp[1]

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

