#  MadGraph5_aMC@NLO v. 3.5.0, 2023-05-12
#  By the MadGraph5_aMC@NLO Development Team
#  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch

import madjax.madjax_patch as cmath
from madjax.madjax_patch import complex

def calculate_full_parameters(slha):
        """ Instantiates using default value or the path of a SLHA param card."""
       
        
        _result_params = {'ZERO': 0.0}

        # Computing independent parameters
        mdl_WH = slha.get(("decay", 25), 6.382339e-03);
        mdl_WT = slha.get(("decay", 6), 1.491500e+00);
        mdl_WW = slha.get(("decay", 24), 2.047600e+00);
        mdl_WZ = slha.get(("decay", 23), 2.441404e+00);
        mdl_MTA = slha.get(("mass", 15), 1.777000e+00);
        mdl_MH = slha.get(("mass", 25), 1.250000e+02);
        mdl_MB = slha.get(("mass", 5), 4.700000e+00);
        mdl_MT = slha.get(("mass", 6), 1.730000e+02);
        mdl_MZ = slha.get(("mass", 23), 9.118800e+01);
        mdl_ymtau = slha.get(("yukawa", 15), 1.777000e+00);
        mdl_ymt = slha.get(("yukawa", 6), 1.730000e+02);
        mdl_ymb = slha.get(("yukawa", 5), 4.700000e+00);
        aS = slha.get(("sminputs", 3), 1.180000e-01);
        mdl_Gf = slha.get(("sminputs", 2), 1.166390e-05);
        aEWM1 = slha.get(("sminputs", 1), 1.325070e+02);
        mdl_conjg__CKM3x3 = 1.0
        mdl_CKM3x3 = 1.0
        mdl_conjg__CKM1x1 = 1.0
        mdl_complexi = complex(0,1)
        mdl_MZ__exp__2 = mdl_MZ**2
        mdl_MZ__exp__4 = mdl_MZ**4
        mdl_sqrt__2 =  cmath.sqrt(2) 
        mdl_MH__exp__2 = mdl_MH**2
        mdl_aEW = 1/aEWM1
        mdl_MW = cmath.sqrt(mdl_MZ__exp__2/2. + cmath.sqrt(mdl_MZ__exp__4/4. - (mdl_aEW*cmath.pi*mdl_MZ__exp__2)/(mdl_Gf*mdl_sqrt__2)))
        mdl_sqrt__aEW =  cmath.sqrt(mdl_aEW) 
        mdl_ee = 2*mdl_sqrt__aEW*cmath.sqrt(cmath.pi)
        mdl_MW__exp__2 = mdl_MW**2
        mdl_sw2 = 1 - mdl_MW__exp__2/mdl_MZ__exp__2
        mdl_cw = cmath.sqrt(1 - mdl_sw2)
        mdl_sqrt__sw2 =  cmath.sqrt(mdl_sw2) 
        mdl_sw = mdl_sqrt__sw2
        mdl_g1 = mdl_ee/mdl_cw
        mdl_gw = mdl_ee/mdl_sw
        mdl_vev = (2*mdl_MW*mdl_sw)/mdl_ee
        mdl_vev__exp__2 = mdl_vev**2
        mdl_lam = mdl_MH__exp__2/(2.*mdl_vev__exp__2)
        mdl_yb = (mdl_ymb*mdl_sqrt__2)/mdl_vev
        mdl_yt = (mdl_ymt*mdl_sqrt__2)/mdl_vev
        mdl_ytau = (mdl_ymtau*mdl_sqrt__2)/mdl_vev
        mdl_muH = cmath.sqrt(mdl_lam*mdl_vev__exp__2)
        mdl_I1x33 = mdl_yb*mdl_conjg__CKM3x3
        mdl_I2x33 = mdl_yt*mdl_conjg__CKM3x3
        mdl_I3x33 = mdl_CKM3x3*mdl_yt
        mdl_I4x33 = mdl_CKM3x3*mdl_yb
        mdl_ee__exp__2 = mdl_ee**2
        mdl_sw__exp__2 = mdl_sw**2
        mdl_cw__exp__2 = mdl_cw**2

        # Computing independent couplings
        GC_2 = (2*mdl_ee*mdl_complexi)/3.
        GC_3 = -(mdl_ee*mdl_complexi)
        GC_50 = -(mdl_cw*mdl_ee*mdl_complexi)/(2.*mdl_sw)
        GC_58 = -(mdl_ee*mdl_complexi*mdl_sw)/(6.*mdl_cw)
        GC_59 = (mdl_ee*mdl_complexi*mdl_sw)/(2.*mdl_cw)
        GC_100 = (mdl_ee*mdl_complexi*mdl_conjg__CKM1x1)/(mdl_sw*mdl_sqrt__2)

        # Computing dependent parameters
        mdl_sqrt__aS =  cmath.sqrt(aS) 
        G = 2*mdl_sqrt__aS*cmath.sqrt(cmath.pi)
        mdl_G__exp__2 = G**2

        # Computing independent parameters


        # ------------------------------
        # Building Dictionary
        # ------------------------------
        # Setting independent parameters
        # Model parameters independent of aS
        _result_params["mdl_WH"] = (mdl_WH.real)
        _result_params["mdl_WT"] = (mdl_WT.real)
        _result_params["mdl_WW"] = (mdl_WW.real)
        _result_params["mdl_WZ"] = (mdl_WZ.real)
        _result_params["mdl_MTA"] = (mdl_MTA.real)
        _result_params["mdl_MH"] = (mdl_MH.real)
        _result_params["mdl_MB"] = (mdl_MB.real)
        _result_params["mdl_MT"] = (mdl_MT.real)
        _result_params["mdl_MZ"] = (mdl_MZ.real)
        _result_params["mdl_ymtau"] = (mdl_ymtau.real)
        _result_params["mdl_ymt"] = (mdl_ymt.real)
        _result_params["mdl_ymb"] = (mdl_ymb.real)
        _result_params["aS"] = (aS.real)
        _result_params["mdl_Gf"] = (mdl_Gf.real)
        _result_params["aEWM1"] = (aEWM1.real)
        _result_params["mdl_conjg__CKM3x3"] = (mdl_conjg__CKM3x3.real)
        _result_params["mdl_CKM3x3"] = (mdl_CKM3x3.real)
        _result_params["mdl_conjg__CKM1x1"] = (mdl_conjg__CKM1x1.real)
        _result_params["mdl_complexi"] = complex(mdl_complexi)
        _result_params["mdl_MZ__exp__2"] = (mdl_MZ__exp__2.real)
        _result_params["mdl_MZ__exp__4"] = (mdl_MZ__exp__4.real)
        _result_params["mdl_sqrt__2"] = (mdl_sqrt__2.real)
        _result_params["mdl_MH__exp__2"] = (mdl_MH__exp__2.real)
        _result_params["mdl_aEW"] = (mdl_aEW.real)
        _result_params["mdl_MW"] = (mdl_MW.real)
        _result_params["mdl_sqrt__aEW"] = (mdl_sqrt__aEW.real)
        _result_params["mdl_ee"] = (mdl_ee.real)
        _result_params["mdl_MW__exp__2"] = (mdl_MW__exp__2.real)
        _result_params["mdl_sw2"] = (mdl_sw2.real)
        _result_params["mdl_cw"] = (mdl_cw.real)
        _result_params["mdl_sqrt__sw2"] = (mdl_sqrt__sw2.real)
        _result_params["mdl_sw"] = (mdl_sw.real)
        _result_params["mdl_g1"] = (mdl_g1.real)
        _result_params["mdl_gw"] = (mdl_gw.real)
        _result_params["mdl_vev"] = (mdl_vev.real)
        _result_params["mdl_vev__exp__2"] = (mdl_vev__exp__2.real)
        _result_params["mdl_lam"] = (mdl_lam.real)
        _result_params["mdl_yb"] = (mdl_yb.real)
        _result_params["mdl_yt"] = (mdl_yt.real)
        _result_params["mdl_ytau"] = (mdl_ytau.real)
        _result_params["mdl_muH"] = (mdl_muH.real)
        _result_params["mdl_I1x33"] = complex(mdl_I1x33)
        _result_params["mdl_I2x33"] = complex(mdl_I2x33)
        _result_params["mdl_I3x33"] = complex(mdl_I3x33)
        _result_params["mdl_I4x33"] = complex(mdl_I4x33)
        _result_params["mdl_ee__exp__2"] = (mdl_ee__exp__2.real)
        _result_params["mdl_sw__exp__2"] = (mdl_sw__exp__2.real)
        _result_params["mdl_cw__exp__2"] = (mdl_cw__exp__2.real)

        # Setting independent couplings
        # Model parameters dependent on aS
        _result_params["mdl_sqrt__aS"] = (mdl_sqrt__aS.real)
        _result_params["G"] = (G.real)
        _result_params["mdl_G__exp__2"] = (mdl_G__exp__2.real)

        # Setting dependent parameters
        # Model couplings independent of aS
        _result_params["GC_2"] = complex(GC_2)
        _result_params["GC_3"] = complex(GC_3)
        _result_params["GC_50"] = complex(GC_50)
        _result_params["GC_58"] = complex(GC_58)
        _result_params["GC_59"] = complex(GC_59)
        _result_params["GC_100"] = complex(GC_100)

        # Setting independent parameters
        # Model couplings dependent on aS




        return _result_params