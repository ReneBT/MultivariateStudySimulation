import os
import numpy as np
import pandas as pd
from src.file_locations import data_folder, images_folder
import matplotlib.pyplot as plt
from scipy import stats

### class specSim
class specSim:
### __init__
    def __init__( self ,
                 proteins, # name of protien as in filename
                 postTranslation, # proportion of amino acids converted to another
                 x, # array of x axis positions
                 ):
        # row 0 is the one letter code for the amino acid, the 3 letter version
        # is preferred in code as they are easier to decipher for non protein specialists
        # row 1 is relative weighting to be given to the Raman intensity
        self.x_cal = x
        self.proteins = proteins
        self.AA_comp= pd.DataFrame({'Ala':['a', 0.2, *proteins],
                  'Arg':['r', 1, *proteins],
                  'Asp':['d', 1, *proteins],
                  'Asn':['n', 1, *proteins],
                  'Cys':['c', 1, *proteins], # not simulated for manuscript due to low prevalence
                  'Glu':['e', 1, *proteins],
                  'Gln':['q', 1, *proteins],
                  'Gly':['g', 0.1, *proteins],
                  'His':['h', 10, *proteins],
                  'Hyp':['o', 5, *proteins],
                  'Ile':['i', 0.3, *proteins],
                  'Leu':['l', 0.3, *proteins],
                  'Lys':['k', 1, *proteins],
                  'Met':['m', 1, *proteins], # not simulated for manuscript due to low prevalence
                  'Phe':['f', 20, *proteins],
                  'Pro':['p', 5, *proteins],
                  'Ser':['s', 1, *proteins],
                  'Thr':['t', 1, *proteins],
                  'Tyr':['y', 20, *proteins],
                  'Trp':['w', 20, *proteins],
                  'Val':['v', 0.5, *proteins], },
                  index = [ '1 Letter Code' , 'Raman Intensity' , *proteins ] )

        for i in range(len(proteins)):
            temp = ''.join(open(os.path.join(data_folder, 'Primary Structure '
                                             + proteins[i]+'.txt'),
                        'r').read().split())
            peptideLength = len(temp)

            for col in range(len(self.AA_comp.columns)):
                self.AA_comp.iloc[i+2, col] = temp.count(self.AA_comp.iloc[0, col])/peptideLength

            for pt in range(len(postTranslation['Protein'])):
                if postTranslation['Protein'][pt]==proteins[i]: # this doesn't assume the same modifications are defined for all proteins
                    self.AA_comp[postTranslation['PTM'][pt]][i+2] = (
                        self.AA_comp[postTranslation['Original'][pt]][i+2] *
                        postTranslation['Proportion'][pt] )
                    self.AA_comp[postTranslation['Original'][pt]][i+2] = (
                        self.AA_comp[postTranslation['Original'][pt]][i+2] *
                        (1 - postTranslation['Proportion'][pt] ) )

        # load in the peak parameters for all the spectral constituents
        # stored in  excel spredsheet with one spreadsheet per constituent
        # each spreadsheet should have four columns with a header for
        #       p - central position of peak
        #       a - amplitude or intensity of the peak
        #       w - width of the peak (full width at half maximum)
        #       g - gaussian proportion (assumes non-gaussian is lorentzian)
        fname = os.path.join(data_folder, 'Spectral Peaks.xlsx')
        self.constituents = pd.ExcelFile(fname).sheet_names
        self.constituentSpec = pd.DataFrame()
        i = 0

        while i < len(self.constituents):
                WS = pd.read_excel( fname, sheet_name=self.constituents[i],
                                   header=None )
                self.constituentSpec[self.constituents[i]] = ( specSim.brickSpec( self,
                                                                x,
                                                                WS[0][1:],
                                                                WS[2][1:],
                                                                WS[1][1:],
                                                                WS[3][1:]) *
                                                                WS[4][1]  )
#                self.constituentSpec[self.constituents[i]] = (
#                    self.constituentSpec[ self.constituents[i]] /
#                print(self.constituents[i]+': '+str(np.mean(self.constituentSpec[self.constituents[i]])))
#                    *100)
                i += 1

        # now create a set of background shapes for 0 Collagen, 1 Elastin,
        # 3 dehydrated and
        # 4 emissive AGE-ALE that is too low concentration to be detected
        # directly but weakly correlated with AGE/ALEs that share chemical pathway
        # these will perturb the noise but currently not explictly combined as
        # baseline correction not currently included as a factor
        self.backs = np.column_stack( (
            np.sum( np.array([ (x/1000-0.75)**5,
                              0.0005*(x/175-10)**4,
                              -0.0006*(x/125-15)**3,
                              -0.0015*(x/20-50)**2,
                              0.0026*(x/10+300)]), axis=0),
            np.sum( np.array([ 0.0002*(x/125-13)**4,
                              -0.0002*(x/105-19)**3,
                              -0.0002*(x/15-50)**2,
                              0.005*(300 - x/10),
                              np.ones(x.shape)/2]), axis=0),
             np.sum( np.array([ 0.01*(x/215-4)**3,
                              -0.0002*(x/35-10)**2,
                              0.0013*(500 - x/33),
                              np.ones(x.shape)/1.32]), axis=0),
            np.exp((-(x-1300)**2)
                   /(456)**2),
            ) )

        self.backs = self.backs / np.mean(self.backs, axis=0)
        return
### Base Pure spectra
    def brickSpec( self , # makes the basic spectral building blocks
                    x, # x axis values for which y values are to be returned
                    p, # peak centres (location of maxima) in same units as x
                    w, # full width at half maximum
                    a, # intensity at peak maxima
                    g, # shape parameter (proportion gaussian, 1 for 100% gaussian 0 for 100% Lorentzian)
               ):
        # p, w, a and g should all be the same size
        nPk = p.shape[0]
        p = np.array(p).astype('float')
        w = np.array(w).astype('float')
        a = np.array(a).astype('float')
        g = np.array(g).astype('float')

        if x is not None:
            Gs = np.sum(a*np.exp(
                    (-(np.tile(x.reshape(-1, 1), nPk).astype('float')-p)**2)
                    /(w*0.60056)**2)*g, axis=1)
            Lz = np.sum(a*(w/2)**2/(
                    (np.tile(x.reshape(-1, 1), nPk).astype('float')-p)**2
                    + (w/2)**2)
                *(1-g), axis=1)
            Spc = Gs + Lz
            #Spc = Spc/np.sum(Spc**2)**0.5
        return Spc
### figures of base pure spectra
    def print_Brick(self, per_plot, res):
        for i in range( np.ceil(len(self.constituents)/per_plot).astype('int')):
            plt.plot(self.x_cal,
                     (self.constituentSpec.iloc[:, slice(i*per_plot, i*per_plot+per_plot)]/
                     np.mean(self.constituentSpec.iloc[:, slice(i*per_plot, i*per_plot+per_plot)]))-
                     np.arange(per_plot)*2)
            plt.gca().set_yticklabels([])
            plt.legend(self.constituents[slice(i*per_plot, i*per_plot+per_plot)],
                       fontsize='x-small')
            image_name = ("Simulated Base Spectra" + str(i*per_plot+1) + " to " +
                          str (i*per_plot+per_plot) + ".png")
            plt.savefig(os.path.join( images_folder , image_name),
                        dpi=res)
            plt.close()

 ### generate observation spectra
    def specSample( self , simSam , samRep , mnCnts ):
        params = simSam.columns[ (simSam.columns != 'Matrix') &
                                (simSam.columns != 'Youngs Modulus') &
                                (simSam.columns != 'pH') &
                                (simSam.columns != 'Hydration') &
                                (simSam.columns != 'Youngs Modulus Elastin') &
                                (simSam.columns != 'Youngs Modulus Collagen') &
                                (simSam.columns != 'Youngs Modulus Measurement Error')
                                ]
        self.samSpec = np.zeros((len(self.x_cal), simSam.shape[0]))
        # sampling variablility applied to parameters i.e. the amount of a parameter captured by the spectrometer
        # Should influence low concentration constituents more and each consitiuent will have an independent error
        simSam[simSam<0] = 0 # no negative concentrations possible
        randomVariation = stats.norm.rvs( 0 , 1 , size=( simSam.shape ) )
        self.sample_variation = 10**( randomVariation / samRep * ( 1 - simSam**0.5 ) )
        simSam = simSam *  self.sample_variation
        simSam[simSam<0] = 0 # no negative concentrations possible
        simSam = simSam/np.mean(simSam)
        hydration = simSam['Hydration']# hydration is weakly influenced at high matrix concentrations (matrix in excess), but impact accelerates at low levels

        self.signalInt = stats.norm.rvs( mnCnts , mnCnts**0.5*samRep ,
                                        size=( simSam.shape[0] ) ) # variability of focus affected by spot size which is encoded in samRep
        self.signalInt[self.signalInt<mnCnts/10] = mnCnts/10 # model reanalysis of low intensity samples until they are 1/10th intensity of mean

        self.simBack = ( np.outer( self.backs[:, 0], np.abs(self.signalInt*simSam['Collagen']*hydration*
                        stats.norm.rvs( 1 , 0.2 , size=( simSam.shape[0] ) ) ) )+
                        np.outer( self.backs[:, 1], np.abs(self.signalInt*3*simSam['Elastin']*hydration*
                        stats.norm.rvs( 1 , 0.22 , size=( simSam.shape[0] ) ) ) )  +
                        np.outer( self.backs[:, 2], np.abs(self.signalInt/3*(1-hydration)*
                        stats.norm.rvs( 2 , 0.4 , size=( simSam.shape[0] ) ) ) )  +
                        np.outer( self.backs[:, 3], np.abs((self.signalInt*simSam['Pentosidine']*10 + # AGE due to glucose metabolite present in amounts undetectable by Raman scattering, but with high quantum efficicency broad emission spectrum
                                            simSam['MGO']*3 + simSam['CML']*5)*
                        stats.norm.rvs( 3 , 0.75 , size=( simSam.shape[0] ) ) ) )
                        )+800

        for p in range(len(params)):
#            H = hydration
#            D = (1-hydration)
            if params[p]+' Scar' in self.constituents:
#                H = H - simSam['Scar']*hydration #start this anew so only affected by parameters that influence the constituent
 #               D = D - simSam['Scar']*(1-hydration)
                self.samSpec = ( self.samSpec +
                           np.outer(self.constituentSpec[params[p]+' Scar'],
                                    simSam[params[p]]*simSam['Scar']  ))
                if np.sum(np.isnan(self.samSpec))>0:
                    print('Scar NaNs: ' + str(np.sum(np.isnan(self.samSpec))))
                    return
            if params[p]+' HPO4_2H2O' in self.constituents:
#                H = H - simSam['HPO4_2H2O']*hydration
#                D = D - simSam['HPO4_2H2O']*(1-hydration)
                self.samSpec = ( self.samSpec +
                           np.outer(
                               self.constituentSpec[params[p]+' HPO4_2H2O'],
                               simSam[params[p]]*simSam['HPO4_2H2O']  ))
                if np.sum(np.isnan(self.samSpec))>0:
                    print('Brushite NaNs: ' + str(np.sum(np.isnan(self.samSpec))))
                    return

            if params[p]+' CML' in self.constituents:
                totAGEacid = simSam['CML'] + simSam['CEL']#model same impact on protein
#                H = H - totAGEacid*hydration
#                D = D - totAGEacid*(1-hydration)
                self.samSpec = ( self.samSpec +
                           np.outer(self.constituentSpec[params[p]+' CML'],
                                    simSam[params[p]]*totAGEacid  ) )
                if np.sum(np.isnan(self.samSpec))>0:
                    print(str(p) + ': ' + str(np.sum(np.isnan(self.samSpec))))
                    return

            if params[p]+' GH1' in self.constituents:
#                H = H - simSam['GH1']*hydration
#                D = D - simSam['GH1']*(1-hydration)
                self.samSpec = ( self.samSpec +
                           np.outer(self.constituentSpec[params[p]+' GH1'],
                                    simSam[params[p]]*simSam['GH1']  ) )
                if np.sum(np.isnan(self.samSpec))>0:
                    print(str(p) + ': ' + str(np.sum(np.isnan(self.samSpec))))
                    return

            if params[p]+' CEP' in self.constituents:
#                H = H - simSam['CEP']*hydration
#                D = D - simSam['CEP']*(1-hydration)
                self.samSpec = ( self.samSpec +
                           np.outer(self.constituentSpec[params[p]+' CEP'],
                                    simSam[params[p]]*simSam['CEP']  ) )
                if np.sum(np.isnan(self.samSpec))>0:
                    print(str(p) + ': ' + str(np.sum(np.isnan(self.samSpec))))
                    return

            if params[p]+' Pentosidine' in self.constituents:
#                H = H - simSam['Pentosidine']*hydration
#                D = D - simSam['Pentosidine']*(1-hydration)
                self.samSpec = ( self.samSpec +
                           np.outer(
                               self.constituentSpec[params[p]+' Pentosidine'],
                                    simSam[params[p]]*simSam['Pentosidine']) )
                if np.sum(np.isnan(self.samSpec))>0:
                    print(str(p) + ': ' + str(np.sum(np.isnan(self.samSpec))))
                    return
#            H[H>1] = 1
#            H[H<0] = 0
#            D[D>1] = 1
#            D[D<0] = 0
            if params[p]+' H' in self.constituents:
                self.samSpec = ( self.samSpec +
                           np.outer(self.constituentSpec[params[p]+' H'],
                                    hydration*simSam[params[p]]  ) )
                if np.sum(np.isnan(self.samSpec))>0:
                    print(str(p) + ': ' + str(np.sum(np.isnan(self.samSpec))))
                    return
            if params[p]+' D' in self.constituents:
                self.samSpec = ( self.samSpec +
                           np.outer(self.constituentSpec[params[p]+' D'],
                                    (1-hydration)*simSam[params[p]]  ) )
                if np.sum(np.isnan(self.samSpec))>0:
                    print(str(p) + ': ' + str(np.sum(np.isnan(self.samSpec))))
                    return
            if params[p] in self.constituents: #if there are no deviations it will be unappended
                self.samSpec = ( self.samSpec +
                           np.outer(self.constituentSpec[params[p]],
                                    simSam[params[p]]  ) )
                if np.sum(np.isnan(self.samSpec))>0:
                    print(str(p) + ': ' + str(np.sum(np.isnan(self.samSpec))))
                    return

        for iP in self.proteins:
            for iAA in range(self.AA_comp.shape[1]):
                cAA = self.AA_comp.columns[iAA]
                if cAA + ' D' in self.constituents:
                    self.samSpec = ( self.samSpec + simSam[iP].values.reshape(-1)*
                               np.outer(self.constituentSpec[cAA + ' D'],
                                        self.AA_comp.loc[ iP, cAA]*(1-hydration) ) )
                    if np.sum(np.isnan(self.samSpec))>0:
                        print(str(iP) + ' ' + str(iAA) +  ': ' +
                              str(np.sum(np.isnan(self.samSpec))))
                        return
                if cAA + ' H' in self.constituents:
                    self.samSpec = ( self.samSpec + simSam[iP].values.reshape(-1)*
                               np.outer(self.constituentSpec[cAA + ' H'],
                                        self.AA_comp.loc[ iP, cAA]*hydration ) )
                    if np.sum(np.isnan(self.samSpec))>0:
                        print(str(iP) + ' ' + str(iAA) +  ': ' +
                              str(np.sum(np.isnan(self.samSpec))))
                        return
        mn = np.mean(self.samSpec, axis=0)
        mn[ mn==0 ] = 1
        self.samSpec = self.samSpec/mn*self.signalInt

        nans = np.isnan(self.samSpec)
        self.samSpec[nans] = 0
        self.shotnoise = ( stats.norm.rvs( 0 , 1, size=( self.samSpec.shape )
                                  )*(self.samSpec + self.simBack )**0.5 )
        print('Spectra Simulation Complete')
        if np.sum(nans)>0:
            print('Number of Nans: ' + str(np.sum(nans)/
                                           self.samSpec.shape[0]))
        return
