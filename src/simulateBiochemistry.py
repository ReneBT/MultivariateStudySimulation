# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 03:50:57 2021

@author: Rene
"""
import numpy as np
import pandas as pd
from scipy import stats

class simulateBiochemistry:
    def __init__(
        self,
        simCop,
        trends,
        ):
        self.simCop =  simCop
        self.trends = trends
        self.simDat =  pd.DataFrame(np.copy(simCop), columns =
        ['Calorie Intake', 'Calorie Expenditure', 'Purchasing Power', 'Saturated Fat',
         'PUFA', 'Antiox', 'Glucose', 'Fructose', 'Glucose Control', 'Age'])
        self.diabetes_risk = pd.DataFrame(
            np.empty(( self.trends['Sample Size'].astype('int32'), 7 )),
            columns = ['Genetic', 'Calorie Imbalance', 'Activity', 'Saturated Fat',
                       'PUFA', 'Glucose', 'Overall']
            )
        self.simSpec = [] # we will assume fantastically accurate baseline correction! enough complications already
        self.simShot = []
        self.simBase = []

        return

    def simData( self, YM_measurement_error = 0.025 ):
### Demographic information
        #create new object, not link
        self.YM_measurement_error = YM_measurement_error

        #now impute marginal distributions on each variable
        self.simDat['Calorie Intake'] = stats.norm.ppf(self.simCop[:, 0],
                                                self.trends['Calories intake /day Mean'],
                                                self.trends['Calories intake /day SD'])

        self.simDat['Calorie Expenditure'] = stats.norm.ppf(self.simCop[:, 1],
                                                self.trends['Calories burnt /day Mean'],
                                                self.trends['Calories burnt /day SD'])

        # purchasing power
        self.simDat['Purchasing Power'] = stats.skewnorm.ppf(self.simCop[:, 2], #q
                                                            self.trends['Income Skew'], #a
                                                self.trends['Purchasing Power Location'], #loc
                                                self.trends['Purchasing Power Scale'])  #scale
        # saturated fat intake
        self.simDat['Saturated Fat'] = stats.skewnorm.ppf(self.simCop[:, 3],
                                                self.trends['Saturated fat g/day Skew'],
                                                self.trends['Saturated fat g/day Location'],
                                                self.trends['Saturated fat g/day Scale'], )
        self.simDat['Saturated Fat'].loc[self.simDat['Saturated Fat']<2] = 2 #lower bound is 2
        # polyunsaturated fat intake
        self.simDat['PUFA'] = stats.skewnorm.ppf(self.simCop[:, 4],
                                                self.trends['Unsaturated fat g/day Skew'],
                                                self.trends['Unsaturated fat g/day Location'],
                                                self.trends['Unsaturated fat g/day Scale'], )
        self.simDat['PUFA'].loc[self.simDat['PUFA']<1] = 1 #lower bound is 1

        # antioxidant intake
        self.simDat['Antiox'] = stats.skewnorm.ppf(self.simCop[:, 5],
                                                self.trends['Antioxidants Skew'],
                                                self.trends['Antioxidants Location'],
                                                self.trends['Antioxidants Scale'], )
        self.simDat['Antiox'].loc[self.simDat['Antiox']<0.1] = 0.1 #lower bound is 0.1

        # Glucose intake
        self.simDat['Glucose'] = stats.norm.ppf(self.simCop[:, 6],
                                                self.trends['Glucose Mean'],
                                                self.trends['Glucose SD'])
        self.simDat['Glucose'].loc[self.simDat['Glucose']<0] = stats.norm.rvs( 10, 2,
                                                   size=( np.sum(self.simDat['Glucose']<0)))  #convert neagtive values to a low positive value

        # fructose intake
        self.simDat['Fructose'] = stats.norm.ppf(self.simCop[:, 7], self.trends['Fructose Mean'],
                                                self.trends['Fructose SD'])
        self.simDat['Fructose'].loc[self.simDat['Fructose']<np.min([10, self.trends['Fructose Mean']/4])] = np.min([10, self.trends['Fructose Mean']/4]) #lower bound is 10 g per day or 1/4 of mean, whichever is lower

        # glucose control
        self.simDat['Glucose Control'] = stats.norm.ppf(self.simCop[:, 8],
                                                self.trends['Glucose control Mean'],
                                                self.trends['Glucose control SD'], )
        self.simDat['Glucose Control'].loc[self.simDat['Glucose Control']<0] = 0 #lower bound is all glucose available for reaction
        self.simDat['Glucose Control'].loc[self.simDat['Glucose Control']>0.95] = 0.95 #upper bound is 2% glucose available for reaction - even well controlled diabetes has small additional risk above non-diabetes

        # Age ppf(q, a, c, loc=0, scale=1)
        self.simDat['Age'] = stats.exponweib.ppf(self.simCop[:, 9], #q
                                                self.trends['Age Exponent'], # exponent
                                                self.trends['Age Shape'], # shape
                                                self.trends['Age Location'], # loc, minimum recruitment age
                                                self.trends['Age Scale']) # scale

### Risk of relevant disorders
        # layer 2 is the outcome of the biological mechanisms consequent upon the
        # underlying demographics
        # loosely based on webMD https://www.webmd.com/diabetes/risk-diabetes

        self.diabetes_risk['Genetic'] = stats.norm.rvs( 0.5, 0.1935,
                                                   size=( self.trends['Sample Size'].astype('int32')
, 1 ) ).reshape(-1)
        self.diabetes_risk['Calorie Imbalance'] = (self.simDat['Calorie Intake'] - self.simDat['Calorie Expenditure'])/1000
        self.diabetes_risk['Activity'] = (2500-self.simDat['Calorie Expenditure'])/2500
        self.diabetes_risk['Saturated Fat'] = (self.simDat['Saturated Fat']-5)/250
        self.diabetes_risk['PUFA'] = (18-self.simDat['PUFA'])/16
        self.diabetes_risk['Glucose'] = (self.simDat['Glucose']-60)/80
        self.diabetes_risk['Overall'] = self.diabetes_risk.iloc[
            :, :self.diabetes_risk.shape[1]-2].sum(axis=1)

        self.simDat['Diabetes'] = self.diabetes_risk['Overall']>2.3
        # Diabetes prevalence in UK is 4.8M out of 66M (https://www.diabetes.org.uk/professionals/position-statements-reports/statistics/diabetes-prevalence-2019), i.e. 7% so aim for around this mark

        # create a measure of cummulative exposure to reactive metabolic
        # intermediates based on balance between calorie intake and consumption
        # multiplitive factor centred around 1.1 with SD of 0.03-0.055 dependeing on decade
        self.simDat['Glucose Dwelltime'] = ( 1.1 + (
            self.diabetes_risk['Calorie Imbalance'] )/5 )

        # create a general multiplicative sensitivity to oxidative stress across
        # all AGE/ALEs times the individuals metabolic dwell time.
        # This is effectively an unknown DGP summarised on a population level
        self.simDat['Oxidative Sensitivity'] = stats.norm.rvs( 1, 0.05,
                                                             size=(self.trends['Sample Size'].astype('int32'),
                                                                   1 )
                                            ).reshape(-1) * self.simDat['Glucose Dwelltime']
        # scar tissue
        # replace simulated Collagen 1 and Elastin - depends on age and activity
        #       random susceptibility to scaring - multiplicative mean 1, range 0.95-1.05
        #       ageing general wear and tear- baseline taken as 16 (assume minimal tissue renewal after), then square root of time in maturity, times 3 to give range of about 15%
        #       injury - cummulative risk with age and activity. rate declines with age, but scars don't repair once laid down so monotonic relationship
        self.simDat['Risk of Injury'] =  ( 1/(1+np.exp((self.simDat['Age']/10)-4)) *
                       (1-self.diabetes_risk['Calorie Imbalance']/2) * #difference in calories in and burned (negative and marginal calorie intake associated with high risk activity and lifestyle)
                       self.simDat['Calorie Expenditure']/2500 # relative calorie output (total output indicates more activity)
                       ) * stats.norm.rvs( 1, 0.1, size=( self.trends['Sample Size'].astype('int32')))

        self.simDat['Proportion of Scarring'] = ( stats.norm.rvs( 1, 0.015, size=( self.trends['Sample Size'].astype('int32'))) # general susceptibility to scarring
                       *( (self.simDat['Age']**0.5-4)*3 + # age related (>16 yo) scarring from general wear and tear
                         (self.simDat['Risk of Injury']>0.5)*(self.simDat['Calorie Expenditure'])/150 # injury scaled against activity
                         )
                       )/100
### Biochemical composition of ligament
        # Variables that affect composition of ligament -
        #       proportion collagen and elastin (genetic, constant probabilities).
        #                   Independent of dietary and economic factors. Model each protien independently
        #       Injury   - risk and severity (above threshold) related to energy expenditure, cause
        #                   scaring with higher beta sheet content
        #       ageing   - gradual increase in random structure with age, interaction with AGE/ALE modifications
        #       GH-1 ALE - positively associated with PUFA intake
        #       CEP ALE  - weakly positively associated with SFA intake, more strongly with PUFA
        #       CML AGE  - strongly associated with glucose and fructose intake.
        #                   Boosted levels in metabolic dysfunction, glucose effect
        #                   (not fructose) moderated by control.
        #                   Fructose induces at same rate as uncontrolled dysfunctional glucose
        #       Pentosidine - only glucose, affected by control. cross linker so very strong effect on physical properties
        # HPO4_2H2O     Formation of microcrystals on CML, strongly linked to CML, but extent influenced by age.
        # MGO -       strongly associated with fructose, weakly with Glucose
        #

        # Collagen I - https://pdfs.semanticscholar.org/f31e/0932a2f35a6d7feff20977ce08b5b5398c60.pdf
        # range 65-80 %.
        self.simDat['Collagen'] = stats.norm.rvs( 0.72, 0.04,
                                                   size=( self.trends['Sample Size'].astype('int32')))
        # Elastin - https://pdfs.semanticscholar.org/f31e/0932a2f35a6d7feff20977ce08b5b5398c60.pdf
        # range 1-2 %. Proteoglycan/water matrix makes up remainder.
        self.simDat['Elastin'] = stats.norm.rvs( 0.015, 0.003,
                                                   size=( self.trends['Sample Size'].astype('int32')))



        self.simDat['Scar'] = self.simDat[['Collagen', 'Elastin']].sum(axis=1)*self.simDat['Proportion of Scarring'] #
        # Proteoglycan/water - https://pdfs.semanticscholar.org/f31e/0932a2f35a6d7feff20977ce08b5b5398c60.pdf
        # matrix makes up remainder. % will not appear in spectrum but does
        # lubricate joint so associated with joint movement
        self.simDat['Matrix'] = (np.ones( self.trends['Sample Size'].astype('int32') ) -
                             self.simDat[ [ 'Collagen', 'Elastin' ] ].sum(axis=1))
        self.simDat['Hydration']= (4*(self.simDat['Matrix']))**0.15 * stats.norm.rvs(
            1, 0.02, size=( self.trends['Sample Size'].astype('int32')))# hydration is weakly influenced at high matrix concentrations (matrix in excess), but impact accelerates at low levels. Modulated by pH below
        self.simDat['Hydration'] = self.simDat['Hydration'].fillna(0.65)
        self.simDat['Collagen'] = self.simDat['Collagen']*(1-self.simDat['Proportion of Scarring']) #replace original proteins to keep total constant
        self.simDat['Elastin'] = self.simDat['Elastin']*(1-self.simDat['Proportion of Scarring']) #replace original proteins to keep total constant

### Non-enzymatic Protein Modification

        # CEP ALE affected by PUFA and SFA. SD of effect is 0.05
        self.simDat['CEP'] = ((self.simDat['Saturated Fat']/100 +
                                             self.simDat['PUFA']/40) *
                                            self.simDat['Oxidative Sensitivity'] *
                                            self.simDat['Age']**2/1250)/750

        # CML AGE background glucose rate 1% converted, but in uncontrolled
        #           diabetes this escalates x10 but can be moderated by good control
        #           fructose acts like diabetic glucose but without modulation by control
        # A more realistic DGP would need to determine
        # lifetime exposure, including changes in exposure to glucose, fructose
        # and length of diabetes
        self.simDat['CML'] = ( ( ( self.simDat['Glucose']/50 *
                                                self.simDat['Age']**1.9/2400 +
                                                self.simDat['Glucose']/11 *
                                                self.simDat['Diabetes']*
                                                (1-self.simDat['Glucose Control'])
                                                )  +
                                              self.simDat['Fructose']/32 *
                                              self.simDat['Age']**2.1/5400) *
                                            self.simDat['Oxidative Sensitivity'] )/3000
        self.simDat['CEL'] = ( ( ( self.simDat['Glucose']/35 *
                                                self.simDat['Age']**2.03/1111 +
                                                self.simDat['Glucose']/9 *
                                                self.simDat['Diabetes']*
                                                (1-self.simDat['Glucose Control'])
                                                )  +
                                              self.simDat['Fructose']/20 *
                                              self.simDat['Age']**2.03/2500) *
                                            self.simDat['Oxidative Sensitivity'] )/5000

        self.simDat['pH'] = (  stats.norm.rvs( 7.25, 0.12,
                                                   size=( self.trends['Sample Size'].astype('int32')))
                            - 5*self.simDat['CML'] - 8*self.simDat['CEL'] ) # CML ionizes easier than native residues so depresses local pH slightly
        self.simDat['pH'].loc[(self.simDat['pH']-7)>1] = 8 # don't allow out of range values
        self.simDat['pH'].loc[(self.simDat['pH']-7)<-1] = 6 # don't allow out of range values
        hydrationAdj = 10**( ( self.simDat['pH']-7 )**2 )/10
        self.simDat['Hydration'] = self.simDat['Hydration'] - hydrationAdj # pH modulates the hydration of the protein local environment
        self.simDat['Hydration'].loc[self.simDat['Hydration']<0.65] = 0.65

        # HPO4_2H2O - CML acts as nucleation site for mineral crystallisation. Simulated as
        # exponential growth against CML & CEL with age.
        self.simDat['HPO4_2H2O'] = ( self.simDat['CML'] *
                                                (self.simDat['Age']**1.9)/1400 *
                                                (self.simDat['pH']-6.25) +
                                    self.simDat['CEL'] *
                                                (self.simDat['Age']**2.1)/1300 *
                                                (self.simDat['pH']-6.15) ) /10


        # MGO -       strongly associated with fructose, weakly with Glucos (1/4 rate of CML).
        self.simDat['MGO'] = ( ( ( self.simDat['Glucose']/300)  +
                                            (self.simDat['Fructose']/10 ) +
                                            ( self.simDat['PUFA']/10 )  +
                                            ( self.simDat['Saturated Fat']/25 ) ) *
                                          self.simDat['Oxidative Sensitivity'] /2250)
        self.simDat['GH1'] = 0.2*self.simDat['MGO']+( self.simDat['PUFA']/20 *
                                          self.simDat['Oxidative Sensitivity'] *
                                          self.simDat['Age']**2/2000)/500 #MGO is one route for this AGE
        # Pentosidine - only glucose, half rate of CML
        self.simDat['Pentosidine'] = ( (  self.simDat['Glucose']/200*
                                                    self.simDat['Age']**2/1250 ) *
                                                  self.simDat['Oxidative Sensitivity'] )/2000
        self.simDat[self.simDat<0] = 0 # don't allow negative concentrations
### Stiffness of ligament
        # young's modulus is
        #   Elastin:
        #       450 kPa for Elastin - MITHIEUX, ADVANCES IN PROTEIN CHEMISTRY, Vol. 70 p 437
        #       Molecular model for elastin structure and function, WR Gray, LB SANDBERG, JA FOSTER - Nature, 1973 - Springer
        #           … In our model, the principal site of hydration is the peptide chain rather than amino … thermodynamic
        #           parameters of elastin stretching11 • 12 • When dried, elastin becomes brittle, its elastic modulus
        #           being three orders of magnitude greater7 than that of the hydrated form
        #       Hydration:      7*x**3 + 0.28*x**2 + 0.17*x -0.80 (tweaked Collagen trend fitted from ethanol data to give appropriate dynamic range
        #       CML - reduce by 1/3 (Yoshinaga(2012). Journal of Investigative Dermatology, 132(2), 315–323. doi:10.1038/jid.2011.298 )
        #       Pentosidine:    0.006*P**4 - 0.02*P**3 - 0.01*P**2 + 0.2*P + 1 assume elastin modulus evcen more sensitive to cross linking - higher order polynomial than for collagen
        #   Collagen:
        #       Hydration:      1.2*x**3 + 0.38*x**2 + 0.35*x + 0.34 (fitted from ethanol data in Biophysical Journal Volume 97 December 2009 2985–2992 Tuning the Elastic Modulus of Hydrated Collagen Fibrils Colin A. Grant, † David J. Brockwell, † Sheena E. Radford, † and Neil H. Thomson)
        #       HPO42H2O as Ionic Strength:
        #       CML/CEL:    scaling factor = 1 + 0.0024*CML per uMol/g (based on gelatin, Spannenberg 2010, J Agri Food Chem 58, 3580-3585)
        #       Pentosidine:    0.01*P**3 - 0.02*P**2 + 0.1*P + 1 (Experimental Diab. Res., 5:143–153, 2004ISSN: 1543-8600 print / 1543-8619, converted to scaling relative to zero pentosidine) - multiply YM calculated already
        #       GH1:            MGO pathway reduces YM by half on average (Retamal et al J Periodontol 2017 Vol 88, by reducing adhesion between fibrils)
        #       pH:             independent random variable with small contribution from CML and HPO4
        #   Scar:
        #        shoud be proportionally bigger effect as drop in fibril alignment
        #        and so cooperativity is reduced which will not be as obvious in the spectrum
        #        Reeves(2009). Journal of Biomechanics, 42(7), 797–803. doi:10.1016/j.jbiomech.2009.01.030
        #       ' Young’s modulus was significantly lower in the OP tendons by 24%' which would be mostly formation of new scar tissue

        self.simDat['Youngs Modulus Elastin'] = (
            (self.simDat['Elastin']*( 10**(1.1*(1-self.simDat['Hydration'])**3 +
                                    1.2*(1-self.simDat['Hydration'])**2 +
                                    (1-self.simDat['Hydration']) -
                                    0.34 ) ) #set base modulus for elastin based on hydration
            *
            (50*(self.simDat['CML']+self.simDat['CEL'])+1)#check that the simulated values give expected adjustment of up to 1/3
            *
            (200000*self.simDat['Pentosidine']**2 +
             -170*self.simDat['Pentosidine'] +
             1 )
            *
            (15*self.simDat['Scar']+1)
            *
            2**(0.01*self.simDat['pH']**2 - 0.31*self.simDat['pH'] + 1.7))
            )

        self.simDat['Youngs Modulus Collagen'] = (
            self.simDat['Collagen']*( 10**(1.06*(1-self.simDat['Hydration'])**3 +
                                    0.38*(1-self.simDat['Hydration'])**2 +
                                    0.34*(1-self.simDat['Hydration']) +
                                    0.33 )) #set base modulus for collagen based on hydration
            *
            (35*(self.simDat['CML']+self.simDat['CEL'])+1)
            *
            ( 5500*self.simDat['Pentosidine']**3 -
             3400*self.simDat['Pentosidine']**2 +
             75*self.simDat['Pentosidine'] +
             1 )
            /
            (self.simDat['GH1']*30+1)#check that the simulated values give expected adjustment of up to x0.5
            *
            2**(0.07*self.simDat['pH']**2 - 1.14*self.simDat['pH'] + 4.6)
            *
            (2*self.simDat['Scar']+1)
            )

        self.simDat['Youngs Modulus Measurement Error'] =  (
              10**stats.norm.rvs( 0, self.YM_measurement_error,
                                 size=( self.trends['Sample Size'].astype('int32'))
                             ))
        self.simDat['Youngs Modulus'] = (
            self.simDat['Youngs Modulus Measurement Error'] *
            self.simDat['Youngs Modulus Collagen'] *
            (1+self.simDat['Youngs Modulus Elastin'])**2
            )

        return self.simDat


## Pathology
    def simulatePathology( self, cutoff = 2, pathology_measurement_error = 0.05):
        self.pathology_measurement_error = pathology_measurement_error
        unexplainedVar = stats.norm.rvs( 0, pathology_measurement_error,
                                   size=( self.simDat['Youngs Modulus'].shape)
                                  )
        self.pathology_risk = ( self.simDat['Youngs Modulus'] +
                        unexplainedVar +
                        (self.diabetes_risk['Genetic']/5 +
                        self.diabetes_risk['Calorie Imbalance'] -
                        self.diabetes_risk['Activity']*2 +
                        self.diabetes_risk['Saturated Fat']-
                        self.diabetes_risk['PUFA']/2+
                        (1-self.simDat['Glucose Control'])*
                        self.simDat['Diabetes']*2-
                        self.simDat['Antiox']/5)
                        )
        self.pathology = self.pathology_risk > cutoff
        return