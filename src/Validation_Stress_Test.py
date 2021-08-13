# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 09:43:22 2020

@author: J Renwick Beattie
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from src.file_locations import data_folder, images_folder
from src import specSim, simulateBiochemistry, modelData
from operator import itemgetter
#  This script simulates a series of cohorts in an ongoing epidemiological
# studyevaluating the impact of diet on health. As part of this study some
# samplesof ligaments are analysed using Raman spectroscopy to determine if
# the methodcan be used to predict joint stiffness. The study collects cohort
# data every decade.

### Defaults
cohorts =['Training Population', 'Validation Cohort 1',
          'Validation Cohort 2', 'Validation Cohort 3']
# list of names of trial repeats to be read in from excel sheet
N_cohorts = np.shape(cohorts)[0]
N_reps = [1, 100, 100, 100]
statsRec = ['YM_R2', 'YM_R2_SD', 'YM_R2_-', 'YM_R2_+',
            'AUCROC_Mean', 'AUCROC_SD', 'AUCROC_-', 'AUCROC_+',
            'Sensitivity_Mean', 'Sensitivity_SD', 'Sensitivity_-',
            'Sensitivity_+', 'Specificity_Mean',
            'Specificity_SD', 'Specificity_-', 'Specificity_+',
            'Percent Chance of Success']
fig_Resolution = 300  # desired output resolution of images
heterogeneity = 50  # typical domain diameter of sample variability
AUCROC_target = 0.87
sensitivity_target = 0.71
specificity_target = 0.83
# This simulation is comprised of multiple layers.
# Layer one is the underlying socioeconomic and dietary factors that influence
# biochemistry

trends = pd.read_excel(os.path.join(data_folder,
                                    'Data Generating Process.xlsx'),
                       sheet_name='Trends')
# to eliminate variation in any aspect its details can be made consistent in
# the xlsx file and loaded in instead e.g. uncomment below to fix the analytic
# method
WS = pd.read_excel(os.path.join(data_folder, 'Data Generating Process.xlsx'),
                   sheet_name=cohorts[0], )
WS_np = np.array(WS)
nCopVar = WS.columns.shape[0]-1
rho = np.ndarray((nCopVar, nCopVar, N_cohorts))
irho = np.ndarray((nCopVar, nCopVar, N_cohorts))
copula = np.ndarray((nCopVar, nCopVar, N_cohorts))
MAD_mean = np.ndarray(N_cohorts)
MAD_rho = np.ndarray(N_cohorts)
MAD_sd = np.ndarray(N_cohorts)
simCop = [None]*N_cohorts
simDat = [None]*N_cohorts
simSpec_noiseless = [None]*N_cohorts
simSpec = [None]*N_cohorts
simShot = [None]*N_cohorts
simBase = [None]*N_cohorts
ligament = [None]*N_cohorts
pathology = [None]*N_cohorts
YM_plsR2_val = [None]*N_cohorts
plsR2_diff = [None]*N_cohorts
AUCROC = [None]*N_cohorts
sensitivity = [None]*N_cohorts
specificity = [None]*N_cohorts
diabetes_risk = [None]*N_cohorts
diabetes = [None]*N_cohorts
Lig_plsR2_val = [None]*N_cohorts
MeanStats = pd.DataFrame(np.nan, index=range(N_cohorts+1), columns=statsRec)


# Load in parameters to simulate spectra
# Spectral parameters estimated from Glenn 2007 FASEB, Pawlak 2008 JRS,
# Beattie 2010 FASEB, Beattie 2011 IOVS and Caraher 2018
proteins = ['Collagen', 'Elastin']
postTranslation = pd.DataFrame({'Protein': ['Collagen', 'Elastin'],
                                'Original': ['Pro', 'Pro'],
                                'PTM': ['Hyp', 'Hyp'],
                                'Proportion': [0.5714, 0.1]})
# based on Neuman THE DETERMINATION OF COLLAGEN AND ELASTIN IN TISSUES 1950
# J. Biol. Chem. 1950, 186:549-556.

baseSpectra = specSim.specSim(proteins, postTranslation, np.arange(500, 1801))
baseSpectra.print_Brick(10, fig_Resolution)
print('Base spectra generated')

np.random.seed(8173635)

### generate simulated studies
for iC in range(N_cohorts):
    ###    Underlying dietary and economic factors
    WS = pd.read_excel(os.path.join(data_folder,
                                    'Data Generating Process.xlsx'),
                       sheet_name=cohorts[iC], )
    WS_np = np.array(WS)
    rho[:, :, iC] = (WS_np[:, 1:11]+WS_np[:, 1:11].T).astype('float')
    np.fill_diagonal(rho[:, :, iC], 1)
    irho[:, :, iC] = np.sin(np.array(rho[:, :, iC])*np.pi/6)
    # calculate inverse rho
    print('Cycle ' + str(iC+1) + ' Inverse correlation calculated')

    if ~np.all(np.linalg.eigvals(irho[:, :, iC]) > 0):
        # check correlation matrix is valid
        print('Rank Correlation matrix is not positive semi definite: ' +
              cohorts[iC])
        # find closest valid matrix by reconstructing from positive eigenvalue
        #  PCs(note that badly conditioned matrices may take more iterations)
        eigval, eigveci = np.linalg.eigh(irho[:, :, iC])
        eigvalr, eigvec = np.linalg.eigh(rho[:, :, iC])
        rhoU = np.triu(np.inner(np.inner(rho[:, :, iC], eigvec[eigval > 0, :]),
                                eigvec[eigval > 0, :].T))
        # since sparse PCA so result is likely assymmetric, use upper triangle
        # and its transpose to create symmetric version
        rho[:, :, iC] = rhoU+rhoU.T
        # sparse PCA will not have reconstructed the diagnonal of ones
        np.fill_diagonal(rho[:, :, iC], 1)
        irho[:, :, iC] = np.sin(np.array(rho[:, :, iC])*np.pi/6)
        # calculate inverse rho

    # calculate mixing matrix required to achieve desired correlation
    copula[:, :, iC] = np.linalg.cholesky(irho[:, :, iC])
    # standardise sum of squares to retain original SD
    copula[:, :, iC] = copula[:, :, iC]/np.sum(copula[:, :, iC]**2,
                                               axis=1)**0.5
    # simulate correlated data (this is done on gaussian distributed data
    # to achieve desired joint distribution)
    # for raw random variables stick with default mean 0 and SD 1

    simCop[iC] = np.inner(stats.norm.rvs(
        size=(trends['Sample Size'][iC].astype('int32'), 10)),
        (copula[:, :, iC]))
    MAD_mean[iC] = np.mean(np.abs(np.mean(simCop[iC], axis=0)))
    # sanity check on achieved mean before reversion to uniform (target = 0)
    MAD_sd[iC] = np.mean(np.abs(np.std(simCop[iC], axis=0)-1))
    # sanity check on achieved SD before reversion to uniform (target = 1)

    simCop[iC] = stats.norm.cdf(simCop[iC])
    # convert to uniform for conversion to marginal distributions

    testcorr = stats.spearmanr(simCop[iC])[0]
    MAD_rho[iC] = np.mean(np.abs(rho[:, :, iC]-testcorr))
    # sanity check on achieved rank correlation
    print('Cycle ' + str(iC+1) + ' Copula calculated')

    simDat[iC] = [None]*N_reps[iC]
    YM_plsR2_val[iC] = [None]*N_reps[iC]
    pathology[iC] = [None]*N_reps[iC]
    AUCROC[iC] = [None]*N_reps[iC]
    sensitivity[iC] = [None]*N_reps[iC]
    specificity[iC] = [None]*N_reps[iC]
    for iR in range(N_reps[iC]):
        simDat[iC][iR] = simulateBiochemistry.simulateBiochemistry(simCop[iC],
                                                        trends.iloc[iC, :], )
        simDat[iC][iR].simData(YM_measurement_error=0.01)

        ### Spectra of ligament
        # spectral data very memory intensive so overwrite on each repetition
        # to store only the most recent
        baseSpectra.specSample(simDat[iC][iR].simDat.loc[:, 'Collagen':],
                    (simDat[iC][iR].trends['Spot Size']/heterogeneity)**3,
                               simDat[iC][iR].trends['Mean total Counts'])
        meanInt = np.mean(baseSpectra.samSpec, axis=0)
        simSpec_noiseless[iC] = baseSpectra.samSpec / meanInt
        simSpec[iC] = (baseSpectra.samSpec + baseSpectra.shotnoise)/meanInt
        # we will assume fantastically accurate baseline correction! enough
        # complications already
        simShot[iC] = baseSpectra.shotnoise/meanInt
        simBase[iC] = baseSpectra.simBack/meanInt

        simDat[iC][iR].simulatePathology(cutoff=2.25,
                                         pathology_measurement_error=0.05)
        if iC == 0:
            ### PLS regression between spectra and Young's modulus
            YM_plsModel = modelData.modelData(test_size=0.25,
                                              random_state=3476,
                                              max_iter=100, nFold=5)
            YM_plsModel.fit(simDat[iC][iR], simSpec[iC])
            MeanStats['YM_R2'].iloc[iC] = np.mean(
                YM_plsModel.YM_cv_results['test_score'])
            #R2 is already squared so expressed as variance, so
            # straightforward mean
            MeanStats['YM_R2_SD'].iloc[iC] = (np.std(
                YM_plsModel.YM_cv_results['test_score']))
            MeanStats['YM_R2_-'].iloc[iC] = (MeanStats['YM_R2'].iloc[iC] -
                                         1.98*MeanStats['YM_R2_SD'].iloc[iC])
            MeanStats['YM_R2_+'].iloc[iC] = (MeanStats['YM_R2'].iloc[iC] +
                                         1.98*MeanStats['YM_R2_SD'].iloc[iC])
            MeanStats['YM_R2'].iloc[iC+1] = YM_plsModel.YM_pls_R2_holdout

            MeanStats['AUCROC_Mean'].iloc[iC] = YM_plsModel.AUCROC_cal
            MeanStats['Sensitivity_Mean'].iloc[iC] = (
                YM_plsModel.sensitivity_cal)

            MeanStats['Specificity_Mean'].iloc[iC] = (
                YM_plsModel.specificity_cal)

            MeanStats['AUCROC_Mean'].iloc[iC+1] = YM_plsModel.AUCROC_val
            MeanStats['Sensitivity_Mean'].iloc[iC+1] = (
                YM_plsModel.sensitivity_val)

            MeanStats['Specificity_Mean'].iloc[iC+1] = (
                YM_plsModel.specificity_val)

        else:
            (YM_plsR2_val[iC][iR], AUCROC[iC][iR], sensitivity[iC][iR],
             specificity[iC][iR]) = YM_plsModel.project_models(simDat[iC][iR],
                                                            simSpec[iC])
        print('Cycle ' + str(iC + 1) + ' Iteration ' + str(iR + 1) +
              ' of ' + str(N_reps[iC]))
    if iC>0:
        MeanStats['YM_R2'].iloc[iC+1] = np.mean(YM_plsR2_val[iC])
        MeanStats['YM_R2_SD'].iloc[iC+1] = np.std(YM_plsR2_val[iC])
        # DO NOT TURN INTO SE! WANT TO QUANTIFY GROSS VARIABILITY
        MeanStats['YM_R2_-'].iloc[iC+1] = (
            MeanStats['YM_R2'].iloc[iC+1] -
            1.98*MeanStats['YM_R2_SD'].iloc[iC+1])
        MeanStats['YM_R2_+'].iloc[iC+1] = (
            MeanStats['YM_R2'].iloc[iC+1] +
            1.98*MeanStats['YM_R2_SD'].iloc[iC+1])
        MeanStats['AUCROC_Mean'].iloc[iC+1] = np.mean(AUCROC[iC])
        MeanStats['AUCROC_SD'].iloc[iC+1] = np.std(AUCROC[iC])
        MeanStats['AUCROC_-'].iloc[iC+1] = (
            MeanStats['AUCROC_Mean'].iloc[iC+1] -
            1.98*MeanStats['AUCROC_SD'].iloc[iC+1])
        MeanStats['AUCROC_+'].iloc[iC+1] = (
            MeanStats['AUCROC_Mean'].iloc[iC+1] +
            1.98*MeanStats['AUCROC_SD'].iloc[iC+1])
        MeanStats['Sensitivity_Mean'].iloc[iC+1] = np.mean(sensitivity[iC])
        MeanStats['Sensitivity_SD'].iloc[iC+1] = np.std(sensitivity[iC])
        MeanStats['Sensitivity_-'].iloc[iC+1] = (
            MeanStats['Sensitivity_Mean'].iloc[iC+1] -
            1.98*MeanStats['Sensitivity_SD'].iloc[iC+1])
        MeanStats['Sensitivity_+'].iloc[iC+1] = (
            MeanStats['Sensitivity_Mean'].iloc[iC+1] +
            1.98*MeanStats['Sensitivity_SD'].iloc[iC+1])
        MeanStats['Specificity_Mean'].iloc[iC+1] = np.mean(specificity[iC])
        MeanStats['Specificity_SD'].iloc[iC+1] = np.std(specificity[iC])
        # DO NOT TURN INTO SE! WANT TO QUANTIFY GROSS VARIABILITY
        MeanStats['Specificity_-'].iloc[iC+1] = (
            MeanStats['Specificity_Mean'].iloc[iC+1] -
            1.98*MeanStats['Specificity_SD'].iloc[iC+1])
        MeanStats['Specificity_+'].iloc[iC+1] = (
            MeanStats['Specificity_Mean'].iloc[iC+1] +
            1.98*MeanStats['Specificity_SD'].iloc[iC+1])
        MeanStats['Percent Chance of Success'].iloc[iC+1] = np.mean(
            np.logical_and(
                np.logical_and(np.array(AUCROC[iC]) > AUCROC_target,
                    np.array(sensitivity[iC]) > sensitivity_target),
                    np.array(specificity[iC]) > specificity_target))*100

    print('Cycle ' + str(iC + 1) + ' of ' + str(N_cohorts) +
          ' is complete.')
### Stress Testing: Sample Size
iC = 1  # perturb the validation on same population
N = np.arange(100,2600,100)
simDat_Sam = [None]*N_reps[iC]
simSpec_Sam = [None]*N_reps[iC]
YM_plsR2_Sam = np.empty((len(N), N_reps[iC]))
AUCROC_Sam = np.empty((len(N), N_reps[iC]))
sensitivity_Sam = np.empty((len(N), N_reps[iC]))
specificity_Sam = np.empty((len(N), N_reps[iC]))
trendsNew = trends.copy(deep=True).iloc[1]
trendsNew['Sample Size'] = np.max(N)
simCop_Sam = np.inner(stats.norm.rvs(
    size=(trendsNew['Sample Size'].astype('int32'), 10)),
    (copula[:, :, 1]))
simCop_Sam = stats.norm.cdf(simCop_Sam)
for iR in range(N_reps[iC]):
    simDat_Sam[iR] = simulateBiochemistry.simulateBiochemistry(
        simCop_Sam, trendsNew, )
    simDat_Sam[iR].simData(YM_measurement_error = 0.01)
    simDat_Sam[iR].simulatePathology(cutoff=2.25,
                                     pathology_measurement_error = 0.05)
    baseSpectra.specSample(simDat_Sam[iR].simDat.loc[:, 'Collagen':],
                               (trendsNew['Spot Size']/heterogeneity)**3,
                               trendsNew['Mean total Counts'])
    meanInt = np.mean(baseSpectra.samSpec, axis=0)
    simSpec_Sam[iR] = (baseSpectra.samSpec + baseSpectra.shotnoise) / meanInt
    for iSam in range(len(N)):
        nSam = N[iSam]
        (YM_plsR2_Sam[iSam, iR], AUCROC_Sam[iSam, iR],
         sensitivity_Sam[iSam, iR], specificity_Sam[iSam, iR]) = (
             YM_plsModel.project_models(simDat_Sam[iR], simSpec_Sam[iR][:,:nSam]))
        print('Sample Size ' + str(nSam) + ' Iteration ' +
              str(iR+1) + ' of ' + str(N_reps[iC]))

figSensitivityPrecision, axSensitivityPrecision = plt.subplots(2, 2, figsize = [8, 8])
figSensitivityPrecision.subplots_adjust(top=0.98, left = 0.08, right=0.99, bottom=0.12,
                               hspace=0.25)
axSensitivityPrecision[0,0].plot(N, np.std(sensitivity_Sam, axis=1), '.')
axSensitivityPrecision[0,0].plot([1, np.max(N)+100], [0.02,0.02], '--k')
axSensitivityPrecision[0,1].plot(N, sensitivity_Sam, '.')
axSensitivityPrecision[0,1].plot([1, np.max(N)+100], np.tile(np.mean(sensitivity_Sam),(2)), '--k')
for i in range(0,9,2):
    adj = 0.01*(i-4)
    axSensitivityPrecision[1,0].plot(N, np.mean(
        sensitivity_Sam > np.mean(sensitivity_Sam)+adj, axis=1 ) * 100, '.')
    axSensitivityPrecision[0,1].plot([1, np.max(N)+100], np.tile(np.mean(sensitivity_Sam)+adj,(2)), '--',color=np.tile(0.5+adj*7,3))
    axSensitivityPrecision[0,1].plot(1, np.mean(sensitivity_Sam)+adj, '*')
    axSensitivityPrecision[1,1].plot(N, np.mean(np.logical_and(
        sensitivity_Sam > np.mean(sensitivity_Sam)+adj,
        specificity_Sam > np.mean(specificity_Sam)+adj), axis=1 ) * 100, '.')
axSensitivityPrecision[0, 0].set_ylabel('Standard Deviation in Sensitivity  (Arbitrary)')
axSensitivityPrecision[0, 1].set_ylabel('Sensitivity (Arbitrary)')
axSensitivityPrecision[1, 0].set_ylabel('% Chance of Success')
axSensitivityPrecision[1, 1].set_ylabel('% Chance of Success')
for i in range(2):
    for j in range(2):
        axSensitivityPrecision[i, j].set_xlabel('AGmber of Samples')

image_name = "Effect of Samples AGmbers on sensitivity.tif"
figSensitivityPrecision.savefig(os.path.join(images_folder, image_name),
                       dpi=fig_Resolution)        # plt.show()
plt.close()
print('Sample Size Complete')

### Stress testing: Signal to Noise and Sampling Area
# Spectral Signal to Noise
# Correlation [compare independent, inverse and correlation from new country]
# Age - do simple change in distribution parameters and also one where the
#       correlation matrix varies depending on mean age

iR = N_reps[iC]-1
NoiseLevels = np.hstack((np.arange(1, 4)/4, 4/np.arange(4, 0, -1)))
SignalCnts = simDat[iC][iR].trends['Mean total Counts']/NoiseLevels**2
SampleDiameter = simDat[iC][iR].trends['Spot Size']/NoiseLevels**2
nSNR = len(NoiseLevels)
simSpec_SNR = [None]*nSNR
YM_plsR2_SNR = [None]*nSNR
AUCROC_SNR = [None]*nSNR
sensitivity_SNR = [None]*nSNR
specificity_SNR = [None]*nSNR
simSpec_SampErr = [None]*nSNR
YM_plsR2_SampErr = [None]*nSNR
AUCROC_SampErr = [None]*nSNR
sensitivity_SampErr = [None]*nSNR
specificity_SampErr = [None]*nSNR

for iSNR in range(nSNR):
    simSpec_SNR[iSNR] = simSpec_noiseless[iC] + simShot[iC] * NoiseLevels[iSNR]
    (YM_plsR2_SNR[iSNR], AUCROC_SNR[iSNR], sensitivity_SNR[iSNR],
             specificity_SNR[iSNR]) = YM_plsModel.project_models(
                 simDat[iC][iR], simSpec_SNR[iSNR])
    baseSpectra.specSample(simDat[iC][iR].simDat.loc[:, 'Collagen':],
                           (SampleDiameter[iSNR]/heterogeneity)**3,
                           SignalCnts[0])  # use ultra high intensity spectra
    # to focus on sampling variability
    meanInt = np.mean(baseSpectra.samSpec, axis=0)
    meanInt[meanInt == 0] = 1  # dont allow divide by zero errors
    simSpec_SampErr[iSNR] = ((baseSpectra.samSpec + simShot[iC] *
                              NoiseLevels[0])/meanInt)

    (YM_plsR2_SampErr[iSNR], AUCROC_SampErr[iSNR], sensitivity_SampErr[iSNR],
             specificity_SampErr[iSNR]) = YM_plsModel.project_models(
                 simDat[iC][iR], simSpec_SampErr[iSNR])
    print('SNR Cycle ' + str(iSNR + 1) + ' of ' + str(nSNR))


figSNRSATrends, axSNRSATrends = plt.subplots(1, 2, figsize = [8, 4])
figSNRSATrends.subplots_adjust(top=0.98, left = 0.08, right=0.99, bottom=0.12,
                               hspace=0.25)
axSNRSATrends[0].plot(np.log10(SignalCnts), YM_plsR2_SNR, 'P')
axSNRSATrends[0].plot(np.log10(SignalCnts), AUCROC_SNR, 'X')
axSNRSATrends[0].plot(np.log10(SignalCnts), sensitivity_SNR, '*')
axSNRSATrends[0].plot(np.log10(SignalCnts), specificity_SNR, 'o')
axSNRSATrends[0].plot(np.tile(np.log10(
                            simDat[iC][iR].trends['Mean total Counts']), 2),
                      [0, 1], '--k', linewidth=0.5)
axSNRSATrends[0].set_ylabel('Value')
axSNRSATrends[0].set_xlabel('Log10 Signal Counts')
axSNRSATrends[0].set_ylim([0, 1])

axSNRSATrends[1].plot(np.log10(SampleDiameter), YM_plsR2_SampErr, 'P')
axSNRSATrends[1].plot(np.log10(SampleDiameter), AUCROC_SampErr, 'X')
axSNRSATrends[1].plot(np.log10(SampleDiameter), sensitivity_SampErr, '*')
axSNRSATrends[1].plot(np.log10(SampleDiameter), specificity_SampErr, 'o')
axSNRSATrends[1].plot(np.tile(np.log10(
                            simDat[iC][iR].trends['Spot Size']), 2),
                      [0, 1], '--k', linewidth=0.5)
axSNRSATrends[1].set_ylabel('Value')
axSNRSATrends[1].set_ylim([0, 1])
axSNRSATrends[1].set_xlabel('Log10 Sampling Area')
axSNRSATrends[1].legend(('R$^2$', 'AUCROC', 'Sensitivity', 'Specificity'),
                        fontsize='small')
subplts = ['a)', 'b)']
for c in range(2):
    axSNRSATrends[c].annotate(subplts[c],
            xy=(0.2, 0.04),
            xytext=(-0.01, 0.97),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=10,
            color=[0, 0, 0],
            horizontalalignment="right",
            va="top",
)

image_name = "Effect of Signal intensity and sampling area on validtion metrics.tif"
figSNRSATrends.savefig(os.path.join(images_folder, image_name),
                       dpi=fig_Resolution)        # plt.show()
plt.close()
print('SNR and Sampling Complete')

### Stress Testing: Glucose / Age
# Extend range of
glucMean = trends['Glucose Mean'].iloc[iC]
glucStep = np.round(glucMean-glucMean*0.9)
glucose = np.arange(glucMean-3*glucStep, glucMean+4*glucStep, glucStep)
ageMean = trends['Age Scale'].iloc[iC]
ageStep = np.round(ageMean-ageMean*0.9)
age = np.arange(ageMean-3*ageStep, ageMean+4*ageStep, ageStep)
nMF = len(age)
simDat_AG =[ [None]*nMF]*nMF
simSpec_AG  =[ [None]*nMF]*nMF
YM_plsR2_val_AG = np.empty([nMF, nMF])
AUCROC_AG = np.empty([nMF, nMF])
sensitivity_AG = np.empty([nMF, nMF])
specificity_AG = np.empty([nMF, nMF])

for iA in range(nMF):
    trendsNew = trends.copy(deep=True).iloc[iC]
    trendsNew['Age Scale'] = age[iA]
    trendsNew['Age Shape'] = 10 #look at smallish ranges of age
    for iG in range(nMF):
        trendsNew['Glucose Mean'] = glucose[iG]
        simDat_AG[iA][iG] = simulateBiochemistry.simulateBiochemistry(
            simCop[iC], trendsNew, )
        simDat_AG[iA][iG].simData(YM_measurement_error = 0.005)

        baseSpectra.specSample(simDat_AG[iA][iG].simDat.loc[:, 'Collagen':],
                                   (trendsNew['Spot Size']/heterogeneity)**3,
                                   trendsNew['Mean total Counts'])
        meanInt = np.mean(baseSpectra.samSpec, axis=0)
        simSpec_AG[iA][iG] = (baseSpectra.samSpec +
                              baseSpectra.shotnoise) / meanInt
        simDat_AG[iA][iG].simulatePathology(cutoff=2.25,
                                         pathology_measurement_error = 0.05)
        (YM_plsR2_val_AG[iA][iG], AUCROC_AG[iA, iG], sensitivity_AG[iA, iG],
        specificity_AG[iA, iG]) = YM_plsModel.project_models(simDat_AG[iA][iG],
                                                       simSpec_AG[iA][iG])
        print('Glucose Cycle ' + str(iG + 1) + ' of ' + str(nMF) + ' in '
              + 'Age Cycle ' + str(iA + 1))
    print('Age Cycle ' + str(iA + 1) + ' of ' + str(nMF) + ' complete.')

print('Age and Glucose simulated')

figAgeGlucoseTrends, axAgeGlucoseTrends = plt.subplots(2, 2, figsize = [8, 8])
figAgeGlucoseTrends.subplots_adjust(top=0.98, left = 0.08, right=0.99,
                                    bottom=0.12, hspace=0.25)
axAgeGlucoseTrends[0, 0].imshow(YM_plsR2_val_AG, extent=[0, nMF-1, 0, nMF-1],
                               vmin=0, vmax=1)
axAgeGlucoseTrends[0, 1].imshow(AUCROC_AG, extent=[0, nMF-1, 0, nMF-1],
                                vmin=0.5, vmax=1)
#dynamic range of AUC is 0.5-1.0
axAgeGlucoseTrends[1, 0].imshow(sensitivity_AG, extent=[0, nMF-1, 0, nMF-1],
                                vmin=0, vmax=1)
axAgeGlucoseTrends[1, 1].imshow(specificity_AG, extent=[0, nMF-1, 0, nMF-1],
                                vmin=0, vmax=1)
subplts = [['a)', 'b)'], ['c)', 'd)']]
for i in range(2):
    for j in range(2):
        axAgeGlucoseTrends[i, j].set_ylabel('Age / Years')
        axAgeGlucoseTrends[i, j].set_yticklabels(age)
        axAgeGlucoseTrends[i, j].set_xlabel('Glucose (g/day)')
        axAgeGlucoseTrends[i, j].set_xticklabels(glucose)
        axAgeGlucoseTrends[i, j].plot((nMF-1)*(trends['Glucose Mean'].iloc[1]-
                                               np.min(glucose))/np.ptp(glucose),
                                      (nMF-1)*(trends['Age Scale'].iloc[1]-
                                               np.min(age))/np.ptp(age), '+',
                                      color=[1, 0, 0])
        axAgeGlucoseTrends[i, j].plot((nMF-1)*(trends['Glucose Mean'].iloc[2]-
                                               np.min(glucose))/np.ptp(glucose),
                                      (nMF-1)*(trends['Age Scale'].iloc[2]-
                                               np.min(age))/np.ptp(age), 'o',
                                      color=[1, 0.3, 1])
        axAgeGlucoseTrends[i, j].plot((nMF-1)*(trends['Glucose Mean'].iloc[3]-
                                               np.min(glucose))/np.ptp(glucose),
                                      (nMF-1)*(trends['Age Scale'].iloc[3]-
                                               np.min(age))/np.ptp(age), '>',
                                      color=[1, 0.3, 1])
        axAgeGlucoseTrends[i, j].annotate(subplts[i][j],
                 xy=(0.2, 0.04),
                 xytext=(-0.01, 0.97),
                 textcoords="axes fraction",
                 xycoords="axes fraction",
                 fontsize=10,
                 color=[0, 0, 0],
                 horizontalalignment="right",
                 va="top",
                 )
image_name = "Effect of glucose and age on validtion metrics.tif"
figAgeGlucoseTrends.savefig(os.path.join(images_folder, image_name),
                            dpi=fig_Resolution)        # plt.show()
plt.close()
print('Age and Glucose complete')

### Sanity Checks: Ligament Biochemical Composition

Lig_X_train, Lig_X_test, Lig_y_train, Lig_y_test = train_test_split(
    simDat[0][0].simDat.loc[:, 'Collagen':'Pentosidine'],
    simDat[0][0].simDat.loc[:, 'Youngs Modulus'],
    test_size=0.25,
    random_state = 3476)
Lig_mean = np.mean(Lig_X_train, axis=0)

Lig_pls_by_ncomp = [None]*10
Lig_pls_r2val_ncomp = [None]*10
for iComp in range(1, 11):
    Lig_pls_by_ncomp[iComp-1] = PLSRegression(n_components = iComp,
                                              scale = False, max_iter=100)
    Lig_cv_results = cross_validate(Lig_pls_by_ncomp[iComp-1],
                                    Lig_X_train-Lig_mean, Lig_y_train, cv=5)
    Lig_pls_r2val_ncomp[iComp-1] = np.mean(Lig_cv_results['test_score'])
Lig_YM_plsModel_nComp = np.where(
    Lig_pls_r2val_ncomp==np.max(Lig_pls_r2val_ncomp))[0][0]+1
Lig_YM_plsModel = PLSRegression(n_components = Lig_YM_plsModel_nComp,
                                scale = False, max_iter=100)

Lig_YM_plsModel.fit(Lig_X_train-Lig_mean, Lig_y_train)
Lig_plsR2_cal = Lig_YM_plsModel.score(Lig_X_train-Lig_mean, Lig_y_train)
Lig_plsR2_val[0] = Lig_YM_plsModel.score(Lig_X_test-Lig_mean, Lig_y_test)
for iC in range(1, N_cohorts):
    Lig_plsR2_val[iC] = Lig_YM_plsModel.score(
        simDat[iC][iR].simDat.loc[:, 'Collagen':'Pentosidine']-
        Lig_mean, simDat[iC][iR].simDat.loc[:, 'Youngs Modulus'])
print('Ligament Correlations calculated')


### Sanity Checks: Population Characteristic Histograms
iR = [ 0, np.where(YM_plsR2_val[1]==np.min(YM_plsR2_val[1]))[0][0],
      np.where(YM_plsR2_val[2]==np.min(YM_plsR2_val[2]))[0][0],
      np.where(YM_plsR2_val[3]==np.min(YM_plsR2_val[3]))[0][0], ]
# reset repetition index to first rep

# Underlying dietary and economic factors

figSim, axSim = plt.subplots(4, 4, figsize = [8, 8])
figSim.subplots_adjust(top=0.99, left = 0.02, right=0.99, bottom=0.05,
                       hspace=0.25)
for iAx in range(16):
    for iC in range(N_cohorts):
        x = np.floor(iAx/4).astype('int')
        y = np.round((iAx/4-x)*4).astype('int')
        if iC==0:
            axSim[x, y] = plt.subplot2grid((4, 4), (x, y))
            axSim[x, y].set_yticklabels([])
        if iAx==10:
            axSim[x, y].hist(
                np.array(simDat[iC][iR[iC]].simDat.iloc[:, [iAx]]).astype('int'),
                density=True, alpha=0.25)
        else:
            axSim[x, y].hist(
                np.array(simDat[iC][iR[iC]].simDat.iloc[:, [iAx]]),
                density=True, alpha=0.66)
        axSim[x, y].set_xlabel(simDat[iC][iR[iC]].simDat.columns[iAx],
                               fontsize='small', labelpad=1)
        axSim[x, y].tick_params(labelsize='small')
        axSim[x, y].annotate(chr(97+iAx)+')', xy=(0.02, 0.95),
                             xytext=(0.02, 0.95),
                            textcoords="axes fraction",
                            xycoords="axes fraction",
                            ha="left", va="top",
)
axSim[2, 2].legend(('Training', 'Outpatient', 'Inpatient', 'General Ward'),
                   fontsize='small')
image_name = "Simulated Populations.png"
figSim.savefig(os.path.join(images_folder, image_name), dpi=fig_Resolution)
plt.close()
print('Simulated populations plotted')


### Sanity Checks: Risk of Metabolic Dysfunction
figDiabetes, axDiabetes = plt.subplots(4, 3, figsize = [8, 8])
figDiabetes.subplots_adjust(top=0.99, left = 0.02, right=0.99, bottom=0.05,
                            hspace=0.25)
for iAx in range(simDat[iC][iR[iC]].diabetes_risk.shape[1]):
    for iC in range(N_cohorts):
        x = np.floor(iAx/3).astype('int')
        y = np.round((iAx/3-x)*3).astype('int')
        if iC==0:
            if iAx<6:
                axDiabetes[x, y] = plt.subplot2grid((4, 3), (x, y))
            else:
                axDiabetes[x, y] = plt.subplot2grid((4, 3), (x, y),
                                                    colspan =3, rowspan=2)
            axDiabetes[x, y].set_yticklabels([])

        axDiabetes[x, y].hist(
            np.array(simDat[iC][iR[iC]].diabetes_risk.iloc[:, [iAx]]),
            density=True, alpha=0.66)
        axDiabetes[x, y].set_xlabel(
            simDat[iC][iR[iC]].diabetes_risk.columns[iAx],
            fontsize='small', labelpad=1)
        axDiabetes[x, y].tick_params(labelsize='small')
        axDiabetes[x, y].annotate(chr(97+iAx)+')', xy=(0.02, 0.95),
                                  xytext=(0.02, 0.95),
                            textcoords="axes fraction",
                            xycoords="axes fraction",
                            ha="left", va="top",
)
axDiabetes[2, 0].legend(('Training', 'Outpatient', 'Inpatient', 'General Ward'),
                        fontsize='small')
axDiabetes[x, y].plot([2.3, 2.3], [0, 1], '--k')
image_name = "Simulated Diabetes Risk.png"
figDiabetes.savefig(os.path.join(images_folder, image_name),
                    dpi=fig_Resolution)        # plt.show()
plt.close()
print('Diabetes plotted')

### Sanity Checks: plot ligament composition
xlabs = pd.concat(
    [simDat[iC][iR[iC]].simDat.loc[:, 'Collagen':'Youngs Modulus Collagen'],
     simDat[iC][iR[iC]].simDat.loc[:, 'Youngs Modulus']], axis=1).columns
figSim, axSim = plt.subplots(np.ceil(len(xlabs)/4).astype('int'), 4,
                              figsize = [8, 8])
figSim.subplots_adjust(top=0.99, left = 0.02, right=0.99, bottom=0.05,
                       hspace=0.25)
for iAx in range(len(xlabs)):
    for iC in range(N_cohorts):
        x = np.floor(iAx/4).astype('int')
        y = np.round((iAx/4-x)*4).astype('int')
        if iC ==0:
            axSim[x, y] = plt.subplot2grid((4, 4), (x, y))
            axSim[x, y].set_yticklabels([])
            axSim[x, y].set_xlabel(xlabs[iAx], fontsize='small', labelpad=1)
            axSim[x, y].tick_params(labelsize='small')
            axSim[x, y].annotate(chr(iAx+97)+')', xy=(0.02, 0.95),
                                 xytext=(0.02, 0.95),
                                textcoords="axes fraction",
                                xycoords="axes fraction",
                                ha="left", va="top",
)
        axSim[x, y].hist(simDat[iC][iR[iC]].simDat.loc[:, xlabs[iAx]],
                         density=True, alpha=0.66)
axSim[1, 3].legend(('Training', 'Outpatient', 'Inpatient', 'General Ward'),
                   fontsize='small')
image_name = "Simulated Ligament Composition.png"
plt.savefig(os.path.join(images_folder, image_name), dpi=fig_Resolution)        # plt.show()
plt.close()
print('Ligament Composition plotted')

### Sanity Checks: Observed Spectra and Model
figSpec, axSpec = plt.subplots(3, 2, figsize = [8, 8])
figSpec.subplots_adjust(top=0.99, left = 0.05, right=0.99, bottom=0.075,
                        hspace=0.05)
axSpec[0, 0].set_yticklabels([])
axSpec[0, 0].set_xticklabels([])
axSpec[0, 0].plot(baseSpectra.x_cal, simSpec[0][:, YM_plsModel.y_train.index],
                  'k', linewidth=0.1, alpha = 0.5)
axSpec[0, 0].tick_params(labelsize='small')
axSpec[0, 0].autoscale(enable=True, tight=True)
axSpec[0, 0].set_ylabel('Raman Intensity')

axSpec[1, 0].set_yticklabels([])
axSpec[1, 0].set_xticklabels([])
axSpec[1, 0].plot(baseSpectra.x_cal, simSpec[0][:, YM_plsModel.y_train.index]-
                  YM_plsModel.meanSpec.reshape(-1, 1), '--k', linewidth=0.1,
                  alpha = 0.5)
axSpec[1, 0].plot(baseSpectra.x_cal, simSpec[0][:, YM_plsModel.y_test.index]-
                  YM_plsModel.meanSpec.reshape(-1, 1), color=[1, 0.3, 0.3],
                  linewidth=0.1, alpha = 0.35)
axSpec[1, 0].tick_params(labelsize='small')
axSpec[1, 0].autoscale(enable=True, tight=True)
axSpec[1, 0].set_ylabel('Raman Deviation')

axSpec[2, 0].set_yticklabels([])
axSpec[2, 0].plot(baseSpectra.x_cal,
                simSpec[0][:, YM_plsModel.y_train.index]-
                YM_plsModel.meanSpec.reshape(-1, 1),
                 '--k', linewidth=0.1, alpha = 0.5)
axSpec[2, 0].plot(baseSpectra.x_cal,
                 simSpec[1]-YM_plsModel.meanSpec.reshape(-1, 1),
                 color=[1, 0.3, 0.3],
                 linewidth=0.3, alpha = 0.35)
axSpec[2, 0].tick_params(labelsize='small')
axSpec[2, 0].autoscale(enable=True, tight=True)
axSpec[2, 0].set_ylabel('Raman Deviation')
axSpec[2, 0].set_xlabel('Wavenumber/cm$^{-1}$')

colors = [[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]]
for iC in range(N_cohorts):
    axSpec[0, 1].plot(baseSpectra.x_cal,
                      np.std(simSpec[iC]-YM_plsModel.meanSpec.reshape(-1, 1),
                             axis=1), linewidth=1/(iC/2+1), color=colors[iC],
                      alpha=0.75)
axSpec[0, 1].annotate('i) SD',
         xy=(0.2, 0.04),
         xytext=(0.01, 0.6),
         textcoords="axes fraction",
         xycoords="axes fraction",
         fontsize=8,
         color=[0, 0, 0],
         horizontalalignment="left",
         va="top",
)
for iC in range(N_cohorts):
    axSpec[0, 1].plot(baseSpectra.x_cal, np.abs(simSpec[iC]-
                                                YM_plsModel.meanSpec.reshape(-1, 1)).mean(axis=1)-0.1, linewidth=1, color=colors[iC], alpha=0.75)
axSpec[0, 1].annotate('ii) Mean Difference',
         xy=(0.2, 0.04),
         xytext=(0.01, 0.33),
         textcoords="axes fraction",
         xycoords="axes fraction",
         fontsize=8,
         color=[0, 0, 0],
         horizontalalignment="left",
         va="top",
)
axSpec[0, 1].set_yticklabels([])
axSpec[0, 1].set_xticklabels([])
axSpec[0, 1].set_ylabel('Raman Variance')
axSpec[0, 1].autoscale(enable=True, tight=True)
axSpec[0, 1].legend(('Training', 'OutPatient', 'InPatient', 'General Ward'),
                    fontsize='small', labelspacing=0.25, framealpha=0.5)


axSpec[1, 1].set_yticklabels([])
axSpec[1, 1].set_xticklabels([])
axSpec[1, 1].plot(baseSpectra.x_cal,
                simSpec[0][:, YM_plsModel.y_train.index]-
                YM_plsModel.meanSpec.reshape(-1, 1),
                 '--k', linewidth=0.2, alpha = 0.5)
axSpec[1, 1].plot(baseSpectra.x_cal,
                 simSpec[2]-YM_plsModel.meanSpec.reshape(-1, 1),
                 color=[1, 0.3, 0.3],
                 linewidth=0.1, alpha = 0.35)
axSpec[1, 1].tick_params(labelsize='small')
axSpec[1, 1].autoscale(enable=True, tight=True)
axSpec[1, 1].set_ylabel('Raman Deviation')



axSpec[2, 1].set_yticklabels([])
axSpec[2, 1].plot(baseSpectra.x_cal,
                simSpec[0][:, YM_plsModel.y_train.index]-
                YM_plsModel.meanSpec.reshape(-1, 1),
                 '--k', linewidth=0.2, alpha = 0.5)
axSpec[2, 1].plot(baseSpectra.x_cal,
                 simSpec[3]-YM_plsModel.meanSpec.reshape(-1, 1),
                 color=[1, 0.3, 0.3],
                 linewidth=0.1, alpha = 0.35)
axSpec[2, 1].tick_params(labelsize='small')
axSpec[2, 1].autoscale(enable=True, tight=True)
axSpec[2, 1].set_ylabel('Raman Deviation')
axSpec[2, 1].set_xlabel('Wavenumber/cm$^{-1}$')


subplts = [['a)', 'b)'], ['c)', 'd)'], ['e)', 'f)']]
for r in range(3):
    for c in range(2):
       axSpec[r, c].annotate(subplts[r][c],
                xy=(0.2, 0.04),
                xytext=(-0.01, 0.99),
                textcoords="axes fraction",
                xycoords="axes fraction",
                fontsize=10,
                color=[0, 0, 0],
                horizontalalignment="right",
                va="top",
)
image_name = "Simulated Sample Spectra.tif"
plt.savefig(os.path.join(images_folder, image_name), dpi=fig_Resolution)
plt.close()
print('Simulated spectra plotted')

### Sanity Checks: Mean Spectra by Cohort
plt.plot(baseSpectra.x_cal, simSpec[0].std(axis=1))
for iC in range(1, N_cohorts):
    plt.plot(baseSpectra.x_cal, np.abs(
        simSpec[iC].mean(axis=1)-YM_plsModel.meanSpec), alpha = 0.75)
plt.gca().set_yticklabels([])
plt.xlabel('Wavenumber/cm$^{-1}$')
plt.gca().set_ylim([plt.gca().get_ylim()[0], plt.gca().get_ylim()[1]*1.1])
plt.legend(('Outpatient Cal Standard Deviation',
            'Mean Outpatient Val - Model Mean',
            'Mean Inpatient Val - Model Mean',
            'Mean General Ward Val - Model Mean'), fontsize='small', ncol=2)
image_name = "Deviations in mean cohort spectra from training set.png"
plt.savefig(os.path.join(images_folder, image_name), dpi=fig_Resolution)
plt.close()
print('Deviations in cohort spectra plotted')

### Sanity Checks: Key Constituent Spectra and Model loadings
temp = simDat[iC][iR[iC]].simDat.loc[0:3, 'Collagen':].copy(deep=True)
temp.loc[:, 'CEP':] = 0
temp.loc[0, 'Collagen':'Hydration'] = [1, 0, 0, 0, 1]
temp.loc[1, 'Collagen':'Hydration'] = [1, 0, 1, 0, 1]
temp.loc[2, 'Collagen':'Hydration'] = [1, 0, 0, 0, 0]
temp.loc[3, 'Collagen':'Hydration'] = [1, 0, 1, 0, 0]
baseSpectra.specSample(temp,
                       0.01,
                       trends['Mean total Counts'][iC])

figLoad, axLoad = plt.subplots(1, 2, figsize = [8, 8])
figLoad.subplots_adjust(top=0.99, left = 0.1, right=0.99, bottom=0.075,
                        hspace=0.25)
axLoad[0] = plt.subplot2grid((1, 2), (0, 0))
tmp = YM_plsModel.YM_plsModel.x_loadings_*YM_plsModel.YM_plsModel.y_loadings_
shft = np.zeros(YM_plsModel.YM_plsModel.n_components+1)
for iPC in range(0, YM_plsModel.YM_plsModel.n_components):
    if iPC>0:
        shft[iPC] = (np.max(tmp[:, iPC])-np.min(tmp[:, iPC-1]))
        tmp[:, iPC] = tmp[:, iPC]-shft[iPC]
    lnColor = (iPC/YM_plsModel.YM_plsModel.n_components,
               iPC**0.5/YM_plsModel.YM_plsModel.n_components,
               1-iPC/YM_plsModel.YM_plsModel.n_components, 0.75)
    axLoad[0].plot(baseSpectra.x_cal, tmp[:, iPC], color=lnColor)
shft[iPC+1] = (np.max(0.1*YM_plsModel.YM_plsModel.coef_)-np.min(tmp[:, iPC]))
axLoad[0].plot(baseSpectra.x_cal, 0.2*YM_plsModel.YM_plsModel.coef_
               -shft[iPC+1], 'k')
axLoad[0].plot(baseSpectra.x_cal[[0, -1]], -np.vstack([shft, shft]), '--',
               color=[0.5, 0.5, 0.5])
axLoad[0].tick_params(labelsize='small')
plt.ylabel('X*Y Weighting')
plt.xlabel('Wavenumber/cm$^{-1}$')
axLoad[0].set_yticklabels([])
axLoad[0].autoscale(enable=True, tight=True)
axLoad[0].set_ylim([axLoad[0].get_ylim()[0]*1.07, axLoad[0].get_ylim()[1]*1.05])
axLoad[0].legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                  'Coefficients'],
                 fontsize='x-small', ncol=5, loc='lower center')
axLoad[0].annotate('a)',
            xy=(0.2, 0.04),
            xytext=(-0.02, 0.99),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=10,
            color=[0, 0, 0],
            horizontalalignment="right",
            va="top",
)

axLoad[1] = plt.subplot2grid((1, 2), (0, 1))
axLoad[1].annotate('b)',
            xy=(0.2, 0.04),
            xytext=(-0.02, 0.99),
            textcoords="axes fraction",
            xycoords="axes fraction",
            fontsize=10,
            color=[0, 0, 0],
            horizontalalignment="right",
            va="top",
)
spec4plot = baseSpectra.constituentSpec.iloc[:,
                             [0, 1, 2, 3, 4, 50, 53, 55, 56, 57, 58, 59]]
axLoad[1].plot(baseSpectra.x_cal, spec4plot/np.max(spec4plot, axis=0)-
               np.arange(12)/2)
axLoad[1].plot(baseSpectra.x_cal[[0, -1]], -np.vstack([np.arange(12)/2,
                                                       np.arange(12)/2]),
               '--', color=[0.5, 0.5, 0.5])

axLoad[1].tick_params(labelsize='small')
plt.ylabel('Raman Intensity')
plt.xlabel('Wavenumber/cm$^{-1}$')
axLoad[1].set_yticklabels([])
axLoad[1].autoscale(enable=True, tight=True)
axLoad[1].set_ylim([axLoad[1].get_ylim()[0]*1.12, axLoad[1].get_ylim()[1]])
axLoad[1].legend(itemgetter(0, 1, 2, 3, 4, 50, 53, 55, 56, 57, 58, 59)(
    baseSpectra.constituents),
                 fontsize='x-small', ncol=3, loc='lower center',
                 labelspacing=0.3, columnspacing=0.2)
image_name = "Model Loadings.tif"
plt.savefig(os.path.join(images_folder, image_name), dpi=fig_Resolution)
plt.close()
print('Model loadings plotted')


## plot ligament correlations
xlabs = simDat[iC][iR[iC]].simDat.loc[:, 'Collagen':].columns
figSim, axSim = plt.subplots(np.ceil(len(xlabs)/4).astype('int'), 4,
                             figsize = [8, 8])
figSim.subplots_adjust(top=0.99, left = 0.02, right=0.99, bottom=0.05,
                       hspace=0.25)
for iAx in range(len(xlabs)-1):
    for iC in range(N_cohorts):
        x = np.floor(iAx/4).astype('int')
        y = np.round((iAx/4-x)*4).astype('int')
        if iC==0:
            if iAx == len(xlabs)-1:
                axSim[x, y] = plt.subplot2grid((4, 4), (x, y),
                                               colspan=len(xlabs)-iAx+2)
            else:
                axSim[x, y] = plt.subplot2grid((4, 4), (x, y))
            axSim[x, y].set_yticklabels([])
            axSim[x, y].annotate(chr(iAx+97)+')'+"{:.3f}".format(
                np.corrcoef(simDat[0][0].simDat.loc[:, xlabs[iAx]],
                        simDat[0][0].simDat.loc[:, 'Youngs Modulus'])[0, 1])
                               , xy=(0.02, 0.95), xytext=(0.02, 0.95),
                                textcoords="axes fraction",
                                xycoords="axes fraction",
                                ha="left", va="top",
)
        axSim[x, y].plot(simDat[iC][iR[iC]].simDat.loc[:, xlabs[iAx]],
                         simDat[iC][iR[iC]].simDat.loc[:, xlabs[-1]], '.')
        axSim[x, y].set_xlabel(xlabs[iAx], fontsize='small', labelpad=1)
        axSim[x, y].tick_params(labelsize='small')
image_name = "Composition vs Youngs Modulus.png"
plt.savefig(os.path.join(images_folder, image_name), dpi=fig_Resolution)
plt.close()
print('Composition vs Youngs Modulus plotted')

nVar = len(xlabs)-4
for iC in range(N_cohorts):
    figLigCorr, axLigCorr = plt.subplots(nVar, nVar, figsize = [8, 8])
    figLigCorr.subplots_adjust(top=0.97, left = 0.03, right=0.99, bottom=0.01,
                               hspace=0.05, wspace=0.05)
    for x in range(nVar):
        for y in range(x, nVar):
            axLigCorr[x, y] = plt.subplot2grid((nVar, nVar), (x, y))
            axLigCorr[x, y].set_yticklabels([])
            axLigCorr[x, y].set_xticklabels([])
            currCorr = np.corrcoef(simDat[iC][iR[iC]].simDat.loc[:, xlabs[x]],
                           simDat[iC][iR[iC]].simDat.loc[:, xlabs[y]])[0, 1]
            colr = [0.25-np.abs(currCorr)/4, np.abs(currCorr), 4*np.abs(
                np.abs((0.5-np.abs(currCorr/2))-0.25)-0.25)]
            axLigCorr[x, y].plot(simDat[iC][iR[iC]].simDat.loc[:, xlabs[x]],
                                 simDat[iC][iR[iC]].simDat.loc[:, xlabs[y]],
                                 '.', color=colr)

            if y > x:
                axLigCorr[y, x] = plt.subplot2grid((nVar, nVar), (y, x))
                axLigCorr[y, x].set_yticklabels([])
                axLigCorr[y, x].set_xticklabels([])
                axLigCorr[y, x].plot(simDat[iC][iR[iC]].simDat.loc[:,
                                                               xlabs[y]],
                                simDat[iC][iR[iC]].simDat.loc[:, xlabs[x]],
                                '.', color=colr)

                axLigCorr[x, y].annotate("{:.3f}".format(currCorr),
                                    xy=(0.02, 0.95), xytext=(0.02, 0.97),
                                    textcoords="axes fraction",
                                    xycoords="axes fraction",
                                    ha="left", va="top", fontsize='x-small'
)
            if x==0:
                axLigCorr[0, y].annotate(xlabs[y],
                                xy=(0.02, 0.95), xytext=(0.5, 1.02),
                                textcoords="axes fraction",
                                xycoords="axes fraction",
                                ha="center", va="bottom", fontsize='x-small'
)
                axLigCorr[y, 0].annotate(xlabs[y],
                                xy=(0.02, 0.95), xytext=(-0.02, 0.5),
                                textcoords="axes fraction",
                                xycoords="axes fraction",
                                ha="right", va="center", fontsize='x-small',
                                rotation=90
)
    image_name = "Composition correlations " + cohorts[iC] + ".png"
    plt.savefig(os.path.join(images_folder, image_name), dpi=fig_Resolution)
    plt.close()
print('Composition correlations plotted')
print('Validation Stress Test Complete')

