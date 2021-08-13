# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 03:50:57 2021

@author: Rene
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score

class modelData():
    def __init__(
        self,
        test_size = 0.25, # proportion of data to hold out for test set
        random_state = 3476, # set random seed for set split
        max_iter = 100, # maximum number of iterations per component
        nFold = 5, #number of CV folds
        ):
        self.test_size = test_size
        self.random_state = random_state
        self.max_iter = max_iter
        self.nFold = nFold
        return

    def fit( self , simObj , simSpec ):
### Demographic information
        #create new object, not link

        #now impute marginal distributions on each variable
        self.X_train, self.X_test, self.y_train, self.y_test, self.p_train, self.p_test = train_test_split( simSpec.T,
                                                        simObj.simDat['Youngs Modulus'],
                                                        simObj.pathology,
                                                        test_size = self.test_size,
                                                        random_state = self.random_state)
        self.meanSpec = np.mean(self.X_train, axis=0)

        self.YM_pls_by_ncomp = [None]*10
        self.YM_pls_r2val_ncomp = [None]*10
        for iComp in range(1, 11):
            self.YM_pls_by_ncomp[iComp-1] = PLSRegression(n_components = iComp,
                                                       scale = False,
                                                       max_iter = self.max_iter)
            self.YM_cv_results = cross_validate(self.YM_pls_by_ncomp[iComp-1],
                                        self.X_train-self.meanSpec, self.y_train,
                                        cv=self.nFold)
            self.YM_pls_r2val_ncomp[iComp-1] = np.mean(self.YM_cv_results['test_score'])
        self.YM_plsModel_nComp = np.where( self.YM_pls_r2val_ncomp ==
                                               np.max(self.YM_pls_r2val_ncomp))[0][0]+1
        self.YM_plsModel = PLSRegression( n_components = self.YM_plsModel_nComp,
                                         scale = False,
                                         max_iter = self.max_iter )

        self.YM_plsModel.fit( self.X_train - self.meanSpec, self.y_train )
        self.YM_pls_R2_cal = self.YM_plsModel.score( self.X_train - self.meanSpec,
                                                    self.y_train)
        self.YM_pls_R2_holdout = self.YM_plsModel.score( self.X_test - self.meanSpec,
                                                        self.y_test)

        self.pathology_pls_by_ncomp = [None]*10
        self.pathology_pls_r2val_ncomp = [None]*10
        for iComp in range(1, 11):
            self.pathology_pls_by_ncomp[iComp-1] = PLSRegression(n_components = iComp,
                                                       scale = False,
                                                       max_iter = self.max_iter)
            self.pathology_cv_results = cross_validate(self.pathology_pls_by_ncomp[iComp-1],
                                        self.X_train-self.meanSpec, self.p_train,
                                        cv=self.nFold)
            self.pathology_pls_r2val_ncomp[iComp-1] = np.mean(self.pathology_cv_results['test_score'])
        self.pathology_plsModel_nComp = np.where( self.pathology_pls_r2val_ncomp ==
                                               np.max(self.pathology_pls_r2val_ncomp))[0][0]+1
        self.pathology_plsModel = PLSRegression( n_components = self.pathology_plsModel_nComp,
                                         scale = False,
                                         max_iter = self.max_iter )

        self.pathology_plsModel.fit( self.X_train - self.meanSpec, self.p_train )
        self.pathology_Pred_cal = self.pathology_plsModel.predict(self.X_train-self.meanSpec)
        self.pathology_pls_R2_cal = self.pathology_plsModel.score( self.X_train - self.meanSpec,
                                                    self.p_train)

        self.AUCROC_cal  = roc_auc_score(self.p_train, self.pathology_Pred_cal)

#        Q1 = # (TN x [TP**2 + TPxFN + (FN**2)/3] + FPx(TP**2)/3])/(Nn x Na**2)
#        Q2 = # (TP x [TN**2 + TNxFP + (FP**2)/3] + FPx(TP**2)/3])/(Nn x Na**2)

#        SE = ( self.AUCROC_cal*(1-self.AUCROC_cal) +
#              (np.sum(self.p_train)-1)*(Q1-self.AUCROC_cal**2)

        C = np.bincount( np.concatenate( [ np.arange(0, 4) , #this forces array to return all possibilities
            ((self.p_train)*2 + (self.pathology_Pred_cal[:, 0]>0.5)).ravel() ]
                                        )
                        ) -1 # now remove the extra value added to force all possibiities
        self.sensitivity_cal = C[3]/C[2:4].sum()
        self.specificity_cal = C[0]/C[:2].sum()

        # Hold out set
        self.pathology_Pred_val = self.pathology_plsModel.predict(self.X_test-self.meanSpec)
        self.pathology_pls_R2_holdout = self.pathology_plsModel.score( self.X_test - self.meanSpec,
                                                        self.p_test)
        self.AUCROC_val  = roc_auc_score(self.p_test, self.pathology_Pred_val)
        C = np.bincount( np.concatenate( [ np.arange(0, 4) , #this forces array to return all possibilities
            ((self.p_test)*2 + (self.pathology_Pred_val[:, 0]>0.5)).ravel() ]
                                        )
                        ) -1 # now remove the extra value added to force all possibiities
        self.sensitivity_val = C[3]/C[2:4].sum()
        self.specificity_val = C[0]/C[:2].sum()

        return

    def project_models( self , simObj , simSpec  ):
        nSam = simSpec.shape[1]  # only model what spectra available
        YM_plsR2_val = self.YM_plsModel.score(simSpec.T-self.meanSpec,
                                  simObj.simDat['Youngs Modulus'].iloc[:nSam])
        pathology_Pred = self.pathology_plsModel.predict(
            simSpec.T-self.meanSpec).reshape(-1)
        pathology_True = simObj.pathology[:nSam]
        AUCROC  = roc_auc_score( pathology_True, pathology_Pred )
        C = np.bincount( np.concatenate( [ np.arange(0, 4) ,  # this forces array to return all possibilities
            ((pathology_True)*2 + (pathology_Pred>0.45)).ravel() ]
                                        )
                        ) -1 # remove the extra value added
        sensitivity = C[3]/C[2:4].sum()
        specificity = C[0]/C[:2].sum()

        return YM_plsR2_val, AUCROC, sensitivity, specificity