import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#for laptop
#os.chdir('C:/Users/nts21/Documents/Citrine_challenge/challenge_data/')
#for linux workstation
os.chdir('/home/nick/Citrine_challenge/challenge_data/')

#load the data
training=pd.read_csv('training_data.csv')

#separate the output vector
output=training['stabilityVec']

#remove the square brackets from the string
output=[m.strip('[') for m in output]
output=[m.strip(']') for m in output]

#split the string on commas
output=[m.split(',') for m in output]

#convert the list of lists into integers instead of strings
temp=list(output)
output=[]
for x in temp:
    output.append(list(map(lambda y: int(float(y)), x)))

#check that the first is always a 1, uncomment to rerun
#temp=[item[0] for item in output]
#print(len(temp))
#print(sum(temp))
#those should be the same

#check that the last is always a 1, uncomment to rerun
#temp=[item[10] for item in output]
#print(len(temp))
#print(sum(temp))
#again, those should be the same

#great, we can get rid of the first and last value since they are always the same
output=[item[1:10] for item in output]

#create a new value of the sum of the stability vector
sum_output=[sum(item) for item in output]

training['stabilityVec_total']=sum_output
training['stabilityVec']=output

for n in range(0,9):
    temp=[]
    [temp.append(x[n]) for x in training['stabilityVec']]
    training['stabilityVec%s' % (n)]=temp

#this command tells me that only the noble gases never react with anything
temp=training['formulaA'][training['stabilityVec_total']==0].value_counts()
for x in temp.index:
    if x in ['Ar', 'Ne', 'He', 'Kr', 'Xe']:
        print('Sum for element %s is ' % (x) + str(np.sum(training['stabilityVec_total'][training['formulaA']==x])))


#add columns that may be of use for the classifier, ratios and differences of different values

training['CovalentRadius_diff']=training['formulaA_elements_CovalentRadius']\
                                -training['formulaB_elements_CovalentRadius']
training['CovalentRadius_ratio']=training['formulaA_elements_CovalentRadius']/\
                                training['formulaB_elements_CovalentRadius']

training['MendeleevNumber_ratio']=training['formulaA_elements_MendeleevNumber']/\
                                  training['formulaB_elements_MendeleevNumber']
training['MendeleevNumber_diff']=training['formulaA_elements_MendeleevNumber']-\
                                 training['formulaB_elements_MendeleevNumber']

training['Electronegativity_diff']=training['formulaA_elements_Electronegativity']-\
                                   training['formulaB_elements_Electronegativity']
#you will have zeros if you don't remove noble gases
#training['Electronegativity_ratio']=training['formulaA_elements_Electronegativity']/\
#                                   training['formulaB_elements_Electronegativity']

training['FirstIonizationEnergy_ratio']=training['formulaA_elements_FirstIonizationEnergy']/\
                                        training['formulaB_elements_FirstIonizationEnergy']
training['FirstIonizationEnergy_diff']=training['formulaA_elements_FirstIonizationEnergy']-\
                                        training['formulaB_elements_FirstIonizationEnergy']

training['AtomicVolume_ratio']=training['formulaA_elements_AtomicVolume']/\
                               training['formulaB_elements_AtomicVolume']

training['BoilingT_ratio']=training['formulaA_elements_BoilingT']/\
                           training['formulaB_elements_BoilingT']

training['BulkModulus_diff']=training['formulaA_elements_BulkModulus']-\
                             training['formulaB_elements_BulkModulus']

training['Column_diff']=training['formulaA_elements_Column']-\
                        training['formulaB_elements_Column']

training['Density_ratio']=training['formulaA_elements_Density']/\
                          training['formulaB_elements_Density']

training['ElectronSurfaceDensityWS_diff']=training['formulaA_elements_ElectronSurfaceDensityWS']-\
                                          training['formulaB_elements_ElectronSurfaceDensityWS']

training['GSenergy_pa_ratio']=training['formulaA_elements_GSenergy_pa']/\
                              training['formulaB_elements_GSenergy_pa']
training['GSenergy_pa_diff']=training['formulaA_elements_CovalentRadius']\
                                -training['formulaB_elements_CovalentRadius']

#I couldn't figure out what this value represents, but from its values these should work
training['GSestBCClatcnt_ratio']=training['formulaA_elements_GSestBCClatcnt']/\
                                 training['formulaB_elements_GSestBCClatcnt']
training['GSestBCClatcnt_diff']=training['formulaA_elements_GSestBCClatcnt']\
                                -training['formulaB_elements_GSestBCClatcnt']

training['GSestFCClatcnt_ratio']=training['formulaA_elements_GSestFCClatcnt']/\
                                 training['formulaB_elements_GSestFCClatcnt']
training['GSestFCClatcnt_diff']=training['formulaA_elements_GSestFCClatcnt']\
                                -training['formulaB_elements_GSestFCClatcnt']

training['GSmagmom_diff']=training['formulaA_elements_GSmagmom']\
                          -training['formulaB_elements_GSmagmom']

training['GSvolume_pa_diff']=training['formulaA_elements_GSvolume_pa']\
                             -training['formulaB_elements_GSvolume_pa']
training['GSvolume_pa_ratio']=training['formulaA_elements_GSvolume_pa']/\
                              training['formulaB_elements_GSvolume_pa']

training['HHIr_diff']=training['formulaA_elements_HHIr']\
                      -training['formulaB_elements_HHIr']
training['HHIp_diff']=training['formulaA_elements_HHIp']\
                      -training['formulaB_elements_HHIp']

#heat capacity data appears incomplete, perhaps best not to use it?

training['ICSDVolume_ratio']=training['formulaA_elements_ICSDVolume']/\
                             training['formulaB_elements_ICSDVolume']
training['ICSDVolume_diff']=training['formulaA_elements_ICSDVolume']\
                            -training['formulaB_elements_ICSDVolume']

training['element_number_ratio']=training['formulaA_elements_Number']/\
                                 training['formulaB_elements_Number']

training['element_number_difference']=training['formulaA_elements_Number']-\
                                      training['formulaB_elements_Number']

training['polarizability_ratio']=training['formulaA_elements_Polarizability']/\
                                 training['formulaB_elements_Polarizability']
training['polarizability_difference']=training['formulaA_elements_Polarizability']-\
                                      training['formulaB_elements_Polarizability']

training['row_difference']=training['formulaA_elements_Row']-\
                           training['formulaB_elements_Row']

training['spacegroupnumber_ratio']=training['formulaA_elements_SpaceGroupNumber']/\
                                   training['formulaB_elements_SpaceGroupNumber']

training['spacegroupnumber_difference']=training['formulaA_elements_SpaceGroupNumber']-\
                                        training['formulaB_elements_SpaceGroupNumber']

training['avg_coordination_ratio']=training['avg_coordination_A']/\
                                   training['avg_coordination_B']
training['avg_coordination_difference']=training['avg_coordination_A']-\
                                        training['avg_coordination_B']

training['avg_nn_ratio']=training['avg_nearest_neighbor_distance_A']/\
                         training['avg_nearest_neighbor_distance_B']
training['avg_nn_difference']=training['avg_nearest_neighbor_distance_A']-\
                              training['avg_nearest_neighbor_distance_B']

training['meltingT_difference']=training['formulaA_elements_MeltingT']-\
                                training['formulaB_elements_MeltingT']

training['mradius_difference']=training['formulaA_elements_MiracleRadius']-\
                               training['formulaB_elements_MiracleRadius']



#Now for the 1/0 values for different areas of the periodic table
training['is_both_alkali']=training['formulaA_elements_IsAlkali']*\
                           training['formulaB_elements_IsAlkali']
training['alkali_difference']=abs(training['formulaA_elements_IsAlkali']-\
                                  training['formulaB_elements_IsAlkali'])

training['is_both_dblock']=training['formulaA_elements_IsDBlock']*\
                           training['formulaB_elements_IsDBlock']
training['dblock_difference']=abs(training['formulaA_elements_IsDBlock']-\
                                  training['formulaB_elements_IsDBlock'])

training['is_both_fblock']=training['formulaA_elements_IsFBlock']*\
                           training['formulaB_elements_IsFBlock']
training['fblock_difference']=abs(training['formulaA_elements_IsFBlock']-\
                                  training['formulaB_elements_IsFBlock'])

training['is_both_metal']=training['formulaA_elements_IsMetal']*\
                          training['formulaB_elements_IsMetal']
training['metal_difference']=abs(training['formulaA_elements_IsMetal']-\
                                 training['formulaB_elements_IsMetal'])

training['is_both_metalloid']=training['formulaA_elements_IsMetalloid']*\
                              training['formulaB_elements_IsMetalloid']
training['metalloid_difference']=abs(training['formulaA_elements_IsMetalloid']-\
                                     training['formulaB_elements_IsMetalloid'])

training['is_both_nonmetal']=training['formulaA_elements_IsNonmetal']*\
                             training['formulaB_elements_IsNonmetal']
training['nonmetal_difference']=abs(training['formulaA_elements_IsNonmetal']-\
                                    training['formulaB_elements_IsNonmetal'])

#Now for the valence electrons in different orbitals

#this first one generated inf values from the zeros, though I think this might still
#be a useful metric
#training['num_unfilled_ratio']=training['formulaA_elements_NUnfilled']/\
#                              training['formulaB_elements_NUnfilled']
training['num_unfilled_difference']=training['formulaA_elements_NUnfilled']-\
                                    training['formulaB_elements_NUnfilled']

training['num_valence_ratio']=training['formulaA_elements_NValance']/\
                              training['formulaB_elements_NValance']
training['num_valence_difference']=training['formulaA_elements_NValance']-\
                                   training['formulaB_elements_NValance']

training['num_dvalence_difference']=training['formulaA_elements_NdValence']-\
                                    training['formulaB_elements_NdValence']

training['num_fvalence_difference']=training['formulaA_elements_NfValence']-\
                                    training['formulaB_elements_NfValence']

training['num_pvalence_difference']=training['formulaA_elements_NpValence']-\
                                    training['formulaB_elements_NpValence']

training['num_svalence_difference']=training['formulaA_elements_NsValence']-\
                                    training['formulaB_elements_NsValence']

training['num_sunfilled_difference']=training['formulaA_elements_NsUnfilled']-\
                                     training['formulaB_elements_NsUnfilled']

training['num_punfilled_difference']=training['formulaA_elements_NpUnfilled']-\
                                     training['formulaB_elements_NpUnfilled']

training['num_dunfilled_difference']=training['formulaA_elements_NdUnfilled']-\
                                     training['formulaB_elements_NdUnfilled']

training['num_funfilled_difference']=training['formulaA_elements_NfUnfilled']-\
                                     training['formulaB_elements_NfUnfilled']



#Remove noble gases
training_nonoble=training[~training['formulaA'].isin(['Xe', 'Ne', 'Ar', 'He', 'Kr'])]
training_nonoble=training_nonoble[~training_nonoble['formulaB'].isin(['Xe', 'Ne', 'Ar', 'He', 'Kr'])]

#make a list with only the columns that we want to use
col_names=list(training.columns)

col_names.remove('formulaA')
col_names.remove('formulaB')
col_names.remove('stabilityVec')
col_names.remove('stabilityVec_total')

for i in range(9):
    col_names.remove('stabilityVec%s' % i)

#now normalize the data
for i in col_names:
    training[i]/=max(abs(training[i]))
    training_nonoble[i]/=max(abs(training_nonoble[i]))

#now, we are ready for classification. Other scripts will start after this point.
