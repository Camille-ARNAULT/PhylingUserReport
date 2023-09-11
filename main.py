"""
@author: camille.arnault@univ-poitiers.fr
"""

#%% INITIALIZATION
import os
os.getcwd()
CurrentPath = os.path.dirname(__file__)
os.chdir(CurrentPath)


#%% TEST

with open('INFOS.txt') as f:
    INFOS = f.readlines()

PiloteName = INFOS[2]
Braquet = float(INFOS[4][0:2])/float(INFOS[4][3:5])
CranksetLength = float(INFOS[6])
WheelCircumference = float(INFOS[8])

del f


#%% DECODAGE DES DONNEES TXT DANS LE FICHIER DATA

from Function.PhylingDecoderExtractor import PhylingDecoder

DataPath = CurrentPath + '\\Data\\'
FileList = os.listdir(DataPath)


# Récupération des noms des fichiers bruts en txt et décodés en csv
TxtFile = []
DecodedFile = []
for file in FileList:
    if file.endswith(".txt"):
        TxtFile.append(file[0:len(file)-4])
    if file.endswith("_Decoded.csv"):
        DecodedFile.append(file[0:len(file)-4])    

# Détection des csv déjà existants pour pas perdre du temps, sinon décodage
for i in TxtFile :
    if i+"_Decoded" in DecodedFile :
        print(i + ' : fichier _Decoded déjà récupéré.')
    else :
        PhylingDecoder(DataPath,RawFileName=i+'_Decoded')
        print(i + ' : décodé.')
        
#%%  EXTRACTION DES DONNEES CLASSIQUES DANS LE FICHIER DATA

from Function.Functions import *

#Récupération des noms des fichiers déjà passés par le code de calcul
CalculatedFile = []
for file in FileList:
    if file.endswith("_CalculatedData.csv"):
        CalculatedFile.append(file[0:len(file)-4]) 
        
# Code de calcul appliqué que si le fichier n'est pas déjà passé par le code de calcul
for i in DecodedFile :
    if i[0:len(i)-8]+"_CalculatedData" in CalculatedFile :
        print(i[0:len(i)-8] + ' : fichier _CalculatedData déjà récupéré.')
    else :
        CalculationForOneStart(DataPath+i+'.csv',
                               CirconferenceRoue=WheelCircumference,Braquet=Braquet,LongueurManivelle=CranksetLength,AngleCadre=6,
                               EspacementAimant=90,
                               VerificationResynchro="N",
                               VerificationCrankPeaks="N",
                               VerificationRevolutionCounterPeaks="N",
                               VerificationRevolutionCounterSpeed="N",
                               VerificationImuOrientation="N")

#%% EXTRACTION DES DONNEES SPECIFIQUEMENT LIEES AU DEPART


#Récupération des noms des fichiers déjà passés par le code de calcul
StartAnalysisFile = []
for file in FileList:
    if file.endswith("_StartAnalysis.csv"):
        StartAnalysisFile.append(file[0:len(file)-4]) 
        
        
# Code de traitement des départs appliqué que si le fichier n'est pas déjà passé par le code de traitement
for i in CalculatedFile:
    if i[0:len(i)-15]+"_StartAnalysis" in StartAnalysisFile :
        print(i[0:len(i)-15] + ' : fichier _StartAnalysis.csv déjà récupéré.')
    else :      
        StartAnalysisResults = StartCalculation(DataPath+i+'.csv',PiedAvant='Droit',VerificationStartDetection = "Oui",VerificationPedalStroke = "Oui")









