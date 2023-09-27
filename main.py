"""
@author: camille.arnault@univ-poitiers.fr
"""

#%% INITIALIZATION
import os
os.getcwd()
CurrentPath = os.path.dirname(__file__)
os.chdir(CurrentPath)


#%% PROGRAM

with open('INFOS.txt') as f:
    INFOS = f.readlines()

PilotName = INFOS[2][0:-1]
BraquetTxt = INFOS[4][0:-1]
Braquet = float(INFOS[4][0:2])/float(INFOS[4][3:5])
CranksetLength = float(INFOS[6])
WheelCircumference = float(INFOS[8])
Piste = INFOS[10][0:-1]

del f

print("\n==================================================")
print("DECODAGE...")
print("==================================================\n")

# ACTUALISATION/RECUPERATION DES FICHIERS EXISTANTS
PathRaw = CurrentPath + '\\Data\\A_Raw\\'
FileList = os.listdir(PathRaw)
FileRaw =[]
for file in FileList:
    if file.endswith(".txt"):
        FileRaw.append(file[0:len(file)-4])
PathDecoded = CurrentPath + '\\Data\\B_Decoded\\'
FileList = os.listdir(PathDecoded)
FileDecoded =[]
for file in FileList:
    if file.endswith("_Decoded.csv"):
        FileDecoded.append(file[0:len(file)-12])

# DECODAGE DES DONNEES NON DECODEES
from Function.PhylingDecoderExtractor import PhylingDecoder

for i in FileRaw :
    if i in FileDecoded :
        print(i + ' : fichier déjà décodé.')
    else :
        PhylingDecoder(InputPath=PathRaw,RawFileName=i,OutputPath=PathDecoded)

print("\n==================================================")
print("EXTRACTION & CALCULS...")
print("==================================================\n")

from Function.Functions import *

# ACTUALISATION/RECUPERATION DES FICHIERS EXISTANTS
FileList = os.listdir(PathDecoded)
FileDecoded =[]
for file in FileList:
    if file.endswith("_Decoded.csv"):
        FileDecoded.append(file[0:len(file)-12])     
PathProcessed = CurrentPath + '\\Data\\C_Processed\\'
FileList = os.listdir(PathProcessed)
FileProcessed =[]
for file in FileList:
    if file.endswith("_Processed.csv"):
        FileProcessed.append(file[0:len(file)-14])

# CALCUL DES DONNEES NON TRAITEES
for i in FileDecoded :
    if i in FileProcessed :
        print(i + ' : Extraction & calcul des données déjà effectués.')
    else :
        CalculationForOneStart(PathDecoded,i,PathProcessed,
                               CirconferenceRoue=WheelCircumference,Braquet=Braquet,LongueurManivelle=CranksetLength,AngleCadre=7,
                               EspacementAimant=90,
                               VerificationResynchro="N",
                               VerificationCrankPeaks="N",
                               VerificationRevolutionCounterPeaks="N",
                               VerificationRevolutionCounterSpeed="O",
                               VerificationImuOrientation="N")
        
print("\n==================================================")
print("ANALYSE SPECIFIQUE DES DEPARTS...")
print("==================================================\n")        

# ACTUALISATION/RECUPERATION DES FICHIERS EXISTANTS   
FileList = os.listdir(PathProcessed)
FileProcessed =[]
for file in FileList:
    if file.endswith("_Processed.csv"):
        FileProcessed.append(file[0:len(file)-14])
PathStartAnalysis = CurrentPath + '\\Data\\D_StartAnalysis\\'
FileList = os.listdir(PathStartAnalysis)
FileStartAnalysis =[]
for file in FileList:
    if file.endswith("_StartAnalysis.xlsx"):
        FileStartAnalysis.append(file[0:len(file)-19])       
del FileList

# ANALYSE DE DEPART DES DONNEES NON ANALYSEES
for i in FileProcessed :
    if i in FileStartAnalysis :
        print(i + " : Analyse de départ déjà effectuée.")
    else :      
        StartAnalysisResults = StartCalculation(PathProcessed,i,PathStartAnalysis,PiedAvant='Droit',VerificationStartDetection = "Non",VerificationPedalStroke = "Oui")

print("\n==================================================")
print("REDACTION DES RAPPORTS D'ETUDE...")
print("==================================================\n")  

# ACTUALISATION/RECUPERATION DES FICHIERS EXISTANTS   
FileList = os.listdir(PathStartAnalysis)
FileStartAnalysis =[]
for file in FileList:
    if file.endswith("_StartAnalysis.xlsx"):
        FileStartAnalysis.append(file[0:len(file)-19])       
del FileList      
PathReport = CurrentPath + '\\Data\\E_Report\\'
FileList = os.listdir(PathReport)
FileReport =[]
FileComparativeReport = []
for file in FileList:
    if file.endswith("_Report.pdf"):
        FileReport.append(file[0:len(file)-11])
    if "ComparativeReport" in file :
        FileComparativeReport.append(file[0:len(file)-4])    
del FileList

# EDITION DU RAPPORT SPECIFIQUE A CHAQUE DEPART
for i in FileStartAnalysis :
    if i in FileReport :
        print(i + " : Rapport d'analyse déjà sorti.")
    else :      
         ReportEdition(PathStartAnalysis,i,PathReport, PilotName, BraquetTxt, Braquet, CranksetLength, WheelCircumference, Piste)

# EDITION DU RAPPORT COMPARATIF

NbReport = len(FileComparativeReport)
ReportEditionComparison(PathStartAnalysis, PathReport, FileStartAnalysis, PilotName, BraquetTxt, CranksetLength, WheelCircumference, Piste,NbReport)

