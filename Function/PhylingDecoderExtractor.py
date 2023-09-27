"""
@author: arnaultcamille@univ-poitiers.fr
"""

import pandas as pd
from decoder import decoder

def PhylingDecoder(InputPath, RawFileName, OutputPath):
    
    """
    USES
    
    * Extract CSV file with all decoded data.
    
    PARAMETERS
    
    * InputPath : path of the Raw File you need to decode. (str)
    
    * RawFileName : name of Phyling raw file with .txt extention. (str)

    * OutputFileName : name of output file with .csv extention (str)

    """
    
    #Decoding raw data
    OutputFileName = RawFileName+'_Decoded.csv'
    extraction = decoder.decode(InputPath+RawFileName+'.txt', verbose=True)
    
    #Trying to load gps data
    try :
        modName1 = "gps"
        data_gps = extraction["modules"][modName1]["data"]
        df_gps = pd.DataFrame(data = {'temps_gps':data_gps["T"],'timestamp_gps':data_gps["gpstime"],
                 'longitude':data_gps["longitude"],'latitude':data_gps["latitude"],
                 'vitesse_gps':data_gps["speed"]})
        #Others: data_gps["altitude"], data_gps["heading"], data_gps["nSat"], data_gps["PDOP"]
        print("GPS : successful data decoding.")
        GPS = 1
    except:
        print("GPS : No data found.")
        GPS = 0
        
    
    #Trying to load imu data
    try :
        modName2 = "imu"
        data_imu = extraction["modules"][modName2]["data"]
        df_imu = pd.DataFrame(data = {'temps_imu':data_imu["T"],'acc_x':data_imu["acc_x"],'acc_y':data_imu["acc_y"],'acc_z':data_imu["acc_z"],
                 'gyro_x':data_imu["gyro_x"],'gyro_y':data_imu["gyro_y"],'gyro_z':data_imu["gyro_z"]})
        print("IMU : successful data decoding.")
        IMU = 1
    except:
        print("IMU : No data found.")
        IMU = 0
    
    
    #Trying to load pedalier data
    try :
        headers = list(extraction['modules'].keys())
        for i in range(0,len(extraction['modules'].keys())):
            if headers[i].find('pedalier')!= (-1):
                modName3 = headers[i]
        data_pedalier = extraction["modules"][modName3]["data"]
        df_pedalier = pd.DataFrame(data = {'temps_pedalier':data_pedalier["T"],'gyro_pedalier':data_pedalier["gyro_z"],
                      'force_d':data_pedalier["force_1"],'force_g':data_pedalier["force_2"],
                      'magneto_pedalier':data_pedalier["mag"]})
        print("PEDALIER : successful data decoding.")
        PEDALIER = 1
    except :
        print("PEDALIER : No data found.")
        PEDALIER = 0
    
    
    #Trying to load Top tour data
    try :
        modName4 = "analog" 
        data_toptour = extraction["modules"][modName4]["data"] 
        df_toptour = pd.DataFrame(data = {'temps_toptour':data_toptour["T"],'magneto_toptour':data_toptour["0"]})
        print("TOP TOUR : successful data decoding.")
        TOPTOUR = 1
    except :
        print("TOP TOUR : No data found.")
        TOPTOUR = 0
        
    
    #Resume of avalaible data
    AvalaibleData = [PEDALIER, IMU, TOPTOUR, GPS]
        
    #if 4/4 vars avalaible :
    if AvalaibleData == [1,1,1,1]:
        all_data = pd.concat([df_pedalier, df_imu, df_toptour, df_gps],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
    #if 3/4 vars availaible :
    elif AvalaibleData==[0,1,1,1]:
        all_data = pd.concat([df_imu, df_toptour, df_gps],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
    elif AvalaibleData==[1,0,1,1]:
        all_data = pd.concat([df_pedalier, df_toptour, df_gps],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
    elif AvalaibleData==[1,1,0,1]:
        all_data = pd.concat([df_pedalier, df_imu, df_gps],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
    elif AvalaibleData==[1,1,1,0]:
        all_data = pd.concat([df_pedalier, df_imu, df_toptour],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
        
    #if 2/4 vars avalaible :    
    elif AvalaibleData==[0,0,1,1]:
        all_data = pd.concat([df_toptour, df_gps],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
    elif AvalaibleData==[0,1,0,1]:
        all_data = pd.concat([df_imu, df_gps],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
    elif AvalaibleData==[0,1,1,0]:
        all_data = pd.concat([df_imu, df_toptour],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
    elif AvalaibleData==[1,0,0,1]:
        all_data = pd.concat([df_pedalier,df_gps],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
    elif AvalaibleData==[1,0,1,0]:
        all_data = pd.concat([df_pedalier, df_toptour],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
    elif AvalaibleData==[1,1,0,0]:
        all_data = pd.concat([df_pedalier, df_imu],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
        
    #if 1/4 vars avalaible :    
    elif AvalaibleData==[0,0,0,1]:
            all_data = pd.concat([df_gps],axis=1,sort=False)
            all_data.to_csv(OutputPath+OutputFileName)
            print("EXTRACTION CSV REUSSIE.")
    elif AvalaibleData==[0,0,1,0]:
        all_data = pd.concat([df_toptour],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
    elif AvalaibleData==[0,1,0,0]:
        all_data = pd.concat([df_imu],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
    elif AvalaibleData==[1,0,0,0]:
        all_data = pd.concat([df_pedalier],axis=1,sort=False)
        all_data.to_csv(OutputPath+OutputFileName)
        print("EXTRACTION CSV REUSSIE.")
        
    #if 0/4 vars avalaible :
    elif AvalaibleData==[0,0,0,0]:
        print("ECHEC DE L'EXTRACTION : Aucune donnée n'a été décodée.")
        
        
        
