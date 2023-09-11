"""
@author: arnaultcamille@univ-poitiers.fr
"""

from FunctionsBasics import *
from FunctionsAreaAnalysis import *


def Start(Data,VerificationStartDetection,VerificationPedalStroke):
        
    #Start Detection
    print('Start Analsysis...')
    try :
        # 1 = nb area to detect
        # 200 = who much frame will be analyzed after the user selection to detect start
        FrameInit,FrameEnd = DetectionDepartsSemiAuto(Data['CadenceTrMin'],1,200,VerificationStartDetection=VerificationStartDetection)
        print("- Start detected")
    except : 
        print(Fore.RED + "ERROR : Start could not be detected.")

    #Pedal stroke area detection
    try :
        # 25 = How much frame will be analyzed before and after the user selection to find the pedal stroke area
        IndexZP = FindZonesPedalees(Data["CadenceTrMin"],25,FrameInit,FrameEnd)
        print("- Pedal stroke area detected.")
    except :
        print(Fore.RED + 'ERROR : Pedal stroke area could not be detected.')    
        
    # Pedal stroke detection
    try :
        IndexCP = DetectionCoupPedaleDepart(Data["PuissanceTotale"],IndexZP,VerificationPedalStroke=VerificationPedalStroke)
        print("- Pedal stroke cycle detected.")
    except :
        print(Fore.RED + "ERROR : Pedal stroke cycle could not be detected.")
    # Impulse & work by pedal stroke for start area
    try :
        ImpulsionDCP = [0 for i in range(len(IndexCP)-1)]
        TravailDCP = [0 for i in range(len(IndexCP)-1)]
        for j in range(0,len(ImpulsionDCP)):
            ImpulsionDCP[j]=np.sum(Data["ImpulsionTotale"][IndexCP[j]:IndexCP[j+1]])
            TravailDCP[j]=np.sum(Data["TravailTotal"][IndexCP[j]:IndexCP[j+1]])
        print("- Impulse & work successfully calculated.")
    except :
        print(Fore.RED + "ERROR : Impulse & work could not be calculated.")
    
    # Max retreat Crank Angle at starting gate
    FrameReculMax = FrameInit+np.argmin(Data['PositionManivelleGauche'][FrameInit:FrameInit+100])
    AngleManivelleGaucheReculMax = Data['PositionManivelleGauche'][FrameReculMax]
    AngleTotalRecul = Data['PositionManivelleGauche'][FrameReculMax]-Data['PositionManivelleGauche'][FrameInit]
         
    return FrameInit, FrameEnd, IndexCP, ImpulsionDCP, TravailDCP, AngleManivelleGaucheReculMax, AngleTotalRecul

def EndStartHill(Data,FrameInit,VerificationEndStartHill="No"):
    """
    USES 
    
    * Find instant of the end of the start hill.
    
    PARAMETERS
    
    * Data = All dataframe of the try, containing at least Forces and Cadence (NxM)
    
    * FrameInit = Start frame of the try. (int)
    
    * VerificationEndStartHill = Yes/No to verify the instant that has been detected for the end of start hill. (str)
    
    """
    # Find instant of end mound
    plt.figure()
    plt.plot(Data['ForceTotaleAbsolue'][FrameInit:FrameInit+800])
    plt.plot(Data['CadenceTrMin'][FrameInit:FrameInit+800]*10,'--')
    plt.legend(['Force Totale Absolue (N)','Cadence*10 (tr/min)'])
    plt.suptitle('END MOUND DETECTION')
    plt.title('Clic on the first force peak after Cadence decrease :')
    plt.xlabel('Frame')
    FrameEncaissement = plt.ginput(n=1)
    FrameEncaissement = int(FrameEncaissement[0][0])
    FrameEncaissement = np.argmax(Data['ForceTotaleAbsolue'][FrameEncaissement-10:FrameEncaissement+10])+(FrameEncaissement-10)
    TempsBasButte = (FrameEncaissement-FrameInit)*0.005
    
    #Verification
    if VerificationEndStartHill in ['O','o','OUI','Oui','oui','Y','y','YES','Yes','yes'] :
        plt.plot(FrameEncaissement,Data['ForceTotaleAbsolue'][FrameEncaissement],'x')
        plt.grid()
    else :
        plt.close()
    
    return TempsBasButte

def FirstJump(Data,FrameInit,VerificationFirstJump="No"):
    """
    USES
    
    * Find take-off instant of the first bump.
    
    PARAMETERS 
    
    * Data : All dataframe of the try, containing at least Forces and Cadence (NxM)
    
    * FrameInit :  Start frame of the try. (int)
    
    * VerificationFirstJump : Yes/No to verify the instant that has been detected for the first jump. (str)
    
    """
    
    # Plot
    plt.figure()
    plt.plot(Data['ForceTotaleAbsolue'][FrameInit:FrameInit+800],label='Force Totale Absolue (N)')  
    plt.plot(Data['CadenceTrMin'][FrameInit:FrameInit+800]*10,'--',label='Cadence x10 (tr/min)')  
    plt.legend()
    plt.suptitle('1st BUMP TAKE-OFF DETECTION')
    plt.title("Clic on the 2nd force peak after Cadence decrease & on minimum after this peak :")
    plt.xlabel('Frame')
    
    # User input, allows to define analysis area
    StartTakeOff = plt.ginput(n=1)
    StartTakeOff = int(StartTakeOff[0][0])
    StartTakeOff = np.argmax(Data['ForceTotaleAbsolue'][StartTakeOff-10:StartTakeOff+10])+(StartTakeOff-10)
    plt.plot(StartTakeOff,Data['ForceTotaleAbsolue'][StartTakeOff],'x')
    EndTakeOff = plt.ginput(n=1)
    EndTakeOff = int(EndTakeOff[0][0])
    EndTakeOff = np.argmin(Data['ForceTotaleAbsolue'][EndTakeOff-10:EndTakeOff+10])+(EndTakeOff-10)
    
    
    #Find max Force slope (=dérivée min) => considered as most repeatable take-off instant
    DeriveeForceTotaleAbsolue = [0 for i in range(StartTakeOff,EndTakeOff)]
    a = 1
    for i in range(StartTakeOff+1,EndTakeOff):
        DeriveeForceTotaleAbsolue[a] = (Data['ForceTotaleAbsolue'][i+1]-Data['ForceTotaleAbsolue'][i-1])/(0.005*2)
        a = a+1
    FrameTakeOff = np.argmin(DeriveeForceTotaleAbsolue)+StartTakeOff
    TimeTakeOff = (FrameTakeOff-FrameInit)*0.005
    
    # Speed calculated as speed mean 25 millisecond before and after detected instant
    TakeOffSpeed = np.mean(Data['VitesseTopTour'][FrameTakeOff-5:FrameTakeOff+5])
    StdTakeOffSpeed =  np.std(Data['VitesseTopTour'][FrameTakeOff-5:FrameTakeOff+5])
    
    
    # Verification
    if VerificationFirstJump in ['O','o','OUI','Oui','oui','Y','y','YES','Yes','yes'] :
        plt.plot(EndTakeOff,Data['ForceTotaleAbsolue'][EndTakeOff],'x')
        plt.plot(FrameTakeOff,Data['ForceTotaleAbsolue'][FrameTakeOff],'x')
        plt.title("Results :")
        plt.grid()
    else :
        plt.close()
    
    return TimeTakeOff, TakeOffSpeed, StdTakeOffSpeed

        