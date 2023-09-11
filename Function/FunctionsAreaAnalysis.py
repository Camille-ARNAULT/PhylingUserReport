"""
@author: arnaultcamille@univ-poitiers.fr
"""

from FunctionsBasics import *

def DetectionDepartsSemiAuto(CadenceTrMin,NbZones,SearchInterval,VerificationStartDetection="No"):
    """
    USES
    
    * Start instant detection.
    
    PARAMETERS
    
    * CadenceTrMin : Cadence data. (Nx1)
    
    * SearchInterval : Number of frame of the interval in which end instant will be detected. (int)
    
    
    """
    # Cadence plot
    plt.figure()
    plt.plot(CadenceTrMin,'-')
    plt.suptitle("START DETECTION")
    plt.title("Clic before start step back.")
    
    # User input
    FrameInitUser = plt.ginput(n=1,timeout=30,show_clicks=True)
    
    # Searching exact frame in 10000 frames after user input (or less if end of Cadence data)
    FrameInit = round(FrameInitUser[0][0])
    ValInit = CadenceTrMin[FrameInit]
    it = 0
    if (FrameInit+10000) < len(CadenceTrMin) :
        while ValInit > -2 and it < 10000 :
            FrameInit = FrameInit+1
            ValInit = CadenceTrMin[FrameInit]
            it = it+1
    else  :
        while ValInit > -2 and it < (len(CadenceTrMin)-FrameInit-1) :
            FrameInit = FrameInit+1
            ValInit = CadenceTrMin[FrameInit]
            it = it+1  
    FrameInit=FrameInit-20
    
    #End Detection, defined as instant after Start where mean cadence is in [-0.5;0.5]         
    it = 0
    MeanCad = np.mean(CadenceTrMin[FrameInit:FrameInit+SearchInterval])
    StdCad = np.std(CadenceTrMin[FrameInit:FrameInit+SearchInterval])
    while not (StdCad < 0.5 and StdCad > -0.5) and (MeanCad < 0.5 or MeanCad > -0.5) :
        it = it+1
        MeanCad = np.mean(CadenceTrMin[FrameInit+it:FrameInit+SearchInterval+it])  
        StdCad = np.std(CadenceTrMin[FrameInit+it:FrameInit+SearchInterval+it])  
    FrameEnd = FrameInit+it
        
    # Plot detected instants
    if VerificationStartDetection in ['O','o','OUI','Oui','oui','Y','y','YES','Yes','yes'] :
        plt.plot(FrameInit,CadenceTrMin[FrameInit],'x')
        plt.plot(FrameEnd,CadenceTrMin[FrameEnd],'x')
        plt.title("Results :")
        plt.grid()
    else :
        plt.close()
        
    return FrameInit,FrameEnd 

def FindZonesPedalees(Data,ResearchInterval,FrameInit,FrameEnd):
    """
    USES
    
    * Find start and end of pedaling zone with user input. 
    
    PARAMETERS
    
    * Data : Cadence Data (Nx1)
    
    * ResearchInterval : Number of frame of the research interval to find minimums of cadence around user input. (int)
    
    * FrameInit : Start frame (int)
    
    *FrameEnd : End frame (int)
        
    """
    
    # User input to determine start & end of pedaling area
    plt.figure()
    plt.plot(Data[FrameInit:FrameEnd],'-')
    plt.xlabel('Frame')
    plt.ylabel('Cadence (Tr/min)')
    plt.suptitle("PEDALING AREA DETECTION")
    plt.title('Clic twice on minimum of cadence at the start and the end of Pedaling area.')
    UserInput = plt.ginput(n=2,timeout=30,show_clicks=True);
    
    # Find real minimums of cadence around user input
    SearchInterval = [0 for i in range(len(UserInput))]
    for NumInt in range(0,len(UserInput)):
        IntervalData=Data[round(UserInput[NumInt][0])-ResearchInterval:round(UserInput[NumInt][0])+ResearchInterval]
        SearchInterval[NumInt]=np.argmin(IntervalData)+(round(UserInput[NumInt][0])-ResearchInterval)
    
    plt.close()
    
    return SearchInterval

def DetectionCoupPedaleDepart(PowerData,LimitsPedalingArea,VerificationPedalStroke='No'):
    """
    USES
    
    *  Pedal stroke detection, defined as hollows on Power data.
    
    PARAMETERS
    
    * PowerData : Total Power data. (Nx1)
    
    * LimitsPedalingArea : Start & End frame of pedaling area. (1x2)
    
    * VerificationPedalStroke : Yes/No, to verify pedal stroke detection. (str)
        
    """
    
    # Peaks detection in -Power to find hollows
    PuissancePeaks,_ = find_peaks(-PowerData[LimitsPedalingArea[0]:LimitsPedalingArea[1]],height=(None,1500),prominence=(500,None))
    PuissancePeaks = PuissancePeaks+LimitsPedalingArea[0]
    PuissancePeaks = np.insert(PuissancePeaks,0,LimitsPedalingArea[0])
    IndexPedalStroke = PuissancePeaks
    
    if VerificationPedalStroke in ['O','o','OUI','Oui','oui','Y','y','YES','Yes','yes'] :
        plt.figure()
        plt.suptitle("PEDAL STROKE DETECTION")
        plt.title('Results :')
        plt.plot(PowerData[LimitsPedalingArea[0]:LimitsPedalingArea[1]])
        plt.plot(PuissancePeaks,PowerData[PuissancePeaks],'x')
        plt.xlabel("Frame")
        plt.ylabel("Power (W)")
        plt.grid()

    return IndexPedalStroke