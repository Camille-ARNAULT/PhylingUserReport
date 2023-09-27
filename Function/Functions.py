"""
@author: camille.arnault@univ-poitiers.fr
"""

import os
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate,Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
import shutil
from sklearn.linear_model import LinearRegression
from datetime import datetime
import imufusion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
from colorama import Fore, init
init(autoreset=True)

def InterpolationResample(xinit, xfinal, y):
    """
    USES

    * Resample data with specific frequency.


    PARAMETERS

    * xinit : original abscissa of the data to be resampled (Nx1)

    * xfinal : abscissa needed (Nx1)

    * y : original ordinate of the data to be resampled (Nx1)

    """
    InterpolatedData = CubicSpline(xinit, y)
    ResampledData = InterpolatedData(xfinal)
    return ResampledData


def NumberOfNonNans(Data):
    """
    USES

    * Find number of non Nans in data.

    PARAMETERS

    * Data : Data set for which you want to know the number of non Nans. (Nx1)
    """
    Count = 0
    for i in Data:
        if not np.isnan(i):
            Count += 1
    return Count


def FiltrageButterworth(Data, FreqAcq, FreqCut):
    """
    USES

    * Apply Butterworth filter on data.


    PARAMETERS

    * Data : Data you want to filter. (Nx1)

    * FreqAcq : Data frequency. (int)

    * FreqCut : Maximum high frequency. (int)

    """
    w = FreqCut / (FreqAcq / 2)  # Normalize the frequency
    b, a = butter(5, w, 'low')
    DataFiltered = filtfilt(b, a, Data)
    return DataFiltered


def IntegrationTrapeze(Data, FreqAcq):
    """
    USES

    * Apply the trapezoidal method integration.


    PARAMETERS

    * Data : Data you want to integrate. (Nx1)

    * FreqAcq : Data frequency. (int)

    """
    IntegratedData = np.zeros(shape=(len(Data), 1))
    for i in range(1, len(Data)):
        RectangleArea = (min(Data[i], Data[i-1]))*(1/FreqAcq)
        TriangleArea = (abs(Data[i]-Data[i-1])*(1/FreqAcq))/2
        IntegratedData[i] = RectangleArea + TriangleArea
    return IntegratedData


def IndexNearestValue(Array, Value):
    """
    USES

    * Find the index of the value that is closest to a given value.

    PARAMETERS

    * Array : data in which to search Value. (Nx1)

    * Value : Value you want to find. (int)

    """
    Array = np.asarray(Array)
    Index = (np.abs(Array - Value)).argmin()
    return Index


def DetectionFrontMontant(DataToDerive, TimeDataToDerive, LimInit, LimEnd):
    """
    USES

    * Detect rising

    PARAMETERS

    * DataToDerive : Data in which you want detect the rising instant. (Nx1)

    * TimeDataToDerive : Abscissa of DataToDerive. (Nx1)

    * LimInit : Analysis start frame. (int)

    * LimEnd : Analysis end frame. (int)

    """
    # Derivative calculation
    DerivatedData = [0 for i in range(0, len(DataToDerive))]
    for i in range(1, len(DataToDerive)-1):
        DerivatedData[i] = (DataToDerive[i+1]-DataToDerive[i-1]) / \
            (TimeDataToDerive[i+1]-TimeDataToDerive[i-1])
    # Standard deviation calculation
    DerivatedDataSTD = np.std(DerivatedData[LimInit:LimEnd])
    # Find frame at which DerivatedData > 3*std
    RisingFrame = LimEnd
    while abs(DerivatedData[RisingFrame]) < DerivatedDataSTD * 3:
        RisingFrame = RisingFrame + 1
    return RisingFrame, DerivatedDataSTD


def CalculVitesseTopTourArriere(Time, DataMagneto, MagnetSpacingDegrees, WheelCircumference, Sensitivity=6):
    """
    USES 

    * Calculate Speed thanks to revolution counter at the rear wheel.

    PARAMETERS

    * Time : Time data of Revolution counter (Nx1)

    * DataMagneto : Revolution counter Data (Nx1)

    * MagnetSpacingDegrees : Spacement between magnets. (int)

    * WheelCircumference : in mm (int)

    """

    # Centering data around 0 and define approximative up and down min limits to search peaks
    Offset = np.mean(DataMagneto[0:600])
    UpThreshold = (np.max(DataMagneto[0:600]) -
                   np.mean(DataMagneto[0:600]))*Sensitivity
    DownThreshold = (
        np.mean(DataMagneto[0:600])-np.min(DataMagneto[0:600]))*Sensitivity
    DataOffset = DataMagneto - Offset
    # Finding peaks
    FrameInitTemps = Time.index[0]
    try:
        PeaksNeg, _ = find_peaks((DataOffset*(-1)), height=(DownThreshold, None),
                                 prominence=(DownThreshold, None), threshold=(0, None))
        PeaksPos, _ = find_peaks(DataOffset, height=(
            UpThreshold, None), prominence=(UpThreshold, None), threshold=(0, None))
        print("- Magnetic peaks detected.")
    except:
        print(Fore.RED + 'ERROR : Magnetic peaks could not be detected.')
    try:
        # Group all peaks in same var
        A = PeaksNeg
        PeaksTotal = np.sort((np.append(A, PeaksPos)))
        del A
        # Distance calculation between two magnets
        MagnetSpacingWheelRound = MagnetSpacingDegrees/360
        DisplacementWheelRound = (
            WheelCircumference/1000)*MagnetSpacingWheelRound
        # Data initialization
        RevolutionCounterSpeedMeterSecond = [0 for i in range(len(PeaksTotal))]
        RevolutionCounterSpeedKilometerHour = [
            0 for i in range(len(PeaksTotal))]
        DisplacementRevolutionCounterMeter = [
            0 for i in range(len(PeaksTotal))]
        Xpeaks = PeaksTotal + FrameInitTemps
        # Speed calculation
        for i in range(1, len(PeaksTotal)-1):
            DisplacementRevolutionCounterMeter[i] = (
                i+1) * DisplacementWheelRound
            RevolutionCounterSpeedMeterSecond[i] = (
                2*DisplacementWheelRound)/(Time[Xpeaks[i+1]]-Time[Xpeaks[i-1]])
        RevolutionCounterSpeedKilometerHour = [
            i * 3.6 for i in RevolutionCounterSpeedMeterSecond]
        print('- Speed calculation successful.')
    except:
        print(Fore.RED + "ERROR : Speed could not be calculated.")
    return Xpeaks[1:], PeaksTotal[1:], RevolutionCounterSpeedKilometerHour, DisplacementRevolutionCounterMeter, PeaksNeg, PeaksPos


def Resynchro(CranksetTime, VerificationResynchro="No"):
    """
    USES

    * Resynchronize data at each frame thanks to ponctual Phyling resynchronization.

    PARAMETERS

    * CranksetTime : Crankset time data.  (Nx1)

    * VerificationResynchro : Yes/No var to verify graphically the resynchronization. (str)

    """

    diff = np.diff(CranksetTime)
    index_diff = np.where(diff < 0)
    # Nous retourne les index du temps juste avant la resynchro (t1)
    NbResynchro = len(index_diff[0])

    CranksetTimeResynchro = (np.zeros(shape=[len(CranksetTime), 1]))*np.nan
    if NbResynchro > 0:
        for r in range(0, NbResynchro):

            # Calculer 1/F moyen
            if r == 0:
                FrameDepartResynchro = 0

            else:
                FrameDepartResynchro = (index_diff[0][r-1])+1

            Pmoy = np.mean(diff[FrameDepartResynchro:index_diff[0][r]])
            # Calculer l'offset deresynchro
            OffsetTotal = CranksetTime[index_diff[0][r]] - \
                (CranksetTime[index_diff[0][r]+1]-Pmoy)

            # Appliquer l'Offset proportionnellement à la frame depuis la dernière synchro
            for t in range(FrameDepartResynchro, index_diff[0][r]+1):
                Correction = ((CranksetTime[t]-CranksetTime[FrameDepartResynchro])*OffsetTotal)/(
                    CranksetTime[index_diff[0][r]+1]-CranksetTime[FrameDepartResynchro])
                CranksetTimeResynchro[t] = CranksetTime[t]-Correction

            # Rajouter les dernières frames qui ne peuvent pas être resynchro
            if r == NbResynchro-1:
                for t in range((index_diff[0][r])+1, NumberOfNonNans(CranksetTime)):
                    CranksetTimeResynchro[t] = CranksetTime[t]

        # Vérification
        if VerificationResynchro in ['Oui', 'oui', 'OUI', 'o', 'O', 'YES', 'Yes', 'yes', 'Y', 'y']:
            plt.figure()
            plt.plot(CranksetTime)
            plt.plot(CranksetTimeResynchro)
            plt.title('VERIFICATION : Resynchro')
            plt.legend(['Raw crankset time', 'Resynchro crankset time'])
            plt.xlabel('Frame')
            plt.ylabel('Temps (s)')
    elif NbResynchro == 0:
        CranksetTimeResynchro = CranksetTime

    return CranksetTimeResynchro


def ImuOrientation(Time, Data, VerificationImuOrientation='No'):
    """
    USES 

    * Determination of IMU orientation in Eulerian parameters thanks to Fusion algorithm (gyrometer + accelerometer).


    PARAMETERS

    * Time : Time associate to Data. (Nx1)

    * Data : X-Y-Z Gyro & X-Y-Z Accelero data. (Nx6)

    * VerificationImuOrientation : Yes/No var to verify graphically the Orientation. (str) 

    """

    # ----------------------------------------------------------- Calcul Offset Gyro

    try:
        # Offset determination & application
        # GyroX
        FrameFinIntOffsetGyroX, _ = DetectionFrontMontant(
            Data['gyro_x'], Time, 0, 100)
        OffsetGyroX = np.mean(Data['gyro_x'][0:FrameFinIntOffsetGyroX-20])
        GyroX = Data['gyro_x'] - OffsetGyroX
        # GyroY
        FrameFinIntOffsetGyroY, _ = DetectionFrontMontant(
            Data['gyro_y'], Time, 0, 100)
        OffsetGyroY = np.mean(Data['gyro_y'][0:FrameFinIntOffsetGyroY-20])
        GyroY = Data['gyro_y'] - OffsetGyroY
        # GyroZ
        FrameFinIntOffsetGyroZ, _ = DetectionFrontMontant(
            Data['gyro_z'], Time, 0, 100)
        OffsetGyroZ = np.mean(Data['gyro_z'][0:FrameFinIntOffsetGyroZ-20])
        GyroZ = Data['gyro_z'] - OffsetGyroZ

        # Filtering
        GyroX = FiltrageButterworth(GyroX, 200, 20)
        GyroY = FiltrageButterworth(GyroY, 200, 20)
        GyroZ = FiltrageButterworth(GyroZ, 200, 20)

    except:
        print(Fore.RED + "ERROR : Gyroscope offset could not be calculated.")

    # =============================================================================
    # ORIENTATION CALCULATION
    # =============================================================================

    # csv creation for data formatting
    d = {'Time (s)': Time, 'Gyroscope X (deg/s)': GyroX, 'Gyroscope Y (deg/s)': GyroY, 'Gyroscope Z (deg/s)': GyroZ,
         'Accelerometer X (g)': Data['acc_x'], 'Accelerometer Y (g)': Data['acc_y'], 'Accelerometer Z (g)': Data['acc_z']}
    DataIMU = pd.DataFrame(data=d)
    DataIMU.to_csv("TEMPORAIRE_DataForOrientationExtraction.csv")

    # Data extraction thanks to fusion algorithm
    # Import sensor data
    data = np.genfromtxt(
        "TEMPORAIRE_DataForOrientationExtraction.csv", delimiter=",", skip_header=1)
    timestamp = data[:, 1]
    gyroscope = data[:, 2:5]
    accelerometer = data[:, 5:8]

    # Process sensor data
    ahrs = imufusion.Ahrs()
    euler = np.empty((len(timestamp), 3))

    ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU,  # convention
                                       0.5,  # gain
                                       10,  # acceleration rejection
                                       20,  # magnetic rejection
                                       5 * 200)  # rejection timeout = 5 seconds

    for index in range(len(timestamp)):
        ahrs.update_no_magnetometer(
            gyroscope[index], accelerometer[index], 1 / 200)  # 200 Hz sample rate
        euler[index] = ahrs.quaternion.to_euler()

    # Changing orientation to be more comprehensive :
        # X = axe dans le sens d'avancement, rotation positive vers la droite
        # Y = axe medio-latéral, rotation positive vers le haut
        # Z = Axe antéro-postérieur, position vers la droite
    OrientationIMU = euler*[-1, -1, 1]

    # Tempory file supression
    import os
    os.remove("TEMPORAIRE_DataForOrientationExtraction.csv")

    if VerificationImuOrientation in ['OUI', 'Oui', 'oui', 'O', 'o', 'YES', 'Yes', 'yes', 'Y', 'y']:
        plt.figure()
        plt.plot(OrientationIMU[:, 0])
        plt.plot(OrientationIMU[:, 1])
        plt.plot(OrientationIMU[:, 2])
        plt.grid()
        plt.legend(["Autour de x", "Autour de y", "Autour de Z"])
        plt.xlabel("Frame")
        plt.ylabel("Angle (°)")
        plt.title('VERIFICATION : Orientation IMU')

    return OrientationIMU


def CalculationForOneStart(InputPath, FileName, OutputPath, CirconferenceRoue=1591.67, Braquet=44/16, LongueurManivelle=177.5, AngleCadre=6, EspacementAimant=90, SensitivityMagnetoTopTour=7.5, VerificationResynchro="No", VerificationCrankPeaks='No', VerificationRevolutionCounterPeaks='No', VerificationRevolutionCounterSpeed='No', VerificationImuOrientation="No"):

    # Message Starting program
    print("===============")
    print("READING DATA...")
    print("===============")
    # Get decoded data
    Raw = pd.read_csv(InputPath+FileName+'_Decoded.csv')
    print("Reading OK")

    # MiniPhyling-MaxiPhyling resynchronization
    Raw["temps_pedalier"] = Resynchro(
        Raw["temps_pedalier"], VerificationResynchro=VerificationResynchro)
    print("Resynchronisation OK")

    # Extract data for each sensor
    try:
        DataPedalier = Raw[["temps_pedalier", "gyro_pedalier",
                            "force_d", "force_g", "magneto_pedalier"]]
        DataPedalier = DataPedalier[0:NumberOfNonNans(Raw['temps_pedalier'])]
        print('Pedalier data extraction OK')
    except:
        print(Fore.RED + "No pedalier data found.")
    try:
        DataImu = Raw[["temps_imu", "acc_x", "acc_y",
                       "acc_z", "gyro_x", "gyro_y", "gyro_z"]]
        DataImu = DataImu[0:NumberOfNonNans(Raw['temps_imu'])]
        print('IMU data extraction OK')
    except:
        print(Fore.RED + "No IMU data found.")
    try:
        DataTopTour = Raw[["temps_toptour", "magneto_toptour"]]
        print('Top tour data extraction OK')
    except:
        print(Fore.RED + "No top tour data found.")
    try:
        DataGps = Raw[["temps_gps", "timestamp_gps",
                       "longitude", "latitude", "vitesse_gps"]]
        DataGps = DataGps[0:NumberOfNonNans(Raw['temps_gps'])]
        print('GPS data extraction OK')
    except:
        print(Fore.RED + "No GPS data found.")

    print("===============")
    print("CALCULATION...")
    print("===============")

    print("Resample Data...")

    # Specify Frequency we want to use for resample
    FreqAcq = 200

    # Create new time base according to MinPhyling Time (longer than MaxiPhyling Time)
    try:
        # Get time limits
        TimeInit = DataPedalier["temps_pedalier"][0]
        TimeEnd = DataPedalier["temps_pedalier"][len(DataPedalier)-1]
        # Create time axis with controled frequency
        x = np.arange(TimeInit, TimeEnd+(1/FreqAcq), (1/FreqAcq), dtype=float)
        print('- MiniPhyling time based.')
    except:
        print(Fore.RED + "ERROR : Creation time base failed.")

    # Resample Pedalier data according to the new base time with CubicSpline
    try:
        gyro_pedalier_resample = InterpolationResample(
            DataPedalier['temps_pedalier'], x, DataPedalier['gyro_pedalier'])
        force_g_resample = InterpolationResample(
            DataPedalier['temps_pedalier'], x, DataPedalier['force_g'])
        force_d_resample = InterpolationResample(
            DataPedalier['temps_pedalier'], x, DataPedalier['force_d'])
        magneto_resample = InterpolationResample(
            DataPedalier['temps_pedalier'], x, DataPedalier['magneto_pedalier'])  # MAGNETO POUR LES ANCIENS ESSAIS
        print("- Crankset data resample.")
    except:
        print(Fore.RED + "ERROR : Crankset data resample failed.")

    # Resample IMU Data according to the new base time with CubicSpline
    try:
        acc_x_resample = InterpolationResample(
            DataImu['temps_imu'], x, DataImu['acc_x'])
        acc_y_resample = InterpolationResample(
            DataImu['temps_imu'], x, DataImu['acc_y'])
        acc_z_resample = InterpolationResample(
            DataImu['temps_imu'], x, DataImu['acc_z'])
        gyro_x_resample = InterpolationResample(
            DataImu['temps_imu'], x, DataImu['gyro_x'])
        gyro_y_resample = InterpolationResample(
            DataImu['temps_imu'], x, DataImu['gyro_y'])
        gyro_z_resample = InterpolationResample(
            DataImu['temps_imu'], x, DataImu['gyro_z'])
        print("- IMU data resample.")
    except:
        print(Fore.RED + "ERROR : IMU data resample failed.")

    # Resample GPS Data according to the new base time with Linear Regression
    try:
        # Linear Regression Model Creation
        x_model = np.array(DataGps['temps_gps']).reshape((-1, 1))
        y_model = np.array(DataGps['timestamp_gps'])
        model = LinearRegression().fit(x_model, y_model)
        RcarreRegressionTimestamp = model.score(x_model, y_model)
        OrdonneeOrigineTimestamp = model.intercept_
        PenteTimestamp = model.coef_
        # Data prediction
        temps_gps_resample = x
        timestamp_gps_resample = model.predict(
            np.array(temps_gps_resample).reshape(-1, 1))
        dt_object = [0 for i in range(0, len(timestamp_gps_resample))]
        # Convert timestamp type to date & time type
        for timestamp in range(0, len(timestamp_gps_resample)):
            dt_object[timestamp] = datetime.fromtimestamp(
                timestamp_gps_resample[timestamp])
        print("- GPS data resample.")
    except:
        print(Fore.RED + "ERROR : GPS data resample failed.")

    # Data Pedalier & IMU Storage in Dataframe
    locals()['Storage'] = {}
    try:
        locals()['Storage']['RawDataResample'] = {}
        locals()['Storage']['RawDataResample'] = pd.DataFrame(data={'time': x, 'gyro_pedalier': gyro_pedalier_resample, 'force_g': force_g_resample, 'force_d': force_d_resample,
                                                                    'magneto': magneto_resample, 'acc_x': acc_x_resample, 'acc_y': acc_y_resample, 'acc_z': acc_z_resample, 'gyro_x': gyro_x_resample, 'gyro_y': gyro_y_resample, 'gyro_z': gyro_z_resample})
        del gyro_pedalier_resample, force_g_resample, force_d_resample, magneto_resample, acc_x_resample, acc_y_resample, acc_z_resample, gyro_x_resample, gyro_y_resample, gyro_z_resample
        print("- Resampled crankset data stored.")
    except:
        print(Fore.RED + "ERROR : Crankset resampled data could not be stored.")

    # Data GPS Storage in DataFrame
    try:
        locals()['Storage']['DataGPS'] = {}
        locals()['Storage']['DataGPS'] = pd.DataFrame(
            data={'TimestampGPS': timestamp_gps_resample, 'DateTimeGPS': dt_object})
        print("- Resampled GPS data stored.")
    except:
        print(Fore.RED + "ERROR : GPS resampled data could not be stored.")

    print("-----------------------------")
    print("Crankset data calculation...")

    try:
        # Butterworth filter of 20Hz cut off
        ForceGaucheFiltree = FiltrageButterworth(
            locals()['Storage']['RawDataResample']['force_g'], FreqAcq, 20)
        ForceDroiteFiltree = FiltrageButterworth(
            locals()['Storage']['RawDataResample']['force_d'], FreqAcq, 20)
        GyroPedalierFiltre = FiltrageButterworth(
            locals()['Storage']['RawDataResample']['gyro_pedalier'], FreqAcq, 20)
        DataMagnetoPedalierFiltrees = FiltrageButterworth(
            locals()['Storage']['RawDataResample']['magneto'], 200, 50)
        print("- Crankset data filtered.")
    except:
        print(Fore.RED + "ERROR : Crankset data could not be filtered.")

    try:
        # Calculer la cadence, la vitesse et la distance
        CadenceTrMin = GyroPedalierFiltre*(60/360)
        Developpement = (CirconferenceRoue/1000)*Braquet  # m/tr
        VitessePedalier = CadenceTrMin * Developpement * (60/1000)  # km/h
        DistancePedalier = VitessePedalier*((1/FreqAcq)/3600)*1000  # m
        # Calculer la Force, le Couple et la Puissance
        ForceTotale = ForceGaucheFiltree + ForceDroiteFiltree
        ForceTotaleAbsolue = abs(ForceGaucheFiltree) + abs(ForceDroiteFiltree)
        CoupleGauche = ForceGaucheFiltree * (LongueurManivelle/1000)
        CoupleDroite = ForceDroiteFiltree * (LongueurManivelle/1000)
        CoupleTotal = CoupleGauche + CoupleDroite
        PuissanceGauche = CoupleGauche * np.radians(GyroPedalierFiltre)
        PuissanceDroite = CoupleDroite * np.radians(GyroPedalierFiltre)
        PuissanceTotale = PuissanceGauche + PuissanceDroite
        # Calculer l'Impulsion et le Travail
        ImpulsionGauche = IntegrationTrapeze(ForceGaucheFiltree, FreqAcq)
        ImpulsionDroite = IntegrationTrapeze(ForceDroiteFiltree, FreqAcq)
        Impulsion = IntegrationTrapeze(ForceTotale, FreqAcq)
        TravailGauche = IntegrationTrapeze(PuissanceGauche, FreqAcq)
        TravailDroite = IntegrationTrapeze(PuissanceDroite, FreqAcq)
        Travail = IntegrationTrapeze(PuissanceTotale, FreqAcq)
        print("- Forces calculation successful.")
    except:
        print(Fore.RED + "ERROR : Forces could not be calculated.")

    # Angular displacement calculation
    try:
        DeplacementAngulairePedalier = IntegrationTrapeze(
            GyroPedalierFiltre, FreqAcq)
        print("- Crank calculation successful.")
    except:
        print(Fore.RED + "ERROR : Crank displacement could not be calculated.")
    # magneto pedalier peaks detection
    try:
        MagnetoPeaks, _ = find_peaks(DataMagnetoPedalierFiltrees, height=(
            None, None), prominence=(500, None))
        if VerificationCrankPeaks in ['Oui', 'oui', 'OUI', 'o', 'O', 'YES', 'Yes', 'yes', 'Y', 'y']:
            plt.figure()
            plt.plot(MagnetoPeaks,
                     DataMagnetoPedalierFiltrees[MagnetoPeaks], 'x')
            plt.plot(DataMagnetoPedalierFiltrees)
        print("- Crank magnetic peaks detected.")
    except:
        print(Fore.RED + "ERROR : Crank magnetic peaks could not be detected.")

    try:
        # Angular correction
        AngleReel = 270-AngleCadre
        SommeDeplacementAngulairePedalier = np.cumsum(DeplacementAngulairePedalier)
        SommeDeplacementAngulairePedalierCorrige = [0 for i in range(0, len(SommeDeplacementAngulairePedalier))]

        for i in range(0, len(MagnetoPeaks)+1):
            if i == 0:
                offset = AngleReel - SommeDeplacementAngulairePedalier[MagnetoPeaks[i]]
                for j in range(0, MagnetoPeaks[0]+1):
                    SommeDeplacementAngulairePedalierCorrige[j] = SommeDeplacementAngulairePedalier[j]+offset
            if i == len(MagnetoPeaks):
                OffsetContinu = (360*(i-1)+AngleReel) -SommeDeplacementAngulairePedalier[MagnetoPeaks[i-1]]
                for j in range(MagnetoPeaks[i-1], len(SommeDeplacementAngulairePedalier)):
                    SommeDeplacementAngulairePedalierCorrige[j] = SommeDeplacementAngulairePedalier[j]+OffsetContinu
            else:
                OffsetContinu = (360*(i-1)+AngleReel) - SommeDeplacementAngulairePedalier[MagnetoPeaks[i-1]]
                for j in range(MagnetoPeaks[i-1], MagnetoPeaks[i]+1):
                    SommeDeplacementAngulairePedalierCorrige[j] = SommeDeplacementAngulairePedalier[j]+OffsetContinu

                OffsetVariable = (360*i+AngleReel)-SommeDeplacementAngulairePedalierCorrige[MagnetoPeaks[i]]
                LongueurApplicationOffset = MagnetoPeaks[i]-MagnetoPeaks[i-1]
                Inc = 0
                for j in range(MagnetoPeaks[i-1], MagnetoPeaks[i]+1):
                    Coef = (OffsetVariable/LongueurApplicationOffset)*Inc
                    SommeDeplacementAngulairePedalierCorrige[j] = SommeDeplacementAngulairePedalierCorrige[j]+Coef
                    Inc = Inc+1

            # Modulo 360
            AngleManivelleGauche = [0 for i in range(
                0, len(SommeDeplacementAngulairePedalierCorrige))]
            AngleManivelleDroite = [0 for i in range(
                0, len(SommeDeplacementAngulairePedalierCorrige))]
            for j in range(0, len(SommeDeplacementAngulairePedalierCorrige)):
                AngleManivelleGauche[j] = SommeDeplacementAngulairePedalierCorrige[j] % 360
                AngleManivelleDroite[j] = (
                    SommeDeplacementAngulairePedalierCorrige[j]-180) % 360
    except:
        # Modulo 360
        AngleManivelleGauche = [0 for i in range(
            0, len(SommeDeplacementAngulairePedalier))]
        AngleManivelleDroite = [0 for i in range(
            0, len(SommeDeplacementAngulairePedalier))]
        for j in range(0, len(SommeDeplacementAngulairePedalier)):
            AngleManivelleGauche[j] = SommeDeplacementAngulairePedalier[j] % 360
            AngleManivelleDroite[j] = (
                SommeDeplacementAngulairePedalier[j]-180) % 360
        print(Fore.RED + "ERROR : Total crank displacement could not be corrected with magneto.")
    # Data Storage in Dataframe
    try:
        locals()['Storage']['DataPedalier'] = {}
        locals()['Storage']['DataPedalier'] = pd.DataFrame(data={'time': x, 'CadenceTrMin': CadenceTrMin, 'VitessePedalier': VitessePedalier, 'DistancePedalier': DistancePedalier,
                                                                 'PositionManivelleGauche': AngleManivelleGauche, 'PositionManivelleDroite': AngleManivelleDroite,
                                                                 'ForceGauche': ForceGaucheFiltree, 'ForceDroite': ForceDroiteFiltree, 'ForceTotale': ForceTotale, 'ForceTotaleAbsolue': ForceTotaleAbsolue,
                                                                 'CoupleGauche': CoupleGauche, 'CoupleDroite': CoupleDroite, 'CoupleTotal': CoupleTotal,
                                                                 'PuissanceGauche': PuissanceGauche, 'PuissanceDroite': PuissanceDroite, 'PuissanceTotale': PuissanceTotale,
                                                                 'ImpulsionGauche': ImpulsionGauche.flatten(), 'ImpulsionDroite': ImpulsionDroite.flatten(), 'ImpulsionTotale': Impulsion.flatten(),
                                                                 'TravailGauche': TravailGauche.flatten(), 'TravailDroite': TravailDroite.flatten(), 'TravailTotal': Travail.flatten()})
        print("- Crankset calculated data stored.")
    except:
        print(Fore.RED + "ERROR : Crankset calculated data could not be stored.")

    print("--------------------------------------")
    print("Revolution counter data calculation...")
    try:
        # Filtering
        DataMagnetoFiltrees = FiltrageButterworth(
            DataTopTour['magneto_toptour'], 800, 300)

        # Magnetic peaks detection & Velocity calculation
        XVitesseTopTour, XDistanceTopTour, VitesseTopTour, DistanceTopTourM, PeaksNeg, PeaksPos = CalculVitesseTopTourArriere(
            DataTopTour['temps_toptour'], DataMagnetoFiltrees, EspacementAimant, CirconferenceRoue, Sensitivity=6)
        # Verification of Magnetic peaks detection
        if VerificationRevolutionCounterPeaks in ['Oui', 'oui', 'OUI', 'o', 'O', 'YES', 'Yes', 'yes', 'Y', 'y']:
            plt.figure()
            plt.plot(DataTopTour['magneto_toptour'])
            plt.plot(DataMagnetoFiltrees)
            plt.plot(PeaksPos, DataMagnetoFiltrees[PeaksPos], 'x')
            plt.plot(PeaksNeg, DataMagnetoFiltrees[PeaksNeg], 'x')
            plt.grid()

        # Top Tour Velocity Resample
        # Create time data before first velocity calculated (because of the spacement between magnet, velocity is not calculated at specific frequency as other mesurement could.)
        ttoptour = DataTopTour['temps_toptour'][XVitesseTopTour[0]]
        LongueurIntervalle1 = IndexNearestValue(x, ttoptour)
        if x[LongueurIntervalle1] > ttoptour:
            LongueurIntervalle1 = LongueurIntervalle1 - 1
        IntervalleZeros1 = pd.Series([0 for i in range(0, LongueurIntervalle1)])
        IntervalleTemps1 = pd.Series([x[0]+(1/FreqAcq)*i for i in range(0, LongueurIntervalle1)])
        
        # Create time data after last velocity calculated
        ttoptour = DataTopTour['temps_toptour'][XVitesseTopTour[-1]]
        LongueurIntervalle2 = IndexNearestValue(x, ttoptour)
        if x[LongueurIntervalle2] < ttoptour and (LongueurIntervalle2+1)<len(x):
            LongueurIntervalle2 = LongueurIntervalle2 + 1
        IntervalleZeros2 = pd.Series([0 for i in range(0, len(x)-LongueurIntervalle2)])
        IntervalleTemps2 = pd.Series([x[LongueurIntervalle2]+(1/FreqAcq)*i for i in range(0, len(x)-LongueurIntervalle2+1)])
        IntervalleDistance2 = pd.Series([DistanceTopTourM[len(DistanceTopTourM)-1] for i in range(0, len(x)-LongueurIntervalle2)])
        
        # Intervalle1 + Données TopTour + Intervalle2
        NewTempsTopTour = pd.concat(
            [IntervalleTemps1, DataTopTour['temps_toptour'][XVitesseTopTour], IntervalleTemps2], ignore_index=True)
        NewVitesseTopTour = pd.concat([IntervalleZeros1, pd.Series(
            VitesseTopTour), IntervalleZeros2], ignore_index=True)
        NewDistanceTopTour = pd.concat([IntervalleZeros1, pd.Series(
            DistanceTopTourM), IntervalleDistance2], ignore_index=True)
    except:
        print('ERROR : Revolution counter resize prep failed.')
    # Linear interpolation between each point
    try:
        vitesse_toptour_resample = np.interp(
            x, NewTempsTopTour, NewVitesseTopTour)
        distance_toptour_resample = np.interp(
            x, NewTempsTopTour, NewDistanceTopTour)
        print("- Revolution counter data resampled.")
    except:
        print(Fore.RED + "ERROR : Revolution counter data could not be resample.")

    try:
        # Filtring
        vitesse_toptour_resample = FiltrageButterworth(
            vitesse_toptour_resample, FreqAcq, 20)
        # Comparison between TopTour & Pedalier Velocity
        if VerificationRevolutionCounterSpeed in ['Oui', 'oui', 'OUI', 'o', 'O', 'YES', 'Yes', 'yes', 'Y', 'y']:
            plt.figure()
            plt.title('Comparaison vitesse Top Tour/Pédalier')
            plt.plot(x, VitessePedalier)
            plt.plot(NewTempsTopTour, NewVitesseTopTour)
            plt.plot(x, vitesse_toptour_resample)
            plt.grid()
            plt.legend(['VitessePedalier', 'Vitesse Top Tour Initiale',
                       'Vitesse Top Tour Interpolée'])
            
    except:
        print('ERROR : Revolution counter resample prep failed.')
    #DataStorage in DataFrame
    try:
        locals()['Storage']['DataTopTour'] = {}
        locals()['Storage']['DataTopTour'] = pd.DataFrame(data={
            'VitesseTopTour': vitesse_toptour_resample, 'DistanceTopTour': distance_toptour_resample})
        print("- Revolution counter calculated data stored.")
    except:
        print(Fore.RED + "ERROR : Revolution counter calculated data could not be stored.")

    print("-----------------------")
    print("IMU data calculation...")

    try:
        BmxOrientation = ImuOrientation(locals()['Storage']['DataPedalier']['time'], locals()['Storage']['RawDataResample'].iloc[:, len(locals(
        )['Storage']['RawDataResample'].columns)-6:len(locals()['Storage']['RawDataResample'].columns)], VerificationImuOrientation=VerificationImuOrientation)
        locals()['Storage']['DataIMU'] = {}
        locals()['Storage']['DataIMU'] = pd.DataFrame(data={'GyroX': locals()['Storage']['RawDataResample']['gyro_x'], 'GyroY': locals()['Storage']['RawDataResample']['gyro_y'], 'GyroZ': locals()['Storage']['RawDataResample']['gyro_z'],
                                                            'AccX': locals()['Storage']['RawDataResample']['acc_x'], 'AccY': locals()['Storage']['RawDataResample']['acc_y'], 'AccZ': locals()['Storage']['RawDataResample']['acc_z'],
                                                            'Roulis': BmxOrientation[:, 0], 'Tangage': BmxOrientation[:, 1], 'Lacet': BmxOrientation[:, 2]})
        print('- Orientation calculation successful.')
    except:
        print(Fore.RED + "ERROR : IMU Orientation calculation failed.")

    # ===========================================================================
    # EXPORT DES DONNEES AU FORMAT CSV
    # ===========================================================================

    print("===============")
    print("STORAGE...")
    print("===============")

    try:
        all_data = pd.concat([locals()['Storage']['DataPedalier'], locals()['Storage']['DataTopTour'], locals()[
                             'Storage']['DataGPS'], locals()['Storage']['DataIMU']], axis=1, sort=False)
        all_data.to_csv(OutputPath+FileName + '_Processed.csv', index=False)
        print("EXTRACTION CSV REUSSIE.")
    except:
        try:
            all_data = pd.concat([locals()['Storage']['DataPedalier'], locals()[
                                 'Storage']['DataTopTour'], locals()['Storage']['DataIMU']], axis=1, sort=False)
            all_data.to_csv(OutputPath+FileName +
                            '_Processed.csv', index=False)
            print("EXTRACTION CSV REUSSIE.")
            print("(Missing GPS Data.)")
        except:
            try:
                all_data = pd.concat([locals()['Storage']['DataPedalier'], locals()[
                                     'Storage']['DataGPS'], locals()['Storage']['DataIMU']], axis=1, sort=False)
                all_data.to_csv(OutputPath+FileName +
                                '_Processed.csv', index=False)
                print("EXTRACTION CSV REUSSIE.")
                print("(Missing Revolution counter Data.)")
            except:
                try:
                    all_data = pd.concat([locals()['Storage']['DataTopTour'], locals()[
                                         'Storage']['DataGPS'], locals()['Storage']['DataIMU']], axis=1, sort=False)
                    all_data.to_csv(OutputPath+FileName +
                                    '_Processed.csv', index=False)
                    print("EXTRACTION CSV REUSSIE.")
                    print("(Missing Crankset Data.)")
                except:
                    try:
                        all_data = pd.concat([locals()['Storage']['DataPedalier'], locals()[
                                             'Storage']['DataTopTour'], locals()['Storage']['DataGPS']], axis=1, sort=False)
                        all_data.to_csv(OutputPath+FileName +
                                        '_Processed.csv', index=False)
                        print("EXTRACTION CSV REUSSIE.")
                        print("(Missing IMU Data.)")
                    except:
                        try:
                            all_data = pd.concat(
                                [locals()['Storage']['DataPedalier']], axis=1, sort=False)
                            all_data.to_csv(
                                OutputPath+FileName + '_Processed.csv', index=False)
                            print("EXTRACTION CSV REUSSIE.")
                            print("(Missing GPS & Top Tour Data.)")
                        except:
                            print('Missing 2 or more sensors : csv writing failed. ')


def DetectionDepartsSemiAuto(CadenceTrMin, VerificationStartDetection="No"):
    """
    USES

    * Start instant detection.

    PARAMETERS

    * CadenceTrMin : Cadence data. (Nx1)

    """
    # Cadence plot
    plt.figure()
    plt.plot(CadenceTrMin, '-')
    plt.suptitle("START DETECTION")
    plt.title("Clic before start step back.")

    # User input
    FrameInitUser = plt.ginput(n=1, timeout=30, show_clicks=True)

    # Searching exact frame in 10000 frames after user input (or less if end of Cadence data)
    FrameInit = round(FrameInitUser[0][0])
    ValInit = CadenceTrMin[FrameInit]
    it = 0
    if (FrameInit+10000) < len(CadenceTrMin):
        while ValInit > -2 and it < 10000:
            FrameInit = FrameInit+1
            ValInit = CadenceTrMin[FrameInit]
            it = it+1
    else:
        while ValInit > -2 and it < (len(CadenceTrMin)-FrameInit-1):
            FrameInit = FrameInit+1
            ValInit = CadenceTrMin[FrameInit]
            it = it+1
    FrameInit = FrameInit-20

    # End Detection, defined as instant after Start where mean cadence is in [-0.5;0.5]
    it = 0
    MeanCad = np.mean(CadenceTrMin[FrameInit:FrameInit+200])
    StdCad = np.std(CadenceTrMin[FrameInit:FrameInit+200])
    while not (StdCad < 0.5 and StdCad > -0.5) and (MeanCad < 0.5 or MeanCad > -0.5):
        it = it+1
        MeanCad = np.mean(CadenceTrMin[FrameInit+it:FrameInit+200+it])
        StdCad = np.std(CadenceTrMin[FrameInit+it:FrameInit+200+it])
    FrameEnd = FrameInit+it

    # Plot detected instants
    if VerificationStartDetection in ['O', 'o', 'OUI', 'Oui', 'oui', 'Y', 'y', 'YES', 'Yes', 'yes']:
        plt.plot(FrameInit, CadenceTrMin[FrameInit], 'x')
        plt.plot(FrameEnd, CadenceTrMin[FrameEnd], 'x')
        plt.title("Results :")
        plt.grid()
    else:
        plt.close()

    return FrameInit, FrameEnd


def FindZonesPedalees(Data, FrameInit, FrameEnd):
    """
    USES

    * Find start and end of pedaling zone with user input. 

    PARAMETERS

    * Data : Cadence Data (Nx1)

    * FrameInit : Start frame (int)

    *FrameEnd : End frame (int)

    """

    # User input to determine start & end of pedaling area
    plt.figure()
    plt.plot(Data[FrameInit:FrameEnd], '-')
    plt.xlabel('Frame')
    plt.ylabel('Cadence (Tr/min)')
    plt.suptitle("PEDALING AREA DETECTION")
    plt.title(
        'Clic twice on minimum of cadence at the start and the end of Pedaling area.')
    UserInput = plt.ginput(n=2, timeout=30, show_clicks=True)

    # Find real minimums of cadence around user input
    SearchInterval = [0 for i in range(len(UserInput))]
    for NumInt in range(0, len(UserInput)):
        IntervalData = Data[round(UserInput[NumInt][0]) -
                            25:round(UserInput[NumInt][0])+25]
        SearchInterval[NumInt] = np.argmin(
            IntervalData)+(round(UserInput[NumInt][0])-25)

    plt.close()

    return SearchInterval


def DetectionCoupPedaleDepart(PowerData, FrameInit, LimitsPedalingArea, VerificationPedalStroke='No'):
    """
    USES

    *  Pedal stroke detection, defined as hollows on Power data.

    PARAMETERS

    * PowerData : Total Power data. (Nx1)

    * LimitsPedalingArea : Start & End frame of pedaling area. (1x2)

    * VerificationPedalStroke : Yes/No, to verify pedal stroke detection. (str)

    """

    # Peaks detection in -Power to find hollows
    PuissancePeaks,_ = find_peaks(PowerData[LimitsPedalingArea[0]:LimitsPedalingArea[1]], height=(None, None),prominence=500)
    PuissancePeaks = PuissancePeaks+LimitsPedalingArea[0]
    
    CreuxPuissance = [0 for i in range(0,len(PuissancePeaks)-1)]
    for i in range(0,len(PuissancePeaks)-1):
        CreuxPuissance[i] = np.argmin(PowerData[PuissancePeaks[i]:PuissancePeaks[i+1]])+PuissancePeaks[i]

    IndexPedalStroke = np.insert(CreuxPuissance, 0, LimitsPedalingArea[0])
    IndexPedalStroke = np.insert(IndexPedalStroke, len(IndexPedalStroke), LimitsPedalingArea[1])

    # Correction First CP Index
    inc = FrameInit+50
    a = PowerData[inc]
    while a < 0:
        inc = inc+1
        a = PowerData[inc]
    IndexPedalStroke[0] = inc
    
    # Correction last CP Index
    inc = IndexPedalStroke[-1]-100
    a = PowerData[inc]
    while a > 0:
        inc = inc+1
        a = PowerData[inc]
    IndexPedalStroke[-1] = inc-1

    if VerificationPedalStroke in ['O', 'o', 'OUI', 'Oui', 'oui', 'Y', 'y', 'YES', 'Yes', 'yes']:
        plt.figure()
        plt.suptitle("PEDAL STROKE DETECTION")
        plt.title('Results :')
        plt.plot(PowerData[LimitsPedalingArea[0]:LimitsPedalingArea[1]])
        plt.plot(IndexPedalStroke, PowerData[IndexPedalStroke], 'x')
        plt.plot(PuissancePeaks, PowerData[PuissancePeaks], 'x')
        plt.xlabel("Frame")
        plt.ylabel("Power (W)")
        plt.grid()

    return IndexPedalStroke


def StartCalculation(InputPath, FileName, OutputPath, PiedAvant, VerificationStartDetection='Non', VerificationPedalStroke='Non'):

    # Start Detection
    print('-----------------------')
    print('Start Analsysis...')
    print('-----------------------')

    try:
        Data = pd.read_csv(InputPath+FileName+'_Processed.csv')
    except:
        print(Fore.RED + 'ERROR : Data extraction failed.')

    try:
        FrameInit, FrameEnd = DetectionDepartsSemiAuto(
            Data['CadenceTrMin'], VerificationStartDetection=VerificationStartDetection)
        print("- Start detected")
    except:
        print(Fore.RED + "ERROR : Start could not be detected.")

    # Pedal stroke area detection
    try:
        IndexZP = FindZonesPedalees(Data["CadenceTrMin"], FrameInit, FrameEnd)
        print("- Pedal stroke area detected.")
    except:
        print(Fore.RED + 'ERROR : Pedal stroke area could not be detected.')

    # Pedal stroke detection
    try:
        # Get Pedal Stroke Index
        IndexCP = DetectionCoupPedaleDepart(Data["PuissanceTotale"], FrameInit, IndexZP, VerificationPedalStroke=VerificationPedalStroke)
        print("- Pedal stroke cycle detected.")
        # Get Mean Pedal Stroke Index
        IndexMeanCP = [0 for i in range(len(IndexCP)-1)]
        for i in range(0, len(IndexMeanCP)):
            IndexMeanCP[i] = IndexCP[i]+((IndexCP[i+1]-IndexCP[i])/2)
    except:
        print(Fore.RED + "ERROR : Pedal stroke cycle could not be detected.")

     # =========================
     # CALCULATION FOR FORCES
     # =========================

     # Force moyenne par CP
    try:
        ForceMoyDCP = [0 for i in range(len(IndexCP)-1)]
        for j in range(0, len(ForceMoyDCP)):
            ForceMoyDCP[j] = np.sum(
                Data["ForceTotale"][IndexCP[j]:IndexCP[j+1]])/(IndexCP[j+1]-IndexCP[j])
        print("- Mean Forces by CP successfully calculated.")
    except:
        print(Fore.RED + "ERROR : Mean Forces by CP could not be calculated.")

    # Impulse by pedal stroke for start area
    try:
        ImpulsionDCP = [0 for i in range(len(IndexCP)-1)]
        for j in range(0, len(ImpulsionDCP)):
            ImpulsionDCP[j] = np.sum(
                Data["ImpulsionTotale"][IndexCP[j]:IndexCP[j+1]])
        print("- Impulse by CP successfully calculated.")
    except:
        print(Fore.RED + "ERROR : Impulse could not be calculated.")

    # =========================
    # CALCULATION FOR POWER
    # =========================

    # Puissance moyenne par CP
    try:
        PuissanceMoyDCP = [0 for i in range(len(IndexCP)-1)]
        for j in range(0, len(PuissanceMoyDCP)):
            PuissanceMoyDCP[j] = np.sum(
                Data["PuissanceTotale"][IndexCP[j]:IndexCP[j+1]])/(IndexCP[j+1]-IndexCP[j])
        print("- Mean power by CP successfully calculated.")
    except:
        print(Fore.RED + "ERROR : Mean power by CP could not be calculated.")

    # Travail par CP
    try:
        TravailDCP = [0 for i in range(len(IndexCP)-1)]
        for j in range(0, len(TravailDCP)):
            TravailDCP[j] = np.sum(Data["TravailTotal"]
                                   [IndexCP[j]:IndexCP[j+1]])
        print("- Work by CP successfully calculated.")
    except:
        print(Fore.RED + "ERROR : Work by CP could not be calculated.")

    # Travail cumulé
    TravailCumule = np.cumsum(Data["TravailTotal"][FrameInit:FrameEnd])

    # Travail cumulé à chaque CP
    try:
        TravailCumuleDCP = [0 for i in range(len(IndexCP)-1)]
        for j in range(0, len(TravailDCP)):
            TravailCumuleDCP[j] = np.sum(
                Data["TravailTotal"][IndexCP[0]:IndexCP[j+1]])
        print("- Cumulative work by CP successfully calculated.")
    except:
        print(Fore.RED + "ERROR : Cumulative work by CP could not be calculated.")

    # RMPD70 for each pedal stroke
    try:
        RMPD70Range = [0 for i in range(len(IndexCP)-1)]
        RMPD70Pourcent = [0 for i in range(len(IndexCP)-1)]
        Pmax = [0 for i in range(len(IndexCP)-1)]
        PmaxIndex = [0 for i in range(len(IndexCP)-1)]
        Fmax = [0 for i in range(len(IndexCP)-1)]
        FmaxIndex = [0 for i in range(len(IndexCP)-1)]

        for j in range(0, len(IndexCP)-1):
            # Récupération puissance
            DataPowerCP = Data["PuissanceTotale"][IndexCP[j]:IndexCP[j+1]]
            DataPowerCP = DataPowerCP.reset_index(drop=True)

            # Récupération position manivelle droite
            PosManivelleDroiteCP = Data["PositionManivelleDroite"][IndexCP[j]:IndexCP[j+1]]
            PosManivelleDroiteCP = PosManivelleDroiteCP.reset_index(drop=True)

            # Récupération position manivelle gauche
            PosManivelleGaucheCP = Data["PositionManivelleGauche"][IndexCP[j]:IndexCP[j+1]]
            PosManivelleGaucheCP = PosManivelleGaucheCP.reset_index(drop=True)

            # Création des variables qui stockeront les données >70% Pmax
            DataPuissanceSup70 = []
            DataAngleSup70 = []
            Range = 0

            # Récupération de Pmax
            Pmax[j] = np.max(DataPowerCP)
            PmaxIndex[j] = np.argmax(
                Data["PuissanceTotale"][IndexCP[j]:IndexCP[j+1]])+IndexCP[j]
            Fmax[j] = np.max(Data["ForceTotale"][IndexCP[j]:IndexCP[j+1]])
            FmaxIndex[j] = np.argmax(
                Data["ForceTotale"][IndexCP[j]:IndexCP[j+1]])+IndexCP[j]

            # Affichage des données pour contrôler
            # plt.figure()
            # plt.plot(DataPowerCP)
            # plt.plot(PosManivelleGaucheCP*10,'r')
            # plt.plot(PosManivelleDroiteCP*10,'b')
            # plt.title("coup de pedale "+str(j+1))
            # plt.legend(['Puissance','Angle Gauche','Angle Droit'])
            # plt.grid()

            # Si pied avant droit en premier
            if PiedAvant in ["D", "d", "R", "r", "DROIT", "Droit", "droit", "RIGHT", "Right", "right"]:
                if j % 2 == 0:
                    for i in range(0, len(DataPowerCP)):
                        if PosManivelleDroiteCP[i] > 30 and PosManivelleDroiteCP[i] < 150:
                            Range = Range+1
                            if DataPowerCP[i] > 0.7*Pmax[j]:
                                DataAngleSup70.append(PosManivelleDroiteCP[i])
                                DataPuissanceSup70.append(DataPowerCP[i])
                else:
                    for i in range(0, len(DataPowerCP)):
                        if PosManivelleGaucheCP[i] > 30 and PosManivelleGaucheCP[i] < 150:
                            Range = Range+1
                            if DataPowerCP[i] > 0.7*Pmax[j]:
                                DataAngleSup70.append(PosManivelleGaucheCP[i])
                                DataPuissanceSup70.append(DataPowerCP[i])
            else:
                if j % 2 == 0:
                    for i in range(0, len(DataPowerCP)):
                        if PosManivelleGaucheCP[i] > 30 and PosManivelleGaucheCP[i] < 150:
                            Range = Range+1
                            if DataPowerCP[i] > 0.7*Pmax[j]:
                                DataAngleSup70.append(PosManivelleGaucheCP[i])
                                DataPuissanceSup70.append(DataPowerCP[i])
                else:
                    for i in range(0, len(DataPowerCP)):
                        if PosManivelleDroiteCP[i] > 30 and PosManivelleDroiteCP[i] < 150:
                            Range = Range+1
                            if DataPowerCP[i] > 0.7*Pmax[j]:
                                DataAngleSup70.append(PosManivelleDroiteCP[i])
                                DataPuissanceSup70.append(DataPowerCP[i])
            RMPD70Range[j] = np.max(DataAngleSup70)-np.min(DataAngleSup70)
            RMPD70Pourcent[j] = (len(DataPuissanceSup70)/Range)*100
    except:
        print(Fore.RED + "ERROR : Impulse, work & RMPD70 could not be calculated.")

    # =========================
    # CALCULATION FOR CRANK
    # =========================

    # Max retreat Crank Angle at starting gate
    FrameReculMax = FrameInit + \
        np.argmin(Data['PositionManivelleGauche'][FrameInit:FrameInit+100])
    AngleManivelleGaucheReculMax = Data['PositionManivelleGauche'][FrameReculMax]
    AngleManivelleDroiteReculMax = Data['PositionManivelleDroite'][FrameReculMax]
    AngleManivelleDroiteDepart = Data['PositionManivelleDroite'][FrameInit]
    AngleManivelleGaucheDepart = Data['PositionManivelleGauche'][FrameInit]
    AngleTotalRecul = Data['PositionManivelleGauche'][FrameReculMax] - \
        Data['PositionManivelleGauche'][FrameInit]

    print('-----------------------')
    print("STORAGE...")
    print('-----------------------')

    # IN A DICT

    locals()["StartAnalysisResults"] = {}
    locals()["StartAnalysisResults"]['FrameInit'] = FrameInit
    locals()["StartAnalysisResults"]['FrameEnd'] = FrameEnd
    locals()["StartAnalysisResults"]['IndexCP'] = IndexCP
    locals()["StartAnalysisResults"]['IndexMeanCP'] = np.array(IndexMeanCP)
    locals()["StartAnalysisResults"]['Temps'] = [
        i*0.005 for i in range(0, len(Data['PuissanceTotale'][FrameInit:FrameEnd]))]

    # Données Force
    locals()[
        "StartAnalysisResults"]['Finstant'] = Data['ForceTotale'][FrameInit:FrameEnd]
    locals()[
        "StartAnalysisResults"]['FGauche'] = Data['ForceGauche'][FrameInit:FrameEnd]
    locals()[
        "StartAnalysisResults"]['FDroite'] = Data['ForceDroite'][FrameInit:FrameEnd]
    locals()["StartAnalysisResults"]['ForceMoyDCP'] = ForceMoyDCP
    locals()["StartAnalysisResults"]['Fmax'] = Fmax
    locals()["StartAnalysisResults"]['FmaxIndex'] = np.array(FmaxIndex)

    locals()["StartAnalysisResults"]['ImpulsionDCP'] = ImpulsionDCP

    # Données Puissance
    locals()[
        "StartAnalysisResults"]['Pinstant'] = Data['PuissanceTotale'][FrameInit:FrameEnd]
    locals()["StartAnalysisResults"]['PuissanceMoyDCP'] = PuissanceMoyDCP
    locals()["StartAnalysisResults"]['Pmax'] = Pmax
    locals()["StartAnalysisResults"]['PmaxIndex'] = np.array(PmaxIndex)

    locals()["StartAnalysisResults"]['RMPD70Range'] = RMPD70Range
    locals()["StartAnalysisResults"]['RMPD70Pourcent'] = RMPD70Pourcent

    locals()["StartAnalysisResults"]['TravailDCP'] = TravailDCP
    locals()["StartAnalysisResults"]['TravailCumuleDCP'] = TravailCumuleDCP
    locals()["StartAnalysisResults"]['TravailCumule'] = TravailCumule

    # Données Manivelle
    locals()["StartAnalysisResults"]['AngleManivelleGauche'] = Data['PositionManivelleGauche'][FrameInit:FrameEnd]
    locals()["StartAnalysisResults"]['AngleManivelleDroite'] = Data['PositionManivelleDroite'][FrameInit:FrameEnd]
    locals()["StartAnalysisResults"]['AngleManivelleAvantReculMax'] = np.min(
        [AngleManivelleDroiteReculMax, AngleManivelleGaucheReculMax])
    locals()["StartAnalysisResults"]['AngleManivelleAvantDepart'] = np.min(
        [AngleManivelleDroiteDepart, AngleManivelleGaucheDepart])
    locals()["StartAnalysisResults"]['AngleTotalRecul'] = AngleTotalRecul
    locals()[
        "StartAnalysisResults"]['Cadence'] = Data['CadenceTrMin'][FrameInit:FrameEnd]
    locals()[
        "StartAnalysisResults"]['VitessePedalier'] = Data['VitessePedalier'][FrameInit:FrameEnd]
    locals()[
        "StartAnalysisResults"]['VitesseTopTour'] = Data['VitesseTopTour'][FrameInit:FrameEnd]

    # IN A CSV FILE
    try:
        all_data = pd.concat([pd.Series(locals()['StartAnalysisResults']['FrameInit']),
                              pd.Series(
                                  locals()['StartAnalysisResults']['FrameEnd']),
                              pd.Series(
                                  (locals()['StartAnalysisResults']['IndexCP']-FrameInit)*0.005),
                              pd.Series(
                                  (locals()['StartAnalysisResults']['IndexMeanCP']-FrameInit)*0.005),
                              pd.Series(
                                  locals()['StartAnalysisResults']['Temps']),
                              (pd.Series(locals()["StartAnalysisResults"]['FGauche'])).reset_index(
                                  drop=True),
                              (pd.Series(locals()["StartAnalysisResults"]['FDroite'])).reset_index(
                                  drop=True),
                              (pd.Series(locals()["StartAnalysisResults"]['Finstant'])).reset_index(
                                  drop=True),
                              pd.Series(
                                  locals()["StartAnalysisResults"]['ForceMoyDCP']),
                              pd.Series(
                                  locals()["StartAnalysisResults"]['Fmax']),
                              pd.Series(
                                  (locals()["StartAnalysisResults"]['FmaxIndex']-FrameInit)*0.005),
                              pd.Series(
                                  locals()["StartAnalysisResults"]['ImpulsionDCP']),
                              (pd.Series(locals()["StartAnalysisResults"]['Pinstant'])).reset_index(
                                  drop=True),
                              pd.Series(
                                  locals()["StartAnalysisResults"]['PuissanceMoyDCP']),
                              pd.Series(
                                  locals()["StartAnalysisResults"]['Pmax']),
                              pd.Series(
                                  (locals()["StartAnalysisResults"]['PmaxIndex']-FrameInit)*0.005),
                              pd.Series(
                                  locals()["StartAnalysisResults"]['RMPD70Range']),
                              pd.Series(
                                  locals()["StartAnalysisResults"]['RMPD70Pourcent']),
                              pd.Series(
                                  locals()["StartAnalysisResults"]['TravailDCP']),
                              pd.Series(
                                  locals()["StartAnalysisResults"]['TravailCumuleDCP']),
                              (pd.Series(locals()["StartAnalysisResults"]['TravailCumule'])).reset_index(
                                  drop=True),
                              (pd.Series(locals()["StartAnalysisResults"]['AngleManivelleGauche'])).reset_index(
                                  drop=True),
                              (pd.Series(locals()["StartAnalysisResults"]['AngleManivelleDroite'])).reset_index(
                                  drop=True),
                              pd.Series(
                                  locals()["StartAnalysisResults"]['AngleManivelleAvantDepart']),
                              pd.Series(
                                  locals()["StartAnalysisResults"]['AngleManivelleAvantReculMax']),
                              pd.Series(
                                  locals()["StartAnalysisResults"]['AngleTotalRecul']),
                              (pd.Series(locals()["StartAnalysisResults"]['Cadence'])).reset_index(
                                  drop=True),
                              (pd.Series(locals()["StartAnalysisResults"]['VitessePedalier'])).reset_index(drop=True),
                              (pd.Series(locals()["StartAnalysisResults"]['VitesseTopTour'])).reset_index(drop=True)], axis=1, sort=False)
        all_data = all_data.set_axis(['FrameInit', 'FrameEnd', 'IndexCP', 'IndexMeanCP', 'Temps',
                                      'ForceGaucheInstant', 'ForceDroiteInstant', 'ForceInstant',
                                      'ForceMoyDCP', 'ForceMaxDCP', 'ForceMaxDCPIndex', 'ImpulsionDCP',
                                      'PuissanceInstant', 'PuissanceMoyDCP', 'PuissanceMaxDCP', 'PuissanceMaxDCPIndex',
                                      'RMPD70Range', 'RMPD70Pourcent',
                                      'TravailDCP', 'TravailCumuleDCP', 'TravailCumule',
                                      'AngleManivelleGauche', 'AngleManivelleDroite',
                                      'AngleManivelleDepart', 'AngleManivelleReculMax', 'AngleTotalRecul',
                                      'Cadence', 'VitessePedalier', 'VitesseTopTour'], axis=1)

        # all_data.to_csv(OutputPath + FileName + '_StartAnalysis.xlsx',mode='a',index=False)

        shutil.copy(OutputPath+"TemplateStartAnalysis.xlsx",
                    OutputPath+FileName+"_StartAnalysis.xlsx")
        with pd.ExcelWriter(OutputPath + FileName + '_StartAnalysis.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            all_data.to_excel(writer, sheet_name='DONNEES', index=False)

        print("EXTRACTION CSV REUSSIE.")
    except:
        print(Fore.RED + "ERROR : CSV Extraction failed.")

    return locals()["StartAnalysisResults"]


def ReportEdition(InputPath, FileName, OutputPath, PilotName, BraquetTxt, Braquet, CranksetLength, WheelCircumference, Piste):

    # IMPORTATION DES DONNEES
    try:
        Data = pd.read_excel(InputPath+FileName+'_StartAnalysis.xlsx')
    except:
        print(Fore.RED + 'ERROR : Data extraction failed.')

    # CREATION DES FIGURES POUR LE RAPPORT

    # Récupération nombre de Coups de pédale
    NbCP = NumberOfNonNans(Data["IndexMeanCP"])

    # Force instantanée
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    plt.grid()
    plt.xlim([0, 3.5])
    plt.plot(Data["IndexMeanCP"], Data["ForceMoyDCP"],
             linestyle='--', marker='o', color='#A5A5A5')
    plt.plot(Data["Temps"][0:700], Data["ForceInstant"]
             [0:700], color='#70AD47', lw=2)
    for i in range(0, NbCP):
        ax.annotate(str(round(Data["ForceMaxDCP"][i])), xy=(Data["ForceMaxDCPIndex"][i], Data["ForceMaxDCP"][i]), xytext=(
            Data["ForceMaxDCPIndex"][i]-
            0.05, Data["ForceMaxDCP"][i]+50))
    plt.title("FORCE INSTANTANEE TOTALE")
    plt.legend(['F moy', 'F instant'])
    plt.xlabel('Temps (s)')
    plt.ylabel('Force (N)')
    plt.savefig(OutputPath+FileName+"_Force.png")
    plt.close()
    
    # Force gauche/droite
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    plt.grid()
    plt.xlim([0, 3.5])
    plt.plot(Data["Temps"][0:700], Data["ForceGaucheInstant"][0:700], color='#C90D50', lw=2)
    plt.plot(Data["Temps"][0:700], Data["ForceDroiteInstant"][0:700], color='#00B0F0', lw=2)
    plt.legend(['Force gauche', 'Force droite'])
    plt.title("FORCE INSTANTANEE GAUCHE/DROITE")
    plt.xlabel('Temps (s)')
    plt.ylabel('Force (N)')
    plt.savefig(OutputPath+FileName+"_ForceGaucheDroite.png")
    plt.close()

    # Cadence Coups de pédale
    for i in range(1, 5):
        if (i % 2 == 1 and Data["AngleManivelleDroite"][0] < Data["AngleManivelleGauche"][0]) or (i % 2 == 0 and Data["AngleManivelleGauche"][0] < Data["AngleManivelleDroite"][0]) :
            # Manivelle avant droite
            r = Data["ForceDroiteInstant"][int(
                Data["IndexCP"][i-1]*200):int(Data["IndexCP"][i]*200)]
            theta = np.radians(Data["AngleManivelleDroite"][int(
                Data["IndexCP"][i-1]*200):int(Data["IndexCP"][i]*200)])
            rupper = np.ma.masked_where(r < Data['ForceMaxDCP'][i-1]*0.7, r)
            rlower = np.ma.masked_where(r > 0, r)
            rmiddle = np.ma.masked_where((r < 0) | (r > Data['ForceMaxDCP'][i-1]*0.7), r)
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.plot(theta, rlower,theta,rmiddle,theta,rupper)
            plt.suptitle("Coup de pédale "+str(i)+" - Pied avant (droit)")
            ax.set_rlabel_position(180)
            ax.set_ylim(-100,3500)
            ax.set_yticks(np.arange(-500,3500,500))
            plt.savefig(OutputPath+FileName+"_ForceCP" + str(i)+"PiedAvant.png")
            plt.close()
            # Manivelle arrière gauche
            r = Data["ForceGaucheInstant"][int(
                Data["IndexCP"][i-1]*200):int(Data["IndexCP"][i]*200)]
            theta = np.radians(Data["AngleManivelleGauche"][int(
                Data["IndexCP"][i-1]*200):int(Data["IndexCP"][i]*200)])
            rupper = np.ma.masked_where(r < Data['ForceMaxDCP'][i-1]*0.7, r)
            rlower = np.ma.masked_where(r > 0, r)
            rmiddle = np.ma.masked_where((r < 0) | (r > Data['ForceMaxDCP'][i-1]*0.7), r)
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.plot(theta, rlower,theta,rmiddle,theta,rupper)
            plt.suptitle("Coup de pédale "+str(i) +
                         " - Pied arrière (gauche)")
            ax.set_rlabel_position(180)
            ax.set_ylim(-100,3500)
            ax.set_yticks(np.arange(-500,3500,500))
            plt.savefig(OutputPath+FileName+"_ForceCP" +
                        str(i)+"PiedArriere.png")
            plt.close()
        else:
            # Manivelle arrière droite
            r = Data["ForceDroiteInstant"][int(
                Data["IndexCP"][i-1]*200):int(Data["IndexCP"][i]*200)]
            theta = np.radians(Data["AngleManivelleDroite"][int(
                Data["IndexCP"][i-1]*200):int(Data["IndexCP"][i]*200)])
            rupper = np.ma.masked_where(r < Data['ForceMaxDCP'][i-1]*0.7, r)
            rlower = np.ma.masked_where(r > 0, r)
            rmiddle = np.ma.masked_where((r < 0) | (r > Data['ForceMaxDCP'][i-1]*0.7), r)
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.plot(theta, rlower,theta,rmiddle,theta,rupper)
            plt.suptitle("Coup de pédale "+str(i) +
                         " - Pied arriere (droit)")
            ax.set_rlabel_position(180)
            ax.set_ylim(-100,3500)
            ax.set_yticks(np.arange(-500,3500,500))
            plt.savefig(OutputPath+FileName+"_ForceCP" +
                        str(i)+"PiedArriere.png")
            plt.close()
            # Manivelle avant gauche
            r = Data["ForceGaucheInstant"][int(
                Data["IndexCP"][i-1]*200):int(Data["IndexCP"][i]*200)]
            theta = np.radians(Data["AngleManivelleGauche"][int(
                Data["IndexCP"][i-1]*200):int(Data["IndexCP"][i]*200)])
            rupper = np.ma.masked_where(r < Data['ForceMaxDCP'][i-1]*0.7, r)
            rlower = np.ma.masked_where(r > 0, r)
            rmiddle = np.ma.masked_where((r < 0) | (r > Data['ForceMaxDCP'][i-1]*0.7), r)
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.plot(theta, rlower,theta,rmiddle,theta,rupper)
            plt.suptitle("Coup de pédale "+str(i)+" - Pied avant (gauche)")
            ax.set_rlabel_position(180)
            ax.set_ylim(-100,3500)
            ax.set_yticks(np.arange(-500,3500,500))
            plt.savefig(OutputPath+FileName+"_ForceCP" +
                        str(i)+"PiedAvant.png")
            plt.close()
    
    # Puissance
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    plt.grid()
    plt.xlim([0, 3.5])
    plt.plot(Data["IndexMeanCP"], Data["PuissanceMoyDCP"],
             linestyle='--', marker='o', color='#A5A5A5')
    plt.plot(Data["Temps"][0:700], Data["PuissanceInstant"]
             [0:700], color='#7030A0', lw=2)
    for i in range(0, NbCP):
        ax.annotate(str(round(Data["PuissanceMaxDCP"][i])), xy=(Data["PuissanceMaxDCPIndex"][i], Data["PuissanceMaxDCP"][i]), xytext=(
            Data["PuissanceMaxDCPIndex"][i]-0.05, Data["PuissanceMaxDCP"][i]+50))
    plt.legend(['P moy', 'P instant'])
    plt.title("PUISSANCE")
    plt.xlabel('Temps (s)')
    plt.ylabel('Puissance (W)')
    plt.savefig(OutputPath+FileName+"_Power.png")
    plt.close()

    # % de Puissance > 70% de Pmax
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    plt.grid()
    plt.bar(np.arange(1, NbCP+1),
            Data["RMPD70Pourcent"][0:NbCP], color='#7030A0')
    plt.ylim([0, 100])
    for i in range(0, NbCP):
        ax.annotate(str(round(Data["RMPD70Pourcent"][i]))+'%', xy=(int(
            i+1), Data["RMPD70Pourcent"][i]), xytext=(int(i+1)-0.1, Data["RMPD70Pourcent"][i]+2))
    plt.title("RMPD70")
    plt.xlabel('Coup de pédale')
    plt.ylabel('% de Puissance > 70% de Pmax')
    plt.savefig(OutputPath+FileName+"_RMPD70.png")
    plt.close()
    
    #Graphique Cadence
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    plt.xlim([0, 3.5])
    ax1.plot(Data["Temps"], Data["Cadence"])
    ax1.set_xlabel('Temps (s)')
    mn, mx = ax1.get_ylim()
    ax1.set_ylabel('Cadence (tr/min)')
    CadToSpeed = (WheelCircumference/1000)*Braquet*(60/1000)
    ax2 = ax1.twinx()
    ax2.set_ylim(mn*CadToSpeed, mx*CadToSpeed)
    ax2.set_ylabel('Vitesse (km/h)')
    ax1.grid() 
    plt.savefig(OutputPath+FileName+"_Cadence.png")
    plt.close()
    
    #Graphique Vitesse
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    plt.xlim([0, 3.5])
    ax1.plot(Data["Temps"],Data["VitesseTopTour"])
    ax1.set_xlabel('Temps (s)')
    mn, mx = ax1.get_ylim()
    ax1.set_ylabel('Vitesse (km/h)')
    plt.grid() 
    plt.savefig(OutputPath+FileName+"_Speed.png")
    plt.close()

    # Travail Cumulé
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    plt.grid()
    plt.xlim([0, 3.5])
    for i in range(0, NbCP):
        ax.plot([Data["Temps"][0], Data["Temps"][699]], [Data["TravailCumuleDCP"],
                 Data["TravailCumuleDCP"]], linestyle='--', color='#A5A5A5', linewidth=0.5)
        ax.annotate(str(round(Data["TravailCumuleDCP"][i])), xy=(
            0.05, Data["TravailCumuleDCP"][i]+10), color='#A5A5A5')
    ax.plot(Data["Temps"][0:700], Data["TravailCumule"]
             [0:700], color='#00B0F0')
    plt.title("ENERGIE CUMULEE")
    plt.xlabel('Temps (s)')
    plt.ylabel('Energie (J)')
    plt.savefig(OutputPath+FileName+"_TravailCumule.png")
    plt.close()

    #-------------------------
        # PAGE CONFIGURATION
    #-------------------------

    pdf = Canvas(OutputPath+FileName+"_Report.pdf")
    # TITLE
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(200, 800, "RAPPORT D'ANALYSE")

    # PILOT INFOS
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, 775, "Pilote :")
    pdf.drawString(50, 750, "Braquet :")
    pdf.drawString(50, 725, "Longueur manivelle :")
    pdf.drawString(350, 775, "Piste :")
    pdf.drawString(350, 750, "Jour :")
    pdf.drawString(350, 725, "Heure :")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(90, 775, PilotName)
    pdf.drawString(102, 750, BraquetTxt)
    pdf.drawString(162, 725, str(CranksetLength)+" cm")
    pdf.drawString(385, 775, Piste)
    pdf.drawString(385, 750, FileName[21:23] +
                   "/"+FileName[19:21]+'/'+FileName[15:19])
    pdf.drawString(393, 725, FileName[24:26]+"h"+FileName[26:28])

    # AFFICHAGE DONNEES FORCE
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(
        120, 680, "FORCE (N) EN FONCTION DE L'ANGLE DE LA MANIVELLE")
    # cp1
    pdf.drawInlineImage(OutputPath+FileName +"_ForceCP1PiedArriere.png", 45, 475, 243, 180)
    pdf.drawInlineImage(OutputPath+FileName +"_ForceCP1PiedAvant.png", 275, 475, 243, 180)
    # cp2
    pdf.drawInlineImage(OutputPath+FileName +"_ForceCP2PiedArriere.png", 45, 285, 243, 180)
    pdf.drawInlineImage(OutputPath+FileName +"_ForceCP2PiedAvant.png", 275, 285, 243, 180)
    # cp3
    pdf.drawInlineImage(OutputPath+FileName+"_ForceCP3PiedArriere.png", 45, 95, 243, 180)
    pdf.drawInlineImage(OutputPath+FileName+"_ForceCP3PiedAvant.png", 275, 95, 243, 180)
    # cp4
    pdf.showPage()
    pdf.drawInlineImage(OutputPath+FileName+"_ForceCP4PiedArriere.png", 45, 605, 243, 180)
    pdf.drawInlineImage(OutputPath+FileName+"_ForceCP4PiedAvant.png", 275, 605, 243, 180)
    # Gauche/Droite
    pdf.drawInlineImage(OutputPath+FileName+"_ForceGaucheDroite.png", 0, 320, 600, 300)
    # instant
    pdf.drawInlineImage(OutputPath+FileName+"_Force.png", 0, 25, 600, 300)
    
    # AFFICHAGE DONNEES PUISSANCE
    pdf.showPage()
    pdf.drawInlineImage(OutputPath+FileName+"_Power.png", 0, 450, 600, 300)
    pdf.drawInlineImage(OutputPath+FileName+"_RMPD70.png", 0, 100, 600, 300)
    pdf.showPage()
    pdf.drawInlineImage(OutputPath+FileName +
                        "_TravailCumule.png", 0, 450, 600, 300)
    
    # AFFICHAGE DONNEES VITESSE
    pdf.showPage()
    pdf.drawInlineImage(OutputPath+FileName+"_Cadence.png", 0, 450, 600, 300)
    pdf.drawInlineImage(OutputPath+FileName+"_Speed.png", 0, 100, 600, 300)

    #SUPPRESSION DES FIGURES EXTERNE
    os.remove(OutputPath+FileName+"_Power.png")
    os.remove(OutputPath+FileName+"_RMPD70.png")
    os.remove(OutputPath+FileName+"_ForceGaucheDroite.png")
    os.remove(OutputPath+FileName+"_Force.png")
    os.remove(OutputPath+FileName+"_TravailCumule.png")
    os.remove(OutputPath+FileName +"_ForceCP1PiedArriere.png")
    os.remove(OutputPath+FileName +"_ForceCP2PiedArriere.png")
    os.remove(OutputPath+FileName +"_ForceCP3PiedArriere.png")
    os.remove(OutputPath+FileName +"_ForceCP4PiedArriere.png")
    os.remove(OutputPath+FileName +"_ForceCP1PiedAvant.png")
    os.remove(OutputPath+FileName +"_ForceCP2PiedAvant.png")
    os.remove(OutputPath+FileName +"_ForceCP3PiedAvant.png")
    os.remove(OutputPath+FileName +"_ForceCP4PiedAvant.png")
    os.remove(OutputPath+FileName +"_Cadence.png")
    os.remove(OutputPath+FileName +"_Speed.png")
    # SAVE
    pdf.save()


def ReportEditionComparison(InputPath, OutputPath, FileStartAnalysis, PilotName, BraquetTxt, CranksetLength, WheelCircumference, Piste, NbReport) :

#%%    
    # CREATION DU SET DE COULEURS
    Couleurs = ["#AE0E0E","#ED7D31","#FFC000","#70AD47","#5E8E32","#22D2CE","#1F83F1","#174795","#8A44D8","#F38DFB","#F12F95","#000000","#434343","#979797","#CDCDCD"]

    #BALAYAGE DE TOUTES LES DONNEES STARTANALYSIS
    LegendName=["" for i in range(0,len(FileStartAnalysis))]
    DataForce=pd.DataFrame()
    DataPuissance=pd.DataFrame()
    DataCadence=pd.DataFrame()
    DataEcum=pd.DataFrame()
    DataECP=pd.DataFrame()
    for i in range(0,len(FileStartAnalysis)):
        #Récupération des données
        try:
            FileName = FileStartAnalysis[i]
            Data = pd.read_excel(InputPath+FileName+'_StartAnalysis.xlsx')
            LegendName[i]= FileName[-9:-7] + "-" + FileName[-11:-9] + "-" + FileName[-13:-11] + " - " + FileName[-6:-4] + "h" + FileName[-4:-2]
        except:
            print(Fore.RED + 'ERROR : Data extraction failed.')
        if len(FileStartAnalysis)<6:
            #graphique des forces superposées
            if i == 0 :
                #Création graphique
                FigForce = plt.figure(figsize=(12, 6))
                AxForce = FigForce.add_subplot(111)
                AxForce.grid()
                AxForce.set_title("FORCE INSTANTANEE TOTALE")
                AxForce.set_xlabel('Temps (s)')
                AxForce.set_ylabel('Force (N)')
            #Ajout data
            AxForce.plot(Data["Temps"][0:700],Data["ForceInstant"][0:700],color=Couleurs[i])
            
            
            #graphique des puissances
            if i == 0 :
                #Création graphique
                FigPower = plt.figure(figsize=(12, 6))
                AxPower = FigPower.add_subplot(111)
                AxPower.grid()
                AxPower.set_title("PUISSANCE INSTANTANEE TOTALE")
                AxPower.set_xlabel('Temps (s)')
                AxPower.set_ylabel('Puissance (W)')
            #Ajout data
            AxPower.plot(Data["Temps"][0:700],Data["PuissanceInstant"][0:700],color=Couleurs[i])
            
            #graphique des cadences
            if i == 0:
                #Création graphique
                FigCad = plt.figure(figsize=(12, 6))
                AxCad = FigCad.add_subplot(111)
                AxCad.grid()
                AxCad.set_title("CADENCE")
                AxCad.set_xlabel('Temps (s)')
                AxCad.set_ylabel('Cadence (tr/min)')
            #Ajout data
            AxCad.plot(Data["Temps"][0:700],Data["Cadence"][0:700],color=Couleurs[i])
            
            #graphique energie cumulée
            if i == 0 :
                #Création graphique
                FigEcum = plt.figure(figsize=(12, 6))
                AxEcum = FigEcum.add_subplot(111)
                AxEcum.grid()
                AxEcum.set_title("ENERGIE CUMULEE")
                AxEcum.set_xlabel('Temps (s)')
                AxEcum.set_ylabel('Energie (J)')
            #Ajout data
            AxEcum.plot(Data["Temps"][0:700],Data["TravailCumule"][0:700],color=Couleurs[i])
        
        #Récupération des données de force
        d = {'ForceTotale_Passage'+str(i+1):Data["ForceInstant"][0:700]}
        DataForce=pd.concat([DataForce,pd.DataFrame(data=d)],axis=1)
        #Récupération des données de puissance
        d = {'PuissanceTotale_Passage'+str(i+1):Data["PuissanceInstant"][0:700]}
        DataPuissance=pd.concat([DataPuissance,pd.DataFrame(data=d)],axis=1)
        #Récupération des données de cadence
        d = {'Cadence_Passage'+str(i+1):Data["Cadence"][0:700]}
        DataCadence=pd.concat([DataCadence,pd.DataFrame(data=d)],axis=1)
        #Récupération des données de cadence
        d = {'Ecum_Passage'+str(i+1):Data["TravailCumule"][0:700]}
        DataEcum=pd.concat([DataEcum,pd.DataFrame(data=d)],axis=1)
        d = {'Ecum_Passage'+str(i+1):Data["TravailDCP"][0:9]}
        DataECP=pd.concat([DataECP,pd.DataFrame(data=d)],axis=1)
    
        #Graphique energie culumée exemple pour page 1
        if i == 0 :
            #Création graphique
            FigEcumEx = plt.figure(figsize=(10, 6))
            AxEcumEx = FigEcumEx.add_subplot(111)
            AxEcumEx.set_title("ENERGIE CUMULEE")
            AxEcumEx.set_xlabel('Temps (s)')
            AxEcumEx.set_ylabel('Energie (J)')
            #Ajout data
            AxEcumEx.plot(Data["Temps"][0:700],Data["TravailCumule"][0:700],color=Couleurs[i])
            #Recup limites
            YlimInf,YlimSup = AxEcumEx.get_ylim()
            YLimSupEcum = np.max(Data["TravailCumuleDCP"])
            XLimSupEcum = np.max(Data["IndexCP"])
            #Tracage CP
            Inc = NumberOfNonNans(Data["IndexCP"])
            for j in range(0,Inc):
                plt.plot([Data["IndexCP"][j],Data["IndexCP"][j]],[YlimInf,YlimSup],'-',c="#C5C3C3",linewidth=1)
            #Barrres horizontales
            AxEcumEx.plot([Data["IndexCP"][0]-0.03,Data["IndexCP"][1]],[Data["TravailCumuleDCP"][0],Data["TravailCumuleDCP"][0]],color = "#434343")
            AxEcumEx.plot([Data["IndexCP"][1]-0.03,Data["IndexCP"][2]],[Data["TravailCumuleDCP"][1],Data["TravailCumuleDCP"][1]],color = "#434343")
            AxEcumEx.plot([Data["IndexCP"][0]-0.2,XLimSupEcum],[YLimSupEcum,YLimSupEcum],color = "#434343")
            #Fleches
            AxEcumEx.arrow(Data["IndexCP"][0],Data["TravailCumule"][int(Data["IndexCP"][0]*200)],0,Data["TravailCumule"][int(Data["IndexCP"][1]*200)]-Data["TravailCumule"][int(Data["IndexCP"][0]*200)],head_width=0.03, width = 0.01,color='k',length_includes_head=True,head_length = 100)
            AxEcumEx.arrow(Data["IndexCP"][1],Data["TravailCumule"][int(Data["IndexCP"][1]*200)],0,Data["TravailCumule"][int(Data["IndexCP"][2]*200)]-Data["TravailCumule"][int(Data["IndexCP"][1]*200)],head_width=0.03, width = 0.01,color='k',length_includes_head=True,head_length = 100)
            AxEcumEx.arrow(Data["IndexCP"][0]-0.2,Data["TravailCumule"][int(Data["IndexCP"][0]*200)],0,YLimSupEcum-Data["TravailCumule"][int(Data["IndexCP"][0]*200)],head_width=0.03, width = 0.01,color='k',length_includes_head=True,head_length = 100)
            #Textes
            AxEcumEx.text(Data["IndexCP"][0]-0.1,Data["TravailCumule"][int(Data["IndexCP"][0]*200)]+100,s='Energie CP1',rotation='vertical')
            AxEcumEx.text(Data["IndexCP"][1]-0.1,Data["TravailCumule"][int(Data["IndexCP"][1]*200)]+100,s='Energie CP2',rotation='vertical')
            AxEcumEx.text(Data["IndexCP"][0]-0.3,YLimSupEcum/3,s='Energie finale',rotation='vertical')
            #Remise à l'échelle
            AxEcumEx.set_ylim(YlimInf,YlimSup)
            AxEcumEx.legend(["Energie cumulée","Coup de pédale"])
            
    #Sauvegarde des figures
    if len(FileStartAnalysis)<6:
        AxForce.legend(LegendName)
        FigForce.savefig(OutputPath + "Comparaison_Forces.png")
        plt.close()
        AxPower.legend(LegendName)
        FigPower.savefig(OutputPath + "Comparaison_Puissances.png")
        plt.close()
        AxCad.legend(LegendName)
        FigCad.savefig(OutputPath + "Comparaison_Cadences.png")
        plt.close()
        AxEcum.legend(LegendName)
        FigEcum.savefig(OutputPath + "Comparaison_EnergiesCumulees.png")
        plt.close()
    FigEcumEx.savefig(OutputPath + "ExampleEcum.png")
    plt.close()
    
    #------------------------------------------------
    #CREATION DES FIGURES MOYENNES + ESSAIS ISOLES
    #------------------------------------------------
    
    #Force
    ForceMoy = np.mean(DataForce,axis=1)
    ForceStd = np.std(DataForce,axis=1)
    #Puissance
    PuissanceMoy = np.mean(DataPuissance,axis=1)
    PuissanceStd = np.std(DataPuissance,axis=1)
    #Cadence
    CadenceMoy = np.mean(DataCadence,axis=1)
    CadenceStd = np.std(DataCadence,axis=1)
    #Energie
    EcumMoy = np.mean(DataEcum,axis=1)
    EcumStd = np.std(DataEcum,axis=1)
    
    for i in range(0,len(FileStartAnalysis)):
        #Rechargement des données
        FileName = FileStartAnalysis[i]
        Data = pd.read_excel(InputPath+FileName+'_StartAnalysis.xlsx')
        #Force
        fig1 = plt.figure(figsize=(12, 6))
        AxForce = fig1.add_subplot(111)
        AxForce.yaxis.grid(True)
        AxForce.set_title("FORCE INSTANTANEE TOTALE")
        AxForce.set_xlabel('Temps (s)')
        AxForce.set_ylabel('Force (N)')
        AxForce.plot(Data["Temps"][0:700],ForceMoy,'--k')
        AxForce.fill_between(Data["Temps"][0:700],ForceMoy-ForceStd,ForceMoy+ForceStd,color="#D9D9D9")
        AxForce.plot(Data["Temps"][0:700],DataForce['ForceTotale_Passage'+str(i+1)],color=Couleurs[i])
        YlimInf,YlimSup = AxForce.get_ylim()
        Inc = NumberOfNonNans(Data["IndexCP"])
        for j in range(0,Inc):
            plt.plot([Data["IndexCP"][j],Data["IndexCP"][j]],[YlimInf,YlimSup],'-',c="#C5C3C3",linewidth=1)
        AxForce.set_ylim(YlimInf,YlimSup)        
        fig1.savefig(OutputPath+"Comparative_ForceTotale_"+LegendName[i]+".png")
        plt.close()
        
        #Puissance
        fig2 = plt.figure(figsize=(12, 6))
        AxPower = fig2.add_subplot(111)
        AxPower.yaxis.grid(True)
        AxPower.set_title("PUISSANCE INSTANTANEE TOTALE")
        AxPower.set_xlabel('Temps (s)')
        AxPower.set_ylabel('Puissance (W)')
        AxPower.plot(Data["Temps"][0:700],PuissanceMoy,'--k')
        AxPower.fill_between(Data["Temps"][0:700],PuissanceMoy-PuissanceStd,PuissanceMoy+PuissanceStd,color="#D9D9D9")
        AxPower.plot(Data["Temps"][0:700],DataPuissance['PuissanceTotale_Passage'+str(i+1)],color=Couleurs[i])
        YlimInf,YlimSup = AxPower.get_ylim()
        Inc = NumberOfNonNans(Data["IndexCP"])
        for j in range(0,Inc):
            plt.plot([Data["IndexCP"][j],Data["IndexCP"][j]],[YlimInf,YlimSup],'-',c="#C5C3C3",linewidth=1)
        AxPower.set_ylim(YlimInf,YlimSup)
        fig2.savefig(OutputPath+"Comparative_PuissanceTotale_"+LegendName[i]+".png")
        plt.close()
        
        #Cadence
        fig3 = plt.figure(figsize=(12, 6))
        AxCad = fig3.add_subplot(111)
        AxCad.yaxis.grid(True)
        AxCad.set_title("CADENCE")
        AxCad.set_xlabel('Temps (s)')
        AxCad.set_ylabel('Cadence (tr/min)')
        AxCad.plot(Data["Temps"][0:700],CadenceMoy,'--k')
        AxCad.fill_between(Data["Temps"][0:700],CadenceMoy-CadenceStd,CadenceMoy+CadenceStd,color="#D9D9D9")
        AxCad.plot(Data["Temps"][0:700],DataCadence['Cadence_Passage'+str(i+1)],color=Couleurs[i])
        YlimInf,YlimSup = AxCad.get_ylim()
        Inc = NumberOfNonNans(Data["IndexCP"])
        for j in range(0,Inc):
            plt.plot([Data["IndexCP"][j],Data["IndexCP"][j]],[YlimInf,YlimSup],'-',c="#C5C3C3",linewidth=1)
        AxCad.set_ylim(YlimInf,YlimSup) 
        fig3.savefig(OutputPath+"Comparative_Cadence_"+LegendName[i]+".png")
        plt.close()
        
        #Energie
        fig4 = plt.figure(figsize=(12, 6))
        AxEcum = fig4.add_subplot(111)
        AxEcum.yaxis.grid(True)
        AxEcum.set_title("ENERGIE CUMULEE")
        AxEcum.set_xlabel('Temps (s)')
        AxEcum.set_ylabel('Energie (J)')
        AxEcum.plot(Data["Temps"][0:700],EcumMoy,'--k')
        AxEcum.fill_between(Data["Temps"][0:700],EcumMoy-EcumStd,EcumMoy+EcumStd,color="#D9D9D9")
        AxEcum.plot(Data["Temps"][0:700],DataEcum['Ecum_Passage'+str(i+1)],color=Couleurs[i])
        YlimInf,YlimSup = AxEcum.get_ylim()
        Inc = NumberOfNonNans(Data["IndexCP"])
        for j in range(0,Inc):
            plt.plot([Data["IndexCP"][j],Data["IndexCP"][j]],[YlimInf,YlimSup],'-',c="#C5C3C3",linewidth=1)
        AxEcum.set_ylim(YlimInf,YlimSup)
        fig4.savefig(OutputPath+"Comparative_Ecum_"+LegendName[i]+".png")
        plt.close()
  
    #------------------------------------------------
    #CREATION DU RAPPORT PDF
    #------------------------------------------------
    
    if NbReport != 0 :
        PDFName = "ComparativeReport("+str(NbReport)+").pdf"
    else :
        PDFName = "ComparativeReport.pdf"
        
    pdf = Canvas(OutputPath+PDFName,pagesize=landscape(A4))
    # TITLE
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(210, 550, "RAPPORT D'ANALYSE - COMPARATIF DEPARTS")

    # PILOT INFOS
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, 515, "Pilote :")
    pdf.drawString(50, 490, "Piste :")
    pdf.drawString(250, 515, "Jour :")
    pdf.drawString(250, 490, "Heure :")
    pdf.drawString(450, 515, "Longueur manivelle :")
    pdf.drawString(450, 490, "Braquet :")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 515, PilotName)
    pdf.drawString(95, 490, Piste)
    pdf.drawString(291, 515, FileName[21:23] + "/"+FileName[19:21]+'/'+FileName[15:19])
    pdf.drawString(300, 490, FileName[24:26]+"h"+FileName[26:28])
    pdf.drawString(577, 515, str(CranksetLength)+" cm")
    pdf.drawString(512, 490, BraquetTxt)
    
    TexteExplicatif1 = "L'énergie mécanique par coup de pédale (CP) correspond à l'énergie totale que le pilote a transmis à son pédalier lors de ce CP."
    TexteExplicatif2 = "L'énergie par CP est obtenue à partir de la cadence, la force et la durée du CP, ce qui en fait un paramètre complet."
    pdf.drawString(50, 455, TexteExplicatif1)
    pdf.drawString(50, 435, TexteExplicatif2)
    
    #AFFICHAGE GRAPHIQUE EXPLICATIF
    pdf.drawInlineImage((OutputPath + "ExampleEcum.png"), -5, 125, 450, 450*0.6)
    os.remove(OutputPath + "ExampleEcum.png")
    
    # AFFICHAGE TABLEAU RECAP
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(
        432, 385, "ENERGIE MECANIQUE DU CP1, CP2, BAS DE BUTTE")
    # Tableau des données d'énergie cumulée par coup de pédale
    DataECPT = DataECP.astype(int) #passage en int
    NbCP = 2 #Nombre de CP étudiés
    DataECPT = DataECPT.iloc[0:NbCP,:].T.values.tolist() #Passage de dataframe en list de list
    ListEnergieTotale = np.sum(DataECP).astype(int).values.tolist()
    #Ajout du nom des départs
    for i in range(0,len(DataECPT)): 
        DataECPT[i].insert(NbCP,ListEnergieTotale[i])
        DataECPT[i].insert(0,LegendName[i])
    #Ajout des Headers
    Headers = ['Départ']
    for i in range(1,NbCP+1):
        Headers = Headers + ['Energie CP'+str(i)]
    Headers = Headers + ["Energie finale"]
    DataECPT = [Headers] + DataECPT
    #Création du Table pour le Canvas
    f = Table(DataECPT)
    #Récupération du max et du min de l'énergie
    LenDegrade = int((len(DataECPT)-1)/2)
    if LenDegrade > 3: LenDegrade =3
    for i in range(0,NbCP+1):
        if i<NbCP:
            #Détection des min
            idxneg = np.argsort(DataECP.iloc[i,:])
            idxneg = idxneg[0:LenDegrade]
            #Détection des max
            idxpos = np.argsort(-DataECP.iloc[i,:])
            idxpos = idxpos[0:LenDegrade]
        else :
            ListEnergieTotaleInverse = [-x for x in ListEnergieTotale]
            #Détection des min
            idxneg = np.argsort(ListEnergieTotale)
            idxneg = idxneg[0:LenDegrade]
            #Détection des max
            idxpos = np.argsort(ListEnergieTotaleInverse)
            idxpos = idxpos[0:LenDegrade]
        for j in range(0,LenDegrade):
            if j == 0:
                f.setStyle(TableStyle([('BACKGROUND',(i+1,idxneg[j]+1),(i+1,idxneg[j]+1),"#418BCF")]))#Bleu foncé
                f.setStyle(TableStyle([('BACKGROUND',(i+1,idxpos[j]+1),(i+1,idxpos[j]+1),"#FB7575")]))#Rouge foncé
            if j == 1:
                f.setStyle(TableStyle([('BACKGROUND',(i+1,idxneg[j]+1),(i+1,idxneg[j]+1),"#9DC3E6")]))#Bleu moyen
                f.setStyle(TableStyle([('BACKGROUND',(i+1,idxpos[j]+1),(i+1,idxpos[j]+1),"#FDADAD")]))#Rouge moyen
            if j == 2:
                f.setStyle(TableStyle([('BACKGROUND',(i+1,idxneg[j]+1),(i+1,idxneg[j]+1),"#DEEBF7")]))#Bleu clair
                f.setStyle(TableStyle([('BACKGROUND',(i+1,idxpos[j]+1),(i+1,idxpos[j]+1),"#FEDADA")]))#Rouge clair  
    #Paramètres d'affichage du Table
    f.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),"Helvetica"),
                           ('FONTNAME',(0,0),(0,-1),"Helvetica-Bold"),
                           ('BACKGROUND',(0,0),(-1,0),"#E7E6E6"),
                           ('FONTSIZE',(0,0),(-1,-1),12),
                           ('ALIGN',(0,0),(-1,-1),'CENTER'),
                           ('GRID',(0,0),(-1,-1),0.25,colors.black),
                           ('BOX',(0,0),(0,-1),1,colors.black),
                           ('BOX',(0,0),(-1,0),1,colors.black),
                           ('BOX',(0,0),(-1,-1),1,colors.black),
                           ('BOX',(-1,0),(-1,-1),1,colors.black)
                           ]))
    f.wrapOn(pdf, 20, 10)
    f.drawOn(pdf, 435, 130)#Position du coin inférieur gauche du Table
    
    #Ajout logo et cadre
    pdf.drawImage((OutputPath+"Logos.png"), 730, 490, 75, 75,mask='auto')
    pdf.drawImage((OutputPath+"Cadre.png"), -2, -4, 848, 604,mask='auto')
    
    
    # AFFICHAGE GRAPHIQUES MOYENS
    if len(FileStartAnalysis)<6 :
        # Nouvelle page
        pdf.showPage()
        # Titre page
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(310, 530, "SUPERPOSITION DES DONNEES")
        # Cadre legende
        pdf.drawImage((OutputPath+"BackgroundLegend.png"),260,476,320,35,mask='auto')
        # Legende départ
        pdf.setFont("Helvetica", 12)
        pdf.setFillColor(Couleurs[i])
        pdf.rect(285,493,12,1, fill=1, stroke=0)
        pdf.setFillColor(colors.black)
        pdf.drawString(310, 490, "Départ")
        # Legende dispersion
        pdf.setFillColor("#D9D9D9")
        pdf.rect(375,490,12,8, fill=1, stroke=0)
        pdf.setFillColor(colors.black)
        pdf.drawString(400, 490, "Dispersion")
        # Legende moyenne
        pdf.setFillColor(colors.black)
        pdf.drawString(485, 490, "---   Moyenne")
        
        # Affichage graphiques
        pdf.drawInlineImage((OutputPath+"Comparaison_Forces.png"), -15, 250, 450, 225)
        os.remove(OutputPath+"Comparaison_Forces.png")
        pdf.drawInlineImage((OutputPath+"Comparaison_Puissances.png"), -15, 25, 450, 225)
        os.remove(OutputPath+"Comparaison_Puissances.png")
        pdf.drawInlineImage((OutputPath+"Comparaison_Cadences.png"), 395, 250, 450, 225)
        os.remove(OutputPath+"Comparaison_Cadences.png")
        pdf.drawInlineImage((OutputPath+"Comparaison_EnergiesCumulees.png"), 395, 25, 450, 225)
        os.remove(OutputPath+"Comparaison_EnergiesCumulees.png")

        #Ajout logo et cadre
        pdf.drawImage((OutputPath+"Logos.png"), 730, 490, 75, 75,mask='auto')
        pdf.drawImage((OutputPath+"Cadre.png"), -2, -4, 848, 604,mask='auto')

    
    
    # AFFICHAGE DONNEES POUR CHAQUE DEPART
    for i in range(0,len(FileStartAnalysis)):
        # Nouvelle page
        pdf.showPage()
        # Titre page
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(310, 530, "DEPART "+LegendName[i])
        # Cadre legende
        pdf.drawImage((OutputPath+"BackgroundLegend.png"),260,476,320,35,mask='auto')
        # Legende départ
        pdf.setFont("Helvetica", 12)
        pdf.setFillColor(Couleurs[i])
        pdf.rect(285,493,12,1, fill=1, stroke=0)
        pdf.setFillColor(colors.black)
        pdf.drawString(310, 490, "Départ ")
        # Legende dispersion
        pdf.setFillColor("#D9D9D9")
        pdf.rect(375,490,12,8, fill=1, stroke=0)
        pdf.setFillColor(colors.black)
        pdf.drawString(400, 490, "Dispersion")
        # Legende moyenne
        pdf.setFillColor(colors.black)
        pdf.drawString(485, 490, "---   Moyenne")
        
        # Affichage graphiques
        pdf.drawInlineImage((OutputPath+"Comparative_ForceTotale_"+LegendName[i]+".png"), -15, 250, 450, 225)
        os.remove(OutputPath+"Comparative_ForceTotale_"+LegendName[i]+".png")
        pdf.drawInlineImage((OutputPath+"Comparative_PuissanceTotale_"+LegendName[i]+".png"), -15, 25, 450, 225)
        os.remove(OutputPath+"Comparative_PuissanceTotale_"+LegendName[i]+".png")
        pdf.drawInlineImage((OutputPath+"Comparative_Cadence_"+LegendName[i]+".png"), 395, 250, 450, 225)
        os.remove(OutputPath+"Comparative_Cadence_"+LegendName[i]+".png")
        pdf.drawInlineImage((OutputPath+"Comparative_Ecum_"+LegendName[i]+".png"), 395, 25, 450, 225)
        os.remove(OutputPath+"Comparative_Ecum_"+LegendName[i]+".png")

        #Ajout logo et cadre
        pdf.drawImage((OutputPath+"Logos.png"), 730, 490, 75, 75,mask='auto')
        pdf.drawImage((OutputPath+"Cadre.png"), -2, -4, 848, 604,mask='auto')

    # SAVE
    pdf.save()
