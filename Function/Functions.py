"""
@author: camille.arnault@univ-poitiers.fr
"""

import os
from reportlab.pdfgen.canvas import Canvas
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
    # Magneto data filtering & magneto pedalier peaks detection
    try:
        DataMagnetoPedalierFiltrees = FiltrageButterworth(
            DataPedalier['magneto_pedalier'], 200, 50)
        MagnetoPeaks, _ = find_peaks(DataMagnetoPedalierFiltrees, height=(
            10000, None), prominence=(500, None))
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
        SommeDeplacementAngulairePedalier = np.cumsum(
            DeplacementAngulairePedalier)
        SommeDeplacementAngulairePedalierCorrige = [
            0 for i in range(0, len(SommeDeplacementAngulairePedalier))]

        for i in range(0, len(MagnetoPeaks)+1):
            if i == 0:
                offset = AngleReel - \
                    SommeDeplacementAngulairePedalier[MagnetoPeaks[i]]
                for j in range(0, MagnetoPeaks[0]+1):
                    SommeDeplacementAngulairePedalierCorrige[j] = SommeDeplacementAngulairePedalier[j]+offset
            if i == len(MagnetoPeaks):
                OffsetContinu = (360*(i-1)+AngleReel) - \
                    SommeDeplacementAngulairePedalier[MagnetoPeaks[i-1]]
                for j in range(MagnetoPeaks[i-1], len(SommeDeplacementAngulairePedalier)):
                    SommeDeplacementAngulairePedalierCorrige[j] = SommeDeplacementAngulairePedalier[j]+OffsetContinu
            else:
                OffsetContinu = (360*(i-1)+AngleReel) - \
                    SommeDeplacementAngulairePedalier[MagnetoPeaks[i-1]]
                for j in range(MagnetoPeaks[i-1], MagnetoPeaks[i]+1):
                    SommeDeplacementAngulairePedalierCorrige[j] = SommeDeplacementAngulairePedalier[j]+OffsetContinu

                OffsetVariable = (
                    360*i+AngleReel)-SommeDeplacementAngulairePedalierCorrige[MagnetoPeaks[i]]
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
        print(Fore.RED + "ERROR : Total crank displacement could not be calculated.")
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
            DataTopTour['magneto_toptour'], 800, 50)

        # Magnetic peaks detection & Velocity calculation
        XVitesseTopTour, XDistanceTopTour, VitesseTopTour, DistanceTopTourM, PeaksNeg, PeaksPos = CalculVitesseTopTourArriere(
            DataTopTour['temps_toptour'], DataMagnetoFiltrees, EspacementAimant, CirconferenceRoue, Sensitivity=6)
        # Verification of Magnetic peaks detection
        if VerificationRevolutionCounterPeaks in ['Oui', 'oui', 'OUI', 'o', 'O', 'YES', 'Yes', 'yes', 'Y', 'y']:
            plt.figure()
            plt.plot(DataMagnetoFiltrees)
            plt.plot(PeaksPos, DataMagnetoFiltrees[PeaksPos], 'x')
            plt.plot(PeaksNeg, DataMagnetoFiltrees[PeaksNeg], 'x')
            plt.grid()

        # Top Tour Velocity Resample
        # Create time data before first velocity calculated (because of the spacement between magnet, velocity is not calculated at specific frequency as other mesurement could.)
        ttoptour = DataTopTour['temps_toptour'][XVitesseTopTour][DataTopTour['temps_toptour']
                                                                 [XVitesseTopTour].index[0]]
        LongueurIntervalle1 = IndexNearestValue(x, ttoptour)
        if x[LongueurIntervalle1] > ttoptour:
            LongueurIntervalle1 = LongueurIntervalle1 - 1
        IntervalleZeros1 = pd.Series(
            [0 for i in range(0, LongueurIntervalle1)])
        IntervalleTemps1 = pd.Series(
            [x[0]+(1/FreqAcq)*i for i in range(0, LongueurIntervalle1)])
        # Create time data after last velocity calculated
        ttoptour = DataTopTour['temps_toptour'][XVitesseTopTour][DataTopTour['temps_toptour']
                                                                 [XVitesseTopTour].index[len(DataTopTour['temps_toptour'][XVitesseTopTour])-1]]
        LongueurIntervalle2 = IndexNearestValue(x, ttoptour)
        if x[LongueurIntervalle2] < ttoptour:
            LongueurIntervalle2 = LongueurIntervalle2 + 1
        IntervalleZeros2 = pd.Series(
            [0 for i in range(0, len(x)-LongueurIntervalle2)])
        IntervalleTemps2 = pd.Series(
            [x[LongueurIntervalle2]+(1/FreqAcq)*i for i in range(0, len(x)-LongueurIntervalle2+1)])
        IntervalleDistance2 = pd.Series([DistanceTopTourM[len(
            DistanceTopTourM)-1] for i in range(0, len(x)-LongueurIntervalle2)])
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
    PuissancePeaks, _ = find_peaks(-PowerData[LimitsPedalingArea[0]                                   :LimitsPedalingArea[1]], height=(None, 1000), prominence=(500, None))
    PuissancePeaks = PuissancePeaks+LimitsPedalingArea[0]
    PuissancePeaks = np.insert(PuissancePeaks, 0, LimitsPedalingArea[0])
    IndexPedalStroke = PuissancePeaks

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
        # 1 = nb area to detect
        # 200 = who much frame will be analyzed after the user selection to detect start
        FrameInit, FrameEnd = DetectionDepartsSemiAuto(
            Data['CadenceTrMin'], VerificationStartDetection=VerificationStartDetection)
        print("- Start detected")
    except:
        print(Fore.RED + "ERROR : Start could not be detected.")

    # Pedal stroke area detection
    try:
        # 25 = How much frame will be analyzed before and after the user selection to find the pedal stroke area
        IndexZP = FindZonesPedalees(Data["CadenceTrMin"], FrameInit, FrameEnd)
        print("- Pedal stroke area detected.")
    except:
        print(Fore.RED + 'ERROR : Pedal stroke area could not be detected.')

    # Pedal stroke detection
    try:
        # Get Pedal Stroke Index
        IndexCP = DetectionCoupPedaleDepart(
            Data["PuissanceTotale"], FrameInit, IndexZP, VerificationPedalStroke=VerificationPedalStroke)
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
        "StartAnalysisResults"]['Vitesse'] = Data['VitessePedalier'][FrameInit:FrameEnd]

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
                              (pd.Series(locals()["StartAnalysisResults"]['Vitesse'])).reset_index(drop=True)], axis=1, sort=False)
        all_data = all_data.set_axis(['FrameInit', 'FrameEnd', 'IndexCP', 'IndexMeanCP', 'Temps',
                                      'ForceGaucheInstant', 'ForceDroiteInstant', 'ForceInstant',
                                      'ForceMoyDCP', 'ForceMaxDCP', 'ForceMaxDCPIndex', 'ImpulsionDCP',
                                      'PuissanceInstant', 'PuissanceMoyDCP', 'PuissanceMaxDCP', 'PuissanceMaxDCPIndex',
                                      'RMPD70Range', 'RMPD70Pourcent',
                                      'TravailDCP', 'TravailCumuleDCP', 'TravailCumule',
                                      'AngleManivelleGauche', 'AngleManivelleDroite',
                                      'AngleManivelleDepart', 'AngleManivelleReculMax', 'AngleTotalRecul',
                                      'Cadence', 'VitessePedalier'], axis=1)

        # all_data.to_csv(OutputPath + FileName + '_StartAnalysis.xlsx',mode='a',index=False)

        shutil.copy(OutputPath+"TemplateStartAnalysis.xlsx",
                    OutputPath+FileName+"_StartAnalysis.xlsx")
        with pd.ExcelWriter(OutputPath + FileName + '_StartAnalysis.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            all_data.to_excel(writer, sheet_name='DONNEES', index=False)

        print("EXTRACTION CSV REUSSIE.")
    except:
        print(Fore.RED + "ERROR : CSV Extraction failed.")

    return locals()["StartAnalysisResults"]


def ReportEdition(InputPath, FileName, OutputPath, PilotName, BraquetTxt, CranksetLength, Piste):

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
            Data["ForceMaxDCPIndex"][i]-0.05, Data["ForceMaxDCP"][i]+50))
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

    # Travail Cumulé
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    plt.grid()
    plt.xlim([0, 3.5])
    for i in range(0, NbCP):
        plt.plot([Data["Temps"][0], Data["Temps"][699]], [Data["TravailCumuleDCP"],
                 Data["TravailCumuleDCP"]], linestyle='--', color='#A5A5A5', linewidth=0.5)
        ax.annotate(str(round(Data["TravailCumuleDCP"][i])), xy=(
            0.05, Data["TravailCumuleDCP"][i]+10), color='#A5A5A5')
    plt.plot(Data["Temps"][0:700], Data["TravailCumule"]
             [0:700], color='#00B0F0')
    plt.title("ENERGIE CUMULEE")
    plt.xlabel('Temps (s)')
    plt.ylabel('Energie (J)')
    plt.savefig(OutputPath+FileName+"_TravailCumule.png")
    plt.close()

    #%% PAGE CONFIGURATION
    pdf = Canvas(OutputPath+FileName+"_Report.pdf")
    # TITLE
    pdf.setFont("Times-Bold", 18)
    pdf.drawString(200, 800, "RAPPORT D'ANALYSE")

    # PILOT INFOS
    pdf.setFont("Times-Bold", 12)
    pdf.drawString(50, 775, "Pilote :")
    pdf.drawString(50, 750, "Braquet :")
    pdf.drawString(50, 725, "Longueur manivelle :")
    pdf.drawString(350, 775, "Piste :")
    pdf.drawString(350, 750, "Jour :")
    pdf.drawString(350, 725, "Heure :")
    pdf.setFont("Times-Roman", 12)
    pdf.drawString(90, 775, PilotName)
    pdf.drawString(102, 750, BraquetTxt)
    pdf.drawString(162, 725, str(CranksetLength)+" cm")
    pdf.drawString(385, 775, Piste)
    pdf.drawString(385, 750, FileName[21:23] +
                   "/"+FileName[19:21]+'/'+FileName[15:19])
    pdf.drawString(393, 725, FileName[24:26]+"h"+FileName[26:28])

    # AFFICHAGE DONNEES FORCE
    pdf.setFont("Times-Bold", 12)
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

    # SAVE
    pdf.save()
