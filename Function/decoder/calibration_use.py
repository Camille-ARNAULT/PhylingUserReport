import numpy as np


##### redefine acc_gyro_correction #####
# from phyling.calib import acc_gyro_correction
def acc_gyro_correction(data, K, b):
    return (K @ (data + b).T).T


##### redefine get_value function ####
# from phylingUtils.data_layer.json_overlay import get_value
def get_value(json_file, module_name, variable_name) -> dict:
    if module_name in json_file:
        if variable_name in json_file[module_name]:
            return json_file[module_name][variable_name]
    return None


"""
==================================================================
                            CALIBRATION
    functions used to adjust the data received from the cpp
==================================================================
"""


def general_calibration(sensor_data, coefficient, offset):
    """
    Parameters:
        sensor_data (float): force data
        coefficient (float): calibration coefficient
        offset (float): calibration offset
    Return:
        calibrated data (float)
    """
    if coefficient is None:
        coefficient = 1
    if offset is None:
        offset = 0
    return (sensor_data + offset) * coefficient


def acc_gyro_calibration(sensor_data, coefficient, offset, type_sensor):
    """
    Parameters:
        sensor_data (3x1 np matrix): force data
        coefficient (3x3 np.matrix): calibration coefficient
        offset (3x1 np.matrix): offset calibration
    """
    if type_sensor not in ["acc", "gyro"]:
        raise ValueError("type_sensor should be either 'acc' or 'gyro'.")
    if coefficient is None:
        coefficient = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if offset is None:
        offset = np.array([0, 0, 0])

    data = np.array(
        [
            sensor_data[type_sensor + "_x"],
            sensor_data[type_sensor + "_y"],
            sensor_data[type_sensor + "_z"],
        ]
    )
    data = acc_gyro_correction(data, coefficient, offset)
    sensor_data[type_sensor + "_x"] = data[0]
    sensor_data[type_sensor + "_y"] = data[1]
    sensor_data[type_sensor + "_z"] = data[2]

    return sensor_data


def calibration(data, module, calib_dict):
    """
    Parameters:
       calibration (dictionary/json): contains the calibration coefficients
       data(dictionary): data received from the cpp
    """

    if module not in calib_dict:
        return data

    module_dict = calib_dict[module]
    dict_imu = {"acc": {}, "gyro": {}}
    for key, value in data.items():
        if key in ["acc_x", "acc_y", "acc_z"]:
            dict_imu["acc"][key] = value
        elif key in ["gyro_x", "gyro_z", "gyro_y"]:
            dict_imu["gyro"][key] = value
        elif key != "T" and key != "bleTime":
            data[key] = general_calibration(
                value,
                get_value(module_dict, key, "coef"),
                get_value(module_dict, key, "offset"),
            )

    # Calibrate acc & gyro
    for imu_var in dict_imu:
        if len(dict_imu[imu_var]) == 3:
            data = acc_gyro_calibration(
                data,
                get_value(module_dict, imu_var, "coef"),
                get_value(module_dict, imu_var, "offset"),
                imu_var,
            )
        else:
            for sensor in dict_imu[imu_var]:
                data[sensor] = general_calibration(
                    dict_imu[imu_var][sensor],
                    get_value(module_dict, sensor, "coef"),
                    get_value(module_dict, sensor, "offset"),
                )

    return data
