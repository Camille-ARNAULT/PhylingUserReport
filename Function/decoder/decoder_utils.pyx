import re
import struct
import logging
import ujson
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import decoder.calibration_use as calib

try:
    from phylingUtils.data_layer.s3 import S3
except Exception:
    class S3:
        def get_filestream_readonly(filename):
            return open(filename, "rb+")

        @classmethod
        def get_file_bytes(
            cls,
            filename,
        ) -> bytes:
            with open(filename, "rb") as f:
                filecontent = f.read()
            return filecontent

class EndOfFileException(Exception):
    pass

cdef dict sizeElemDict = {
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float32": 4,
    "float64": 8,
}

cpdef unsigned int getSizeElem(str valType):
    if valType in sizeElemDict:
        return sizeElemDict[valType]
    raise Exception("Invalid type: {}".format(valType))


cpdef str getTypeElem(str valType):
    if valType.startswith("uint"):
        return "uint"
    if valType.startswith("int"):
        return "int"
    if valType.startswith("float"):
        return "float"
    raise Exception("Invalid type: {}, must be uintX, intX or floatX".format(valType))


cpdef object getElem(char * content, int curPos, str valType, int content_size=0):
    cdef int size = getSizeElem(valType) 
    if content_size > 0 and curPos + size >= content_size:
        raise Exception("File is broken, unpack requires a buffer of {} bytes".format(size))
    if getTypeElem(valType) == "uint":
        return int.from_bytes(
            content[curPos : curPos + size],  # noqa
            byteorder="little",
            signed=False,
        )
    if getTypeElem(valType) == "int":
        return int.from_bytes(
            content[curPos : curPos + size],  # noqa
            byteorder="little",
            signed=True,
        )
    if getTypeElem(valType) == "float":
        return float(
            struct.unpack(
                "<f" if size == 4 else "<d", content[curPos : curPos + size]  # noqa
            )[0]
        )


cpdef object applyFactor(object value, object curMod, object elem):
    if curMod["type"] in ("imu", "miniphyling", "ble") and elem["type"] == "int16":
        if "acc_factor" in curMod and elem["name"].startswith("acc_"):
            return value * curMod["acc_factor"]
        elif "gyro_factor" in curMod and elem["name"].startswith("gyro_"):
            return value * curMod["gyro_factor"]
        elif "mag_factor" in curMod and elem["name"].startswith("mag_"):
            return value * curMod["mag_factor"]
    elif curMod["type"] in ("adc", "analog") and elem["type"] == "uint16":
        if "factor" in curMod:
            return value * curMod["factor"]
    return value


cpdef str getModName(dict header, char * content, int curPos):
    cdef str modName
    for modName, mod in header["modules"].items():
        if mod["id"] == content[curPos]:
            return modName
    return ""


cpdef str getVarName(dict header, str modName, str varBaseName):
    if "variablesNames" in header["description"] \
    and header["description"]["variablesNames"] is not None \
    and modName in header["description"]["variablesNames"] \
    and varBaseName in header["description"]["variablesNames"][modName]:
        return header["description"]["variablesNames"][modName][varBaseName]
    return varBaseName


cpdef object loadOne(dict header, char * content, int curPos, dict calib_dict=None, int content_size=0):
    cdef str curModName
    cdef object curMod
    cdef double modTime
    cdef object modVal
    cdef object modValNamed
    cdef dict data = None
    cdef int tmpCurPos = curPos

    missingByteSize = 0
    while content_size == 0 or tmpCurPos < content_size:
        tmpCurPos = curPos + missingByteSize
        curModName = getModName(header, content, tmpCurPos)
        if curModName == "":
            if content_size == 0:
                raise Exception("module id {} does not exist".format(content[curPos]))
            missingByteSize += 1
            continue
        curMod = header['modules'][curModName]
        if content_size > 0 and tmpCurPos + curMod["size"] > content_size:
            raise EndOfFileException(f"End of file is corrupted (total: {missingByteSize} bytes corrupted from position {curPos})")

        try:
            modTime = 0
            modVal = {}
            modValNamed = {}
            for elem in curMod["description"]:
                if elem["name"] == "id":
                    pass
                elif elem["name"] == "time":
                    modTime = getElem(content, tmpCurPos, elem["type"], content_size=content_size)
                else:
                    varName = getVarName(header, curModName, elem["name"])
                    modVal[elem["name"]] = getElem(content, tmpCurPos, elem["type"], content_size=content_size)
                    modVal[elem["name"]] = applyFactor(modVal[elem["name"]], curMod, elem)
                    modValNamed[varName] = elem["name"]
                tmpCurPos += getSizeElem(elem["type"])
            modVal["T"] = modTime / 1000000 - header["description"]["epoch"]
            if modVal["T"] > 3600 * 24 * 10 or modVal["T"] < -100:  # if time if over 10 days or in the past
                missingByteSize += 1
                continue
            modValNamed["T"] = "T"
            if calib_dict is not None:
                modVal = calib.calibration(modVal, curModName, calib_dict)
            for key, val in modValNamed.items():
                modValNamed[key] = modVal[val]
            data = {
                "type": curMod["type"],
                "name": curModName,
                "data": modValNamed,
            }
            break
        except Exception as e:
            if content_size == 0:
                raise Exception(f"Error on decoding: {str(e)}")
            logging.error(f"Error on decoding: {str(e)}. trying next module")
            missingByteSize += 1
    if data:
        msg = f"Missing some data ({missingByteSize} bytes at {modVal['T']}s)"
    else:
        msg = f"Missing some data ({missingByteSize} bytes from position {curPos})"
    if missingByteSize > 0:  # if we have lost some data
        logging.error(msg)
        if not data:  # if it's impossible to decode some data
            raise Exception(msg)
    return data, missingByteSize + curMod["size"], modTime / 1000000


cpdef object getCalibration(str filename):
    cdef object calibration = ""
    cdef str type = ""
    cdef bytes ln
    cdef bytes filecontent = S3.get_file_bytes(filename)
    for ln in filecontent.splitlines(True):
        if ln == b"":
            break
        if ln == b"<== description ==>\n":
            type = ln.decode("utf-8")
        elif ln == b"<== calibration ==>\n":
            type = ln.decode("utf-8")
        elif ln == b"<== data ==>\n":
            type = ln.decode("utf-8")
        else:
            if type == "<== description ==>\n":
                pass
            elif type == "<== calibration ==>\n":
                calibration += ln.decode("utf-8")
            elif type == "<== data ==>\n":
                break
    if len(calibration) > 0:
        calibration = ujson.loads(calibration)
    return calibration


cpdef object updateCalibration(str filename, str oldFilename, object calibration):
    cdef str type = ""
    cdef bytes ln
    cdef bytes filecontent = S3.get_file_bytes(filename)

    if not S3.file_exists(oldFilename):
        logging.info(f"Save a copy of file in {oldFilename}")
        S3.copy_file(filename, oldFilename)

    pattern = rb"<== calibration ==>\n(.*?)<== data ==>\n"
    replacement = f"<== calibration ==>\n{ujson.dumps(calibration, 4)}\n<== data ==>\n".encode()
    filecontent = re.sub(pattern, replacement, filecontent, flags=re.DOTALL)

    S3.add_file_bytes(filename, filecontent)


cpdef object loadFile(str filename, bint verbose=False, double startingTime=-1):
    if startingTime == -1:
        startingTime = time.time()
    logging.info("load {}...".format(filename))
    cdef str header = ""
    cdef object calibration = ""
    cdef list content = []
    cdef str type = ""
    cdef int totalSz = 0
    cdef int lastPrintSz = 0
    cdef bytes ln
    cdef bytes content_byte
    cdef object header_dict
    cdef bytes filecontent = S3.get_file_bytes(filename)
    for ln in filecontent.splitlines(True):
        totalSz += len(ln)
        # print every 10Mb
        if verbose:
            if totalSz - lastPrintSz > 10000000:
                logging.info(
                    "read {}Mb in {:.2f}s".format(
                        int(totalSz / 1000000), time.time() - startingTime
                    )
                )
                lastPrintSz = totalSz - (totalSz % 10000000)
        if ln == b"":
            break
        if ln == b"<== description ==>\n":
            type = ln.decode("utf-8")
        elif ln == b"<== calibration ==>\n":
            type = ln.decode("utf-8")
        elif ln == b"<== data ==>\n":
            type = ln.decode("utf-8")
        else:
            if type == "<== description ==>\n":
                header += ln.decode("utf-8")
            elif type == "<== calibration ==>\n":
                calibration += ln.decode("utf-8")
            elif type == "<== data ==>\n":
                # content = b"".join([content, ln])
                content.append(ln)
    # print on the end
    if verbose:
        logging.info(
            "read {:.3f}Mb in {:.2f}s".format(
                totalSz / 1000000, time.time() - startingTime
            )
        )

    content_byte = b"".join(content)
    header_dict = ujson.loads(header)
    if len(calibration) > 0:
        calibration = ujson.loads(calibration)
    else:
        calibration = {}
    return header_dict, calibration, content_byte


cpdef void printDecodingInfos(int statsAll, int percent):
    if statsAll < 1000:
        logging.info(
            "[{percent:3}%]: {val:3.0f} data decoded".format(
                percent=percent, val=float(statsAll)
            )
        )
    elif statsAll < 1000000:
        logging.info(
            "[{percent:3}%]: {val:3.0f}k data decoded".format(
                percent=percent, val=float(statsAll / 1000)
            )
        )
    else:
        logging.info(
            "[{percent:3}%]: {val:3.0f}M data decoded".format(
                percent=percent, val=float(statsAll / 1000000.0)
            )
        )


cpdef dict decode(str filename, bint verbose=True, dict config_client=None):
    logging.info("<== decode start [{}] ==>".format(filename))
    cdef bint retSuccess = True
    cdef double start = time.time()

    cdef object header
    cdef object calibration
    cdef bytes content
    header, calibration, content = loadFile(
        filename, verbose=verbose, startingTime=start
    )
    logging.info("start decoding file")
    cdef int curPos = 0
    cdef dict jsonData = {"modules": {}}
    cdef int statsAll = 0
    cdef dict stats = {}
    for modName in header["modules"].keys():
        stats[modName] = 0
    cdef double lastTime = 0
    cdef int dev_id

    cdef int percent
    cdef object newData
    cdef int size
    cdef double timeSec
    cdef str module_name
    cdef list description
    cdef int content_size = len(content)
    try:
        dev_id = int(header["description"]["folder_name"].split("_")[0])
    except Exception:
        dev_id = -1
    if content_size == 0:
        logging.error("File is empty")
        raise Exception("File is empty")
    while 1:
        if content_size <= curPos:
            break
        if content[curPos] == 0:  # id for stopping parsing
            percent = round(curPos / len(content) * 100)
            if percent > 95:
                break
            logging.error("Current module ID is 0, skipping")
        try:
            newData, size, timeSec = loadOne(header, content, curPos, calib_dict=calibration, content_size=content_size)
        except EndOfFileException as e:
            logging.error(f"[ERROR]: unexpected error at end of file {e}")
            retSuccess = True
            break
        except Exception as e:
            logging.error(f"[ERROR]: unexpected error, {e}")
            # retSuccess = False
            break
        statsAll += 1
        module_name = newData["name"]
        stats[module_name] += 1
        curPos += size

        # if first data saving
        if module_name not in jsonData["modules"]:
            jsonData["modules"][module_name] = {
                "description": {"rate": header["modules"][module_name]["rate"]},
                "data": {},
                "data_info": {},
            }
            if "name" in header["modules"][module_name]:
                jsonData["modules"][module_name]["description"]["name"] = header["modules"][module_name]["name"]
            jsonData["modules"][module_name]["data"]["T"] = []
            jsonData["modules"][module_name]["data_info"]["T"] = {
                "unit": "s",
                "description": "Temps",
            }
            description = header["modules"][module_name]["description"]
            for i in range(2, len(description)):
                realVarName = getVarName(header, module_name, description[i]["name"])
                descr = ""
                if config_client and module_name in config_client:
                    if description[i]["name"] in config_client[module_name]:
                        descr = config_client[module_name][realVarName][
                            "description"
                        ]
                jsonData["modules"][module_name]["data"][realVarName] = []
                jsonData["modules"][module_name]["data_info"][
                    realVarName
                ] = {"unit": description[i]["unit"], "description": descr}

        # save data
        if timeSec > lastTime:
            lastTime = timeSec
        for name in newData["data"].keys():
            jsonData["modules"][newData["name"]]["data"][name].append(
                newData["data"][name]
            )

        if verbose:
            if statsAll % 10000 == 0:
                percent = round(curPos / len(content) * 100)
                printDecodingInfos(statsAll, percent)

    jsonData["description"] = {
        "nbData": statsAll,
        "totalTime": lastTime - header["description"]["epoch"],
        "startingTime": header["description"]["epoch"],
        "device_id": dev_id,
        "specificData": {},
    }
    if "specificData" in header["description"] and header["description"]["specificData"] != "":
        try:
            jsonData["description"]["specificData"] = header["description"]["specificData"]
        except Exception:
            jsonData["description"]["specificData"] = {}
            logging.warn("Unable to load specific data")

    if verbose:
        percent = round(curPos / len(content) * 100)
        printDecodingInfos(statsAll, percent)

    logging.info("total: {} data".format(statsAll))
    for key, val in stats.items():
        logging.info("\t{}: {} datas".format(key, val))
    logging.info("<== decode end [{}] ==>".format("SUCCESS" if retSuccess else "ERROR"))
    logging.info("File decoded in {:.3f}s".format(time.time() - start))
    if retSuccess:
        return jsonData
    else:
        raise Exception("Error during decoding")
