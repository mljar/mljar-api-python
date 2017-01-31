
class MljarException(Exception):
    """Base exception class for this module"""
    pass

class TokenException(MljarException):
    pass

class DataReadException(MljarException):
    pass

class JSONReadException(MljarException):
    pass

class NotFoundException(MljarException):
    pass

class AuthenticationException(MljarException):
    pass

class BadRequestException(MljarException):
    pass

class BadValueException(MljarException):
    pass

class UnknownProjectTask(MljarException):
    pass

class IncorrectInputDataException(MljarException):
    pass

class FileUploadException(MljarException):
    pass

class CreateProjectException(MljarException):
    pass

class CreateDatasetException(MljarException):
    pass

class CreateExperimentException(MljarException):
    pass

class UndefinedExperimentException(MljarException):
    pass

class DatasetUnknownException(MljarException):
    pass

class PredictionDownloadException(MljarException):
    pass    
