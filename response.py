
def response_json(success, message, result = False):
    code = 200 if success else 400;
    return {
        "success": success,
        "message": message,
        "result": result,
    }, code

def success(result):
    return response_json(True, "Success!", result)

def error(message):
    return response_json(False, message)