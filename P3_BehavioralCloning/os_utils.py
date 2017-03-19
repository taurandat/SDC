import errno
import json
import os


def save_model(model, model_name='model.json', weights_name='model.h5'):
    """
    Save the model into the hard disk
    """
    silent_delete(model_name)
    silent_delete(weights_name)

    json_string = model.to_json()
    with open(model_name, 'w') as outfile:
        json.dump(json_string, outfile)

    model.save_weights(weights_name)


def silent_delete(file):
    """
    This method delete the given file from the file system if it is available
    Source: http://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
    """
    try:
        os.remove(file)

    except OSError as error:
        if error.errno != errno.ENOENT:
            raise
