
import os
import pickle
import re


class FileStore:
    dir = None

    def __init__(self, dir=None, prefix=''):
        self.dir = dir
        self.prefix = prefix
        
    def set_prefix(self, prefix):
        self.prefix = prefix

    def save_data(self, filename, data, accuracy=0):
        """Save the model."""
        filename = self.prefix + filename
        fullpath = os.path.join(
            dir, f'{filename}.{accuracy:03}.pkl')
        with open(fullpath, 'wb') as f:
            pickle.dump(data, f)
        return fullpath

    def load_data(self, filename):
        """Load the model."""
        filename = self.prefix + filename
        files = [f for f in os.listdir(
            self.dir) if f.startswith(filename)]
        if len(files) > 0:
            # get the highest accuracy score file name
            files.sort(reverse=True)
            fullpath = os.path.join(self.dir, files[0])
        else:
            raise ValueError('No model file found.')
        with open(fullpath, 'rb') as f:
            data = pickle.load(f)
        pattern = re.compile(r".*\.(\d{3})\.pkl")
        match = pattern.match(files[0])
        accuracy = int(match.group(1))
        return (data, accuracy)

    def delete_unused(self):
        """Delete unused model files."""
        pass
