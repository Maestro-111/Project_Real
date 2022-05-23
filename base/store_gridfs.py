import pickle
from re import A
from store_file import FileStore
from pymongo import MongoDB
import gridfs
import pymongo


class GridfsStore(FileStore):
    """Store file in a Gridfs."""

    def __init__(self, db: pymongo.database.Database, collection: str = 'fs', prefix: str = ''):
        self.db = db
        self.fsbucket = gridfs.GridFSBucket(self.db, collection)
        self.fs = gridfs.GridFS(self.db, collection)
        self.prefix = prefix

    def set_prefix(self, prefix):
        self.prefix = prefix

    def save_data(self, filename, data, accuracy=0):
        """Save the model."""
        filename = self.prefix + filename
        f = self.fsbucket.open_upload_stream(
            filename=filename, metadata={'accuracy': accuracy})
        pickle.dump(data, f)
        f.close()
        return filename

    def load_data(self, filename):
        """Load the model."""
        filename = self.prefix + filename
        f = self.fs.get_last_version(filename=filename)
        if f is None:
            raise ValueError('No model file found.')
        if f.metadata is None:
            accuracy = 0
        else:  # f.metadata is not None
            accuracy = f.metadata['accuracy'] or 0
        data = pickle.load(f)
        f.close()
        return {data: accuracy}

    def delete_unused(self):
        """Delete unused model files."""
        pass
