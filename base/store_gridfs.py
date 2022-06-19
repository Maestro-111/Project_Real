import pickle
from re import A
from base.mongo import MongoDB
from base.store_file import FileStore
import gridfs
import pymongo

mongoDb = MongoDB().getDb()


def getGridFSBucket(collection):
    global mongoDb
    return gridfs.GridFSBucket(mongoDb, collection)


def getGridFS(collection):
    global mongoDb
    return gridfs.GridFS(mongoDb, collection)


class GridfsStore(FileStore):
    """Store file in a Gridfs."""

    def __init__(
        self,
        # db: pymongo.database.Database,
        collection: str = 'fs',
        prefix: str = ''
    ):
        #self.db = db
        self.collection = collection
        # self.fsbucket = gridfs.GridFSBucket(self.db, collection)
        # self.fs = gridfs.GridFS(self.db, collection)
        self.prefix = prefix

    def set_prefix(self, prefix):
        self.prefix = prefix

    def save_data(self, filename, data, accuracy=0, meta: dict = None):
        """Save the model."""
        filename = self.prefix + filename
        if meta is not None:
            meta['accuracy'] = accuracy
        f = getGridFSBucket(self.collection).open_upload_stream(
            filename=filename, metadata=meta)
        pickle.dump(data, f)
        f.close()
        return filename

    def load_data(self, filename):
        """Load the model."""
        filename = self.prefix + filename
        f = getGridFS(self.collection).get_last_version(filename=filename)
        if f is None:
            raise ValueError('No model file found.')
        if f.metadata is None:
            accuracy = 0
        else:  # f.metadata is not None
            accuracy = f.metadata['accuracy'] or 0
            meta = f.metadata
        data = pickle.load(f)
        f.close()
        return {data, accuracy, meta}

    def delete_unused(self):
        """Delete unused model files."""
        pass
