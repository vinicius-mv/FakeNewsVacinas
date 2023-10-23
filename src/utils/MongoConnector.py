import pymongo

from src import config as cfg


class MongoConnector(object):
    URI = cfg.mongodb["uri"]
    PORT = cfg.mongodb["port"]
    CLIENT = None
    DATABASE = None
        
    @staticmethod
    def initialize():
        MongoConnector.CLIENT = pymongo.MongoClient(MongoConnector.URI)
        MongoConnector.DATABASE = MongoConnector.CLIENT["tcc"]

    @staticmethod
    def insert(collection, data):
        MongoConnector.DATABASE[collection].insert(data)

    @staticmethod
    def insert_many(collection, data):
        MongoConnector.DATABASE[collection].insert_many(data)

    @staticmethod
    def find(collection, query):
        return MongoConnector.DATABASE[collection].find(query)

    @staticmethod
    def find_one(collection, query):
        return MongoConnector.DATABASE[collection].find_one(query)

    @staticmethod
    def delete_one(collection, query):
        return MongoConnector.DATABASE[collection].delete_one(query)

    @staticmethod
    def delete_many(collection, query):
        return MongoConnector.DATABASE[collection].delete_many(query)

    # myquery = { "address": "Valley 345" }
    # new_values = { "$set": { "address": "Canyon 123" } }
    @staticmethod
    def update_one(collection, query, new_values):
        return MongoConnector.DATABASE[collection].update_one(query, new_values)

    @staticmethod
    def test_connection(collection):
        try:
            MongoConnector.DATABASE[collection].admin.command("ping")
            print("Pinged your deployment. You successfully connected to MongoDB!")
            # Send a ping to confirm a successful connection
        except Exception as e:
            print(e)
          
    @staticmethod
    def rename_collection(collection, new_collection_name):
        try:
            MongoConnector.DATABASE[collection].rename(new_collection_name)
        except Exception as e:
            print(e)
            
    @staticmethod          
    def close():
        print("Connection closed")
        MongoConnector.CLIENT.close()
