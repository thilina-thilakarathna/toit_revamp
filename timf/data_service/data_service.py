#get local data and get remore data, in realworld this will get dfata from other microcel;ls here it vill get the data from the emuilated source



class DataService:
    def __init__(self, local_data_source, remote_data_source):
        self.local_data_source = local_data_source
        self.remote_data_source = remote_data_source

    def get_local_data(self):
        # Simulate fetching data from a local source
        return self.local_data_source.fetch_data()

    def get_remote_data(self):
        # Simulate fetching data from a remote source
        return self.remote_data_source.fetch_data()