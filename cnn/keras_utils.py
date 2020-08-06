from keras.utils import Sequence
class My_Generator(Sequence):
    
    def __init__(self, dataset, batch_size, image_size, channels):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels

    def __len__(self):
        return self.dataset.num_examples // self.batch_size + 1

    def __getitem__(self, idx):
        
        [x_batch, y_batch] =  self.dataset.get_batch(idx, self.batch_size, self.image_size)
        x_batch = x_batch[:,:,:,self.channels]

        return x_batch, y_batch


from keras.utils import Sequence
class My_GeneratorIncremental(Sequence):
    
    def __init__(self, dataset, batch_size, image_size, channels):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels

    def __len__(self):
        return self.dataset.num_examples // self.batch_size + 1

    def __getitem__(self, idx):
        
        [x_batch, y_batch] =  self.dataset.next_batch(self.batch_size, self.image_size)
        x_batch = x_batch[:,:,:,self.channels]


        return x_batch, y_batch