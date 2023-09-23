
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate

class KNN:
    '''
        Some variables
    '''
    valid_encodings = (1, 2,)
    valid_distance_metrics = ('euclidean', 'manhattan', 'cosine', 'minkowski3', 'minkowski4', 'chebyshev')
    columns = {
        'game_id': 0,
        'resnet_emb': 1,
        'vit_emb': 2,
        'label': 3,
        'guess_time': 4
    }
    
    
    '''
        Distance Metrics
    '''
    # Euclidean distance metric
    def Euclidean(self, sample1, sample2):
        return np.sqrt(np.sum((sample1 - sample2)**2))   
    
    # Manhattan distance
    def Manhattan(self, sample1, sample2):
        return np.sum(np.abs(sample1 - sample2)) 
    
    # Cosine distance
    def Cosine(self, sample1, sample2):
        dot_product = np.dot(sample1, sample2)
        norm_point1 = np.linalg.norm(sample1)
        norm_point2 = np.linalg.norm(sample2)
        return 1 - dot_product / (norm_point1 * norm_point2)
    
    # calculating l3 Minkowski Distance
    def Minkowski3(self, sample1, sample2):
        return np.sum(np.abs(sample1 - sample2)**3) 
    
    # calculating l4 Minkowski Distance
    def Minkowski4(self, sample1, sample2):
        return np.sum(np.abs(sample1 - sample2)**4)
    
    # Chebyshev distance
    def Chebyshev(self, sample1, sample2):
        return np.max(np.abs(sample1 - sample2))
    
    
    '''
        Data Preprocessing Techniques
    '''
    # splits data into train and test
    def split_data(self, split):
        split_index = round((100 - split)/100 * len(self.data_encodings))
        self.train_labels, self.test_labels = self.data_labels[:split_index], self.data_labels[split_index:] 
        self.train_encodings, self.test_encodings = self.data_encodings[:split_index], self.data_encodings[split_index:]
          
        
    '''
        Functions that return outputs
    '''        
    # returns nearest neighbour for a given data point
    def predict(self, sample):
        # compute distances from all train encodings
        distances_from_all_train_encodings = [
            [self.dist_metric(self.train_encodings[i], sample), self.train_labels[i]] 
                for i in range(len(self.train_encodings))
        ]
        distances_from_all_train_encodings.sort()
        distances_from_all_train_encodings = np.array(distances_from_all_train_encodings)
        
        # extracting k nearest neighbours
        nearest_neighbour_labels = distances_from_all_train_encodings[:, 1][:self.k]
        
        # code for when we weigh by inverse of distance
        if self.weigh:
            unique_nearest_labels = np.unique(nearest_neighbour_labels)
            nearest_neighbour_distances = distances_from_all_train_encodings[:, 0][:self.k]
            
            # adding 1/(1+weight) for all
            nearest_weighted = {}
            for i in unique_nearest_labels:
                nearest_weighted[i] = 0
            
            for i in range(len(nearest_neighbour_distances)):
                nearest_weighted[nearest_neighbour_labels[i]] += 1/(1 + float(nearest_neighbour_distances[i]))
                
            # maximum statement from GPT
            return max(nearest_weighted, key=nearest_weighted.get)
            
        # fining most repeated neighbour - tie breaking is in alphabetical order
        unique_nearest_labels, label_frequencies = np.unique(nearest_neighbour_labels, return_counts=True)
        max_label_frequency = max(label_frequencies)
        for i in range(len(unique_nearest_labels)):
            if label_frequencies[i] == max_label_frequency:
                return unique_nearest_labels[i]
        
        return unique_nearest_labels[-1]
    
    # returns learning metrics
    def run(self):
        # Predict all labels
        predicted_labels = np.array([self.predict(sample) for sample in self.test_encodings])
        
        accuracy = accuracy_score(self.test_labels, predicted_labels)
        precision = precision_score(self.test_labels, predicted_labels, average='micro')
        recall = recall_score(self.test_labels, predicted_labels, average='micro')
        f1 = f1_score(self.test_labels, predicted_labels, average='micro')
        
        return accuracy, precision, recall, f1
            
    
    # init function to create object
    def __init__(self, k = 1, encoding = 1, dist_metric = 'euclidean', validation_setting = True, split = 20, weigh = False, dataset = None, train_data = None, test_data = None):
        # Checking whether inputs are valid:
        if (k <= 0 or not isinstance(k, int)):
            raise ValueError("k should be a positive natural number")
        if (not encoding in self.valid_encodings):
            raise ValueError("Encoding accepts 1 for resnet and 2 for ViT")
        if (not dist_metric in self.valid_distance_metrics):
            raise ValueError("Distance metric should belong to {}".format(valid_distance_metrics))
        if (split <= 0 or split >= 100):
            raise ValueError("Not a valid split")
        if (not isinstance(weigh, bool)):
            raise ValueError("Weigh is a boolean attribute")
            
        distance_metric_functions = {
            'euclidean': self.Euclidean,
            'manhattan': self.Manhattan,
            'cosine': self.Cosine,
            'minkowski3': self.Minkowski3,
            'minkowski4': self.Minkowski4,
            'chebyshev': self.Chebyshev
        }
        
        # setting varibles
        self.k = k
        self.encoding = encoding
        self.dist_metric = distance_metric_functions[dist_metric]
        self.weigh = weigh
        
        if validation_setting:
            # setting split
            self.split = split

            # Extracting features and labels from dataset
            self.data_encodings = dataset[:, encoding]
            self.data_labels = dataset[:, self.columns['label']]

            # making an array of arrays into a single array
            self.data_encodings = np.array([np.squeeze(nested_array) for nested_array in self.data_encodings])

            # splitting into train and validation
            self.split_data(split)
        else:
            self.train_labels = train_data[:, self.columns['label']]
            self.train_encodings = train_data[:, encoding]
            self.train_encodings = np.array([np.squeeze(nested_array) for nested_array in self.train_encodings])
            
            self.test_labels = test_data[:, self.columns['label']]
            self.test_encodings = test_data[:, encoding]
            self.test_encodings = np.array([np.squeeze(nested_array) for nested_array in self.test_encodings])
