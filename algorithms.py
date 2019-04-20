from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

class MachineLearningAlgorithm:
    """
    Custom class to group the most used methods of a specific algorithm.
    """
    def __init__(self, train_data, predict_data, model):
        """
        Class Constructor \n
        Parameters
        ----------
        train_data: (pd.DataFrame) data to feed the algorithm\n
        predict_data: (pd.DataFrame) data to be predicted\n
        """
        self.train_data = train_data.dropna(axis=0)
        self.predict_data = predict_data
        self.model = model
        
    def set_y(self, feature_name):
        self.y = self.train_data[feature_name]

    def set_X(self, feature_names):
        self.X = self.train_data[feature_names]

    def fit(self):
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=101)
        self.model.fit(self.X, self.y)

    def accuracy(self):
        return round(self.model.score(self.X_test, self.y_test) * 100)

    def predict(self, data_frame_to_be_predicted = None):
        """
        Predicts data \n
        Parameters
        ----------
        data_frame_to_be_predicted: [OPTIONAL] - (pd.dataFrame) - Defaults to self.X
        """
        if data_frame_to_be_predicted:
            _data_frame_to_be_predicted = data_frame_to_be_predicted
        else:
            _data_frame_to_be_predicted = self.X
        return self.model.predict(_data_frame_to_be_predicted)

class RandomForestClassifier_(MachineLearningAlgorithm):
    def __init__(self, train_data, predict_data):
        super().__init__(train_data, predict_data, RandomForestClassifier(n_estimators=100)) 
        
    def setup_model(self,
                    n_estimators='warn',
                    criterion="mse",
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.,
                    max_features="auto",
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.,
                    min_impurity_split=None,
                    bootstrap=True,
                    oob_score=False,
                    n_jobs=None,
                    random_state=None,
                    verbose=0,
                    warm_start=False):
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            criterion=criterion,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                                            max_features=max_features,
                                            max_leaf_nodes=max_leaf_nodes,
                                            min_impurity_decrease=min_impurity_decrease,
                                            min_impurity_split=min_impurity_split,
                                            bootstrap=bootstrap,
                                            oob_score=oob_score,
                                            n_jobs=n_jobs,
                                            random_state=random_state,
                                            verbose=verbose,
                                            warm_start=warm_start)

class DecisionTreeRegressor_(MachineLearningAlgorithm):
    def __init__(self, train_data, predict_data):
        super().__init__(train_data, predict_data, DecisionTreeRegressor(random_state=1))
    
            