__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

class SpamProcessing:

    """
    This class carries out the necessary instructions for processing the spam dataset.
    """

    def __init__(self, spam_path):
        self.spam_path = spam_path
        print("SpamProcessing object created!")

    def preprocess_spam(self):

        """
        Preprocess the raw spam data.
        """

        print('[ INFO ]: Preprocessing spam data...')

        # Rename headers of data frame
        spam_data = pd.read_csv(self.spam_path, header=None)
        spam_data.columns = ['word_freq_make','word_freq_address','word_freq_all','word_freq_3d','word_freq_our',
        'word_freq_over','word_freq_remove','word_freq_internet','word_freq_order','word_freq_mail',
        'word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses',
        'word_freq_free','word_freq_business','word_freq_email','word_freq_you','word_freq_credit',
        'word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp','word_freq_hpl',
        'word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857',
        'word_freq_data','word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts',
        'word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original',
        'word_freq_project','word_freq_re','word_freq_edu','word_freq_table','word_freq_conference','char_freq_;',
        'char_freq_(','char_freq_[','char_freq_!','char_freq_$','char_freq_#','capital_run_length_average',
        'capital_run_length_longest','capital_run_length_total','class']

        # Return the list of features
        df_columns = [spam_data.columns[j] for j in range(len(spam_data.columns)) if spam_data.columns[j] != 'class']

        # Place classes into list
        classes = spam_data['class'].unique().tolist()

        return spam_data, df_columns, classes

    def spam_runner(self):

        """
        Execute the program runner over the spam dataset.
        """

        print('[ INFO ]: Initializing the spam program runner...')

        data, features, classes = self.preprocess_spam()
        spam = alg()
        random_data = spam.random_feature_sample(data, 0.02)
        selected_features, selected_clusters, basePerformance = spam.stepwise_forward_selection(random_data, features, len(classes))
        spam_clusters, spam_performance = spam.recompute_stepwise_with_population(data, selected_features, len(classes))

        return selected_features, spam_clusters, basePerformance, spam_performance
