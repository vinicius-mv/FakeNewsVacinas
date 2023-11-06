import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from src.core import PredictionModel


class ChartGenerator(object):

    @staticmethod
    def get_dataset_classes_proportion(df: pd.DataFrame):
        df['desinformação'] = np.where(df.is_missinginfo == 1, 'Sim', 'Não')
        ax = sns.countplot(data=df, x='desinformação', order=df['desinformação'].value_counts().index)
        plt.xlabel('Desinformação?')
        plt.ylabel('Total de Mensagens')
        plt.show()
