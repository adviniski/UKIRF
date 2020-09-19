import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from recommenders.tf_models.gmf import GMF

from negative_sampling.arm import ARM
from negative_sampling.tf_idf import TF_IDF
from negative_sampling.cosine import Cosine
from negative_sampling.random import Random

from negative_sampling.rejection import TotalLimit, UniqueLimit, Q3Total, Q3Unique

from experimenter import Experimenter

if __name__ == '__main__':
    
    files = {}
    
    #files["ml-100k-gte.csv"] = {"learning_rate": 0.0005,"reg_rate": 0,"batch_size": 256,"num_neg_sample": 4,"num_factor": 32}
    #files["SMDI-400k_max500repeated.csv"] = {"learning_rate": 0.0005,"reg_rate": 0,"batch_size": 256,"num_neg_sample": 4,"num_factor": 32}
    #files["SMDI-400k_max200unique.csv"] = {"learning_rate": 0.0005,"reg_rate": 0,"batch_size": 256,"num_neg_sample": 4,"num_factor": 32}
    #files["SMDI-700k_original.csv"] = {"learning_rate": 0.001,"reg_rate": 0,"batch_size": 256,"num_neg_sample": 4,"num_factor": 32}

    files["ciaodvd-gte.csv"] = {"learning_rate": 0.0005,"reg_rate": 0.0,"batch_size": 256,"num_factor": 10}
    files["movietweetings-gte.csv"] = {"learning_rate": 0.0005,"reg_rate": 0.0,"batch_size": 256,"num_factor": 40}
    files["ml-100k-gte.csv"] = {"learning_rate": 0.0001,"reg_rate": 0.0,"batch_size": 256,"num_factor": 10}

    experimenter = Experimenter()
    experimenter.config_gpu()

    experimenter.addSamplingApproach(Cosine)
    experimenter.addSamplingApproach(TF_IDF)
    experimenter.addSamplingApproach(ARM)
    experimenter.addSamplingApproach(Random)

    experimenter.addMaxRejection(TotalLimit)
    experimenter.addMaxRejection(UniqueLimit)
    experimenter.addMaxRejection(Q3Total)
    experimenter.addMaxRejection(Q3Unique)

    experimenter.setModel(GMF)
    experimenter.setParameterFiles(files)
    experimenter.execute()
    
    """
    learning_rate = [0.001,0.005,0.0001,0.0005]
    latent_factors = [10,20,30,40,50]

    for l_r in learning_rate:
        for factors in latent_factors:
            files = {}
            files["ciaodvd-gte.csv"] = {"learning_rate": l_r,"reg_rate": 0.0,"batch_size": 256,"num_factor": factors}
            files["movietweetings-gte.csv"] = {"learning_rate": l_r,"reg_rate": 0.0,"batch_size": 256,"num_factor": factors}
            files["ml-100k-gte.csv"] = {"learning_rate": l_r,"reg_rate": 0.0,"batch_size": 256,"num_factor": factors}
            experimenter.setModel(GMF)
            experimenter.setParameterFiles(files)
            experimenter.execute()"""